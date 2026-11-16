"""Run actual TERRA TokenLayout/TokenRoute on four local Gloo ranks.

This demonstrates exact routing and its gradient, not training performance.
The provided runtime root must contain the generic experimental terra package.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def worker(args):
    sys.path.insert(0, str(args.runtime_root.resolve()))
    import torch
    import torch.distributed as dist
    from terra.core.dmp.mesh import DMPMesh
    from terra.core.FMP.layout import TokenLayout, TokenRoute
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(args.store).resolve().as_uri(),
                            rank=args.worker, world_size=4,
                            timeout=datetime.timedelta(seconds=40))
    mesh = None
    try:
        mesh = DMPMesh(wp_size=4, sp_size=1, tp_size=1)
        records = []
        for assignment in ('contiguous', 'column_contiguous'):
            plain = TokenLayout(mesh, 16, 32, 4, shift=0,
                                window_assignment=assignment)
            shifted = TokenLayout(mesh, 16, 32, 4, shift=2,
                                  window_assignment=assignment)
            route = TokenRoute(plain, shifted,
                               restore_implementation='inverse_gather',
                               transport_implementation='p2p',
                               window_shift='generic')
            source_ids = plain.token_ids(1)
            target_ids = shifted.token_ids(1)
            x = source_ids.to(torch.float32).reshape(plain.shape(1, 1)).requires_grad_()
            y = route(x)
            expected = target_ids.to(torch.float32).reshape_as(y)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            weights = (target_ids.remainder(7)+1).to(torch.float32).reshape_as(y)
            y.backward(weights)
            expected_gradient = (source_ids.remainder(7)+1).to(torch.float32).reshape_as(x)
            torch.testing.assert_close(x.grad, expected_gradient, rtol=0, atol=0)
            audit = route.window_shift_audit()
            assert audit['actual_transport'] == 'p2p'
            counts = list(route._plans[1][2])
            records.append({'assignment': assignment, 'forward_exact': True,
                            'gradient_exact': True, 'send_tokens_by_peer': counts,
                            'remote_tokens': sum(counts)-counts[args.worker],
                            'window_ids': plain.window_ids(args.worker).tolist()})
            dist.barrier()
        evidence = {'rank':args.worker, 'status':'PASS', 'records':records,
                    'runtime_module':str(Path(sys.modules['terra.core.FMP.layout'].__file__).resolve())}
        (args.output/('rank_%05d.json'%args.worker)).write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    finally:
        if mesh is not None:
            mesh.close()
        dist.destroy_process_group()


def launch(args):
    args.runtime_root = args.runtime_root.resolve()
    if not (args.runtime_root/'terra/core/FMP/layout.py').is_file():
        raise ValueError('--runtime-root must contain terra/core/FMP/layout.py')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory; evidence is never overwritten')
    args.output.mkdir(parents=True)
    store = args.output/'gloo_store'
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                       OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',USE_LIBUV='0')
    processes=[];streams=[];codes=[]
    deadline=time.monotonic()+90
    try:
        for rank in range(4):
            out=(args.output/('worker_%d.stdout.log'%rank)).open('w',encoding='utf-8')
            err=(args.output/('worker_%d.stderr.log'%rank)).open('w',encoding='utf-8')
            streams.extend((out,err))
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--runtime-root',str(args.runtime_root),
                     '--output',str(args.output),'--worker',str(rank),'--store',str(store)]
            processes.append(subprocess.Popen(command,env=environment,stdout=out,stderr=err,
                                               start_new_session=os.name!='nt'))
        for p in processes:
            codes.append(p.wait(timeout=max(1,deadline-time.monotonic())))
        if any(codes):
            raise RuntimeError('A real Gloo worker failed; inspect worker logs')
        ranks=[json.loads((args.output/('rank_%05d.json'%rank)).read_text(encoding='utf-8')) for rank in range(4)]
        for rank,v in enumerate(ranks):
            assert v['rank']==rank and v['status']=='PASS'
            assert Path(v['runtime_module']).is_relative_to(args.runtime_root)
        summaries={}
        for i,assignment in enumerate(('contiguous','column_contiguous')):
            matrix=[v['records'][i]['send_tokens_by_peer'] for v in ranks]
            total=sum(v['records'][i]['remote_tokens'] for v in ranks)
            assert total=={'contiguous':256,'column_contiguous':128}[assignment]
            summaries[assignment]={'send_matrix':matrix,'remote_tokens':total,
                                    'forward_exact':True,'weighted_gradient_exact':True}
        receipt={'format':'terra_real_p2p_layout_demo_v1','status':'PASS','world_size':4,
                 'backend':'gloo','device':'cpu','dtype':'float32','WP':4,'SP':1,'TP':1,
                 'window_grid':[4,8],'window_size':4,'shift':[2,2],
                 'real_processes':True,'mocked_transport':False,'performance_measurement':False,
                 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'layout_module_sha256':hashlib.sha256((args.runtime_root/'terra/core/FMP/layout.py').read_bytes()).hexdigest(),
                 'worker_pids':[p.pid for p in processes], 'worker_exit_codes':codes,
                 'summaries':summaries}
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':receipt['status'],'real_ranks':4,'forward_and_weighted_gradient_exact':True,
                          'remote_tokens':{k:v['remote_tokens'] for k,v in summaries.items()}}))
    finally:
        for p in processes:
            if p.poll() is None:p.kill()
        for p in processes:p.wait(timeout=5)
        for stream in streams:stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    p.add_argument('--store',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker is None:launch(args)
    else:worker(args)


if __name__=='__main__':
    main()
