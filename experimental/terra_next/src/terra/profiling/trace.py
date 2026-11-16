"""Read bounded Chrome JSON and retain IO markers plus actual GPU activities.

Chrome timestamps are microseconds even when displayTimeUnit is ms. Metadata
is joined by the unique record_function label, never by timestamp proximity.
Input is bounded by *expanded* bytes, including gzip input; short capture
windows are intentional. The standard JSON parser still needs additional
memory for its object tree. No original trace is rewritten or discarded.
"""
from dataclasses import dataclass,field
from decimal import Decimal
from collections import defaultdict
from bisect import bisect_right
import gzip,json,math,re
from pathlib import Path

_NCCL=re.compile(r'nccl|hccl|hcom',re.I)
_RUNTIME_CATEGORIES={'cuda_runtime','cuda_driver','cuda_api','gpu_runtime','gpu_driver','npu_runtime'}


@dataclass
class TraceBundle:
    path: str
    metadata: dict
    events: list
    discarded_events: int=0
    malformed_events: int=0
    device_adapter: dict=field(default_factory=dict)
    dispatch_flows: list=field(default_factory=list)


def metadata_path(trace):
    trace=Path(trace);name=trace.name
    stem=name[:-8] if name.endswith('.json.gz') else name[:-5] if name.endswith('.json') else name
    candidates=[trace.with_name(stem+'.metadata.json'),trace.with_name(name+'.metadata.json')]
    return next((p for p in candidates if p.is_file()),candidates[0])


def discover_traces(inputs):
    if isinstance(inputs,(str,Path)):inputs=[inputs]
    paths=[]
    for value in inputs:
        path=Path(value)
        if path.is_dir():
            paths.extend(p for p in path.rglob('*') if p.is_file() and
                         (p.name.endswith('.json.gz') or p.name.endswith('.trace.json') or p.name=='trace.json'))
        elif path.is_file():paths.append(path)
        else:raise FileNotFoundError(path)
    paths=sorted(set(p.resolve() for p in paths))
    if not paths:raise ValueError('No Chrome trace files were found')
    return paths


def _cpu_kind(name,metadata=None):
    if name=='terra/io/load_pt':return 'cpu_load'
    if name=='terra/io/frame_acquire':return 'cpu_acquire'
    if name=='terra/io/map_pt':return 'cpu_map'
    if name=='terra/io/region_materialize':return 'cpu_materialize'
    if name in ('terra/io/buffer_allocate','terra/io/pad_initialize','terra/io/batch_stack',
                'terra/io/pad_allocate','terra/io/pad_assemble'):
        return 'cpu_buffer'
    if name=='terra/io/debug_checks':return 'cpu_debug_checks'
    if name=='terra/io/cpu_prepare':
        operation=(metadata or {}).get('operation')
        if operation=='validate_storage_layout':return 'cpu_metadata'
        if operation in ('crop_dtype_finite_mask_contiguous','cache_local_valid_mask',
                         'validate_local_mask','cast_validate_mask'):
            return 'cpu_debug_checks'
        if operation=='restore_model_input_layout':return 'cpu_buffer'
        # Unspecified older ranges cannot safely be relabeled as optional
        # checks or normal buffer work. Preserve them as legacy preparation.
        return 'cpu_prepare'
    if name=='terra/io/manifest':return 'cpu_manifest'
    if name=='terra/io/h2d_submit':return 'cpu_h2d_submit'
    if name=='terra/data_wait':return 'cpu_data_wait'
    if name.startswith('terra/io/') and 'fake' in name:return 'cpu_fake_generate'
    if name.startswith(('terra/train/','terra/model/','terra/comm/','terra/profile/','terra/reducer/')):return 'cpu_phase'
    return None


def _gpu_kind(event):
    category=str(event.get('cat','')).lower();name=str(event.get('name',''))
    if category in _RUNTIME_CATEGORIES:return None
    if category in ('kernel','gpu_kernel','npu_kernel'):
        if not name.strip():return 'gpu_unknown'
        # Non-NCCL kernel means device kernel work (including packing/indexing),
        # not a claim that every kernel performs useful FLOPS.
        return 'gpu_comm' if _NCCL.search(name) else 'gpu_compute'
    if category in ('gpu_memcpy','gpu_memcpy_async','npu_memcpy'):
        text=name.lower().replace(' ','')
        return 'gpu_h2d' if any(v in text for v in ('htod','h2d','hosttodevice')) else 'gpu_copy_other'
    if category=='gpu_memset':return 'gpu_copy_other'
    if category.startswith('gpu') and category not in ('gpu_user_annotation','gpu_user_annotation_external'):
        return 'gpu_unknown'
    return None


def _identity(args,namespace):
    for key,value in args.items():
        normalized=str(key).lower().replace('_','').replace(' ','')
        if normalized in ({'correlation','correlationid'} if namespace=='correlation' else {'externalid'}):
            if isinstance(value,(int,float,str)):return str(value)
    return None


def _chrome_dispatch(events,rows):
    """Strict CUPTI/Chrome identities; no nearest-timestamp association."""
    runtime_ids=defaultdict(list);runtime_external=defaultdict(list);operator_external=defaultdict(list)
    hosts=defaultdict(list);raw_by_event={e['event_id']:rows[int(e['event_id'].split(':')[1])] for e in events}
    for event in events:
        if event['kind'] not in ('cpu_op','cpu_runtime'):continue
        row=raw_by_event[event['event_id']];args=row.get('args',{})
        if not isinstance(args,dict):args={}
        hosts[event['pid'],event['tid']].append(event)
        correlation,external=_identity(args,'correlation'),_identity(args,'external')
        if event['kind']=='cpu_runtime':
            if correlation is not None:runtime_ids[correlation].append(event)
            if external is not None:runtime_external[external].append(event)
        elif external is not None:operator_external[external].append(event)
    host_index={}
    for key,values in hosts.items():
        values.sort(key=lambda e:e['start_ns']);maximum=[];high=-float('inf')
        for event in values:high=max(high,event['end_ns']);maximum.append(high)
        host_index[key]=values,[e['start_ns'] for e in values],maximum
    def containing(pid,tid,timestamp,kind=None):
        indexed=host_index.get((pid,tid))
        if indexed is None:return None
        values,starts,maximum=indexed;i=bisect_right(starts,timestamp)-1;candidates=[]
        while i>=0 and maximum[i]>timestamp:
            event=values[i]
            if event['start_ns']<=timestamp<event['end_ns'] and (kind is None or event['kind']==kind):candidates.append(event)
            i-=1
        return min(candidates,key=lambda e:e['end_ns']-e['start_ns']) if candidates else None
    starts=defaultdict(list);finishes={}
    for row in rows:
        if isinstance(row,dict) and row.get('ph')=='s':starts[row.get('cat'),row.get('id')].append(row)
    for row in rows:
        if not isinstance(row,dict) or row.get('ph')!='f':continue
        source=starts.get((row.get('cat'),row.get('id')),[])
        if len(source)!=1:continue
        try:key=row.get('pid'),row.get('tid'),float(row['ts'])*1000
        except (KeyError,TypeError,ValueError):continue
        finishes[key]=source[0] if key not in finishes else None
    result=[]
    for target in events:
        if not target['kind'].startswith('gpu_') or target['metadata'].get('device_adapter')=='ascend_native_v1':continue
        row=raw_by_event[target['event_id']];args=row.get('args',{})
        if not isinstance(args,dict):args={}
        flow=finishes.get((target['pid'],target['tid'],target['start_ns']))
        runtime=operator=source=None;evidence='unknown';timestamp=None;raw_flow_id=None
        if flow is not None:
            try:timestamp=float(flow['ts'])*1000
            except (KeyError,TypeError,ValueError):timestamp=None
            if timestamp is not None:
                runtime=containing(flow.get('pid'),flow.get('tid'),timestamp,'cpu_runtime')
                operator=containing(flow.get('pid'),flow.get('tid'),timestamp,'cpu_op')
                source=runtime or operator
                if source:evidence='chrome_flow_exact';raw_flow_id=flow.get('id')
        if source is None:
            correlation,external=_identity(args,'correlation'),_identity(args,'external')
            candidates=runtime_ids.get(correlation,[]) if correlation is not None else []
            if len(candidates)==1:
                runtime=source=candidates[0];evidence='cuda_correlation_exact'
            else:
                candidates=runtime_external.get(external,[]) if external is not None else []
                if len(candidates)==1:runtime=source=candidates[0];evidence='cuda_external_id_exact'
                else:
                    candidates=operator_external.get(external,[]) if external is not None else []
                    if len(candidates)==1:operator=source=candidates[0];evidence='cuda_external_id_exact'
            timestamp=source['start_ns'] if source else None
        if source is None:
            target['metadata'].update(dispatch_exact=False,dispatch_evidence='unknown',source_op_name=None)
            continue
        operator_evidence=('exact_external_id' if source is operator and evidence=='cuda_external_id_exact' else
                           'raw_flow_point_same_thread_containment' if operator is not None else None)
        if operator is None and runtime is not None:
            operator=containing(runtime['pid'],runtime['tid'],runtime['start_ns'],'cpu_op')
            if operator:operator_evidence='same_thread_interval_containment'
        details={'dispatch_exact':True,'dispatch_evidence':evidence,
                 'source_cpu_pid':source['pid'],'source_cpu_tid':source['tid'],
                 'source_cpu_timestamp_ns':timestamp,'source_op_event_id':operator['event_id'] if operator else None,
                 'source_op_name':operator['name'] if operator else None,
                 'runtime_event_id':runtime['event_id'] if runtime else None}
        if operator:details.update(source_op_start_ns=operator['start_ns'],source_op_end_ns=operator['end_ns'])
        if runtime:details.update(launch_name=runtime['name'],launch_start_ns=runtime['start_ns'],launch_end_ns=runtime['end_ns'])
        target['metadata'].update(details)
        result.append({'id':'chrome:'+target['event_id'],'flow_id':raw_flow_id,'exact':True,'evidence':evidence,
                       'source_event_id':source['event_id'],'target_event_id':target['event_id'],
                       'source_ts_ns':timestamp,'target_ts_ns':target['start_ns'],
                       'source_pid':source['pid'],'source_tid':source['tid'],
                       'target_device':target['device'],'target_stream':target['stream'],
                       'target_device_source':'args.device' if 'device' in args else
                                              'args.Device' if 'Device' in args else 'pid_fallback',
                       'source_name':source['name'],'target_name':target['name'],
                       'source_op_event_id':operator['event_id'] if operator else None,
                       'source_op_name':operator['name'] if operator else None,
                       'operator_evidence':operator_evidence,
                       'source_timestamp_semantics':'raw_chrome_flow_point' if evidence=='chrome_flow_exact' else
                                                    'runtime_call_start' if runtime else 'operator_interval_start',
                       'runtime_event_id':runtime['event_id'] if runtime else None})
    return result


def read_trace(path,*,max_input_mb=512):
    path=Path(path)
    if isinstance(max_input_mb,bool) or max_input_mb<=0:raise ValueError('max_input_mb must be positive')
    limit=int(max_input_mb*1024*1024)
    opener=gzip.open if path.name.endswith('.gz') else open
    with opener(path,'rb') as handle:raw=handle.read(limit+1)
    if len(raw)>limit:raise ValueError(f'Trace exceeds expanded input limit of {max_input_mb} MiB: {path}')
    document=json.loads(raw)
    rows=document.get('traceEvents') if isinstance(document,dict) else document
    if not isinstance(rows,list):raise ValueError(f'Expected Chrome traceEvents array: {path}')
    from .native_npu import is_native_npu,adapt_native_npu,time_ns,_plain,is_profiler_envelope
    native=is_native_npu(rows)
    if native:
        # Native timestamps may exceed binary float sub-microsecond precision.
        # CUDA retains its existing JSON/timestamp conversion behavior.
        document=json.loads(raw,parse_float=Decimal)
        rows=document.get('traceEvents') if isinstance(document,dict) else document
    sidecar=metadata_path(path)
    metadata=json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else {}
    if not isinstance(metadata,dict):raise ValueError('Trace metadata must be a mapping')
    annotations=metadata.get('annotations',{})
    if isinstance(annotations,list):annotations={item['label']:item for item in annotations}
    if not isinstance(annotations,dict):raise ValueError('annotations must be a mapping')
    adapted=adapt_native_npu(rows,metadata) if native else None
    events=list(adapted['events']) if adapted else []
    discarded=0;malformed=adapted['malformed_events'] if adapted else 0
    for index,row in enumerate(rows):
        if adapted and index in adapted['skip_indices']:
            if index not in adapted['accepted_indices']:discarded+=1
            continue
        if not isinstance(row,dict) or row.get('ph')!='X':discarded+=1;continue
        # Kineto may repeat the same record_function label as a GPU range.
        # It is an annotation, not another CPU call or a device activity.
        if str(row.get('cat','')).lower().startswith('gpu_user_annotation'):
            discarded+=1;continue
        label=str(row.get('name',''));annotation=annotations.get(label,{})
        if is_profiler_envelope(label):discarded+=1;continue
        base=annotation.get('name',label.split('#',1)[0])
        kind=_gpu_kind(row)
        if kind is None:
            category=str(row.get('cat','')).lower()
            if category not in ('user_annotation','cpu_op','python_function') and category not in _RUNTIME_CATEGORIES and not label.startswith('terra/'):
                discarded+=1;continue
            kind=_cpu_kind(base,annotation.get('metadata',{}))
            if kind is None and category=='cpu_op':kind='cpu_op'
            elif kind is None and category in _RUNTIME_CATEGORIES:kind='cpu_runtime'
        if kind is None:discarded+=1;continue
        try:
            if native:
                start=time_ns(row['ts']);duration=time_ns(row['dur'])
            else:
                start=float(row['ts'])*1000;duration=float(row['dur'])*1000
            if not math.isfinite(start) or not math.isfinite(duration) or duration<0:raise ValueError()
        except (KeyError,TypeError,ValueError):malformed+=1;continue
        args=row.get('args',{})
        if not isinstance(args,dict):args={}
        details=dict(annotation.get('metadata',{}))
        if kind in ('cpu_op','cpu_runtime'):details.update(raw_category=str(row.get('cat','')),raw_args=_plain(args))
        events.append({'event_id':'row:'+str(index),'kind':kind,'name':base,'label':label,'start_ns':start,'end_ns':start+duration,
                       'pid':row.get('pid'),'tid':row.get('tid'),
                       'device':args.get('device',args.get('Device',row.get('pid') if kind.startswith('gpu_') else None)),
                       'stream':args.get('stream',row.get('tid') if kind.startswith('gpu_') else None),
                       'correlation':args.get('correlation',args.get('correlation id')),
                       'metadata':details})
    if not events:raise ValueError(f'Trace has no recognized IO markers or GPU activities: {path}')
    dispatch=list(adapted['dispatch_flows']) if adapted else []
    dispatch.extend(_chrome_dispatch(events,rows))
    return TraceBundle(str(path.resolve()),metadata,events,discarded,malformed,
                       device_adapter=adapted['device_adapter'] if adapted else {},dispatch_flows=dispatch)
