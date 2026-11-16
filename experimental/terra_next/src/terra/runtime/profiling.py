"""Optional process-local PyTorch profiling, shared by all training entries.

CPU ranges measure application wall time, not physical disk service time.
No synchronization is inserted inside measured operations. Recorded updates
optionally drain the device at their end to give a complete-step boundary.
PDF work happens after training and after the collector has stopped.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import copy,itertools,json,math,os,shutil,socket,tempfile,time,uuid,warnings

import torch
import torch.distributed as dist
from .accelerator import as_device, synchronize

_ACTIVE=ContextVar('terra_active_profiler',default=None)
_METADATA=ContextVar('terra_profile_metadata',default={})
_UPDATE_ACTIVE=ContextVar('terra_profile_update_active',default=False)
_PROCESS_ACTIVE=None


def validate_profiling_config(config):
    if config is None:config={}
    if not isinstance(config,dict):raise ValueError('profiling must be a mapping')
    allowed={'enabled','capture_io','export_pdf','ranks','capture_startup','capture_first_step',
             'schedule','record_shapes','profile_memory','with_stack','max_plot_ranks','environment',
             'complete_updates'}
    if set(config)-allowed:raise ValueError('Unknown profiling keys: '+str(sorted(set(config)-allowed)))
    result={'enabled':False,'capture_io':True,'export_pdf':True,'ranks':'all',
            'capture_startup':True,'capture_first_step':True,'record_shapes':False,
            'profile_memory':False,'with_stack':False,'max_plot_ranks':2,'complete_updates':True}
    result.update(copy.deepcopy(config))
    for key in ('enabled','capture_io','export_pdf','capture_startup','capture_first_step','record_shapes','profile_memory','with_stack','complete_updates'):
        if not isinstance(result[key],bool):raise ValueError('profiling.'+key+' must be boolean')
    schedule={'wait':1,'warmup':1,'active':3,'repeat':1,'skip_first':0}
    given=config.get('schedule',{})
    if not isinstance(given,dict) or set(given)-set(schedule):raise ValueError('Invalid profiling.schedule')
    schedule.update(given)
    for key,value in schedule.items():
        minimum=1 if key in ('active','repeat') else 0
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError('Invalid profiling.schedule.'+key)
    result['schedule']=schedule
    ranks=result['ranks']
    if ranks!='all' and (not isinstance(ranks,(list,tuple)) or not ranks or any(isinstance(r,bool) or not isinstance(r,int) or r<0 for r in ranks)):
        raise ValueError('profiling.ranks must be all or a nonempty list of global ranks')
    if isinstance(result['max_plot_ranks'],bool) or not isinstance(result['max_plot_ranks'],int) or result['max_plot_ranks']<1:
        raise ValueError('profiling.max_plot_ranks must be positive')
    environment={'enabled':True,'interval_s':1.0,'baseline_samples':3}
    given=config.get('environment',{})
    if not isinstance(given,dict) or set(given)-set(environment):
        raise ValueError('Invalid profiling.environment')
    environment.update(given)
    if not isinstance(environment['enabled'],bool):raise ValueError('profiling.environment.enabled must be boolean')
    interval=environment['interval_s']
    if isinstance(interval,bool) or not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval<0.2:
        raise ValueError('profiling.environment.interval_s must be at least 0.2 seconds')
    count=environment['baseline_samples']
    if isinstance(count,bool) or not isinstance(count,int) or not 2<=count<=120:
        raise ValueError('profiling.environment.baseline_samples must be between 2 and 120')
    result['environment']=environment
    return result


def is_recording():
    controller=_ACTIVE.get() or _PROCESS_ACTIVE
    return bool(controller is not None and controller.is_recording())


@contextmanager
def profile_range(name,**metadata):
    controller=_ACTIVE.get() or _PROCESS_ACTIVE
    if controller is None:
        yield
    else:
        with controller.range(name,**metadata):yield


@contextmanager
def profile_update(device):
    """Mark one real training update; only captured updates may synchronize.

    The terminal drain is part of the update, never a fence between backward,
    gradient collectives or optimizer kernels. Existing outer update markers
    keep ownership of their own boundary.
    """
    controller=_ACTIVE.get() or _PROCESS_ACTIVE
    if controller is None or not controller.is_recording() or _UPDATE_ACTIVE.get():
        yield;return
    complete=controller.config['complete_updates']
    with controller.range('terra/profile/update',device_complete=complete,
                          synchronization_policy='profiled_update_end' if complete else 'none'):
        yield
        if complete:
            with controller.range('terra/profile/final_drain',device=str(device)):
                synchronize(device)


class ProfilerRuntime:
    def __init__(self,config=None,*,device,output_dir,stage,rank=0,world_size=1,metadata=None):
        global _PROCESS_ACTIVE
        self.config=validate_profiling_config(config)
        self.requested=self.config['enabled']
        self.rank,self.world_size=rank,world_size
        ranks=self.config['ranks']
        if ranks!='all' and any(r>=world_size for r in ranks):raise ValueError('profiling.ranks outside WORLD')
        self.enabled=self.requested and (ranks=='all' or rank in ranks)
        self.device=as_device(device)
        self.output=Path(output_dir)/'profiling'
        self.run_id=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]
        if self.requested and dist.is_available() and dist.is_initialized():
            job=[self.run_id if rank==0 else None]
            dist.broadcast_object_list(job,src=0)
            self.run_id=job[0]
        self.hostname=socket.gethostname()
        self.stage=stage
        self.metadata=dict(metadata or {})
        self.base={'global_step':0,'epoch':0,'phase':'startup'}
        self._prof=None;self._phase=None;self._token=None;self._cycle=0;self._events=itertools.count(1)
        self._annotations={};self._traces=[];self._steady_steps=0;self._first_done=False
        self._suspended=False;self._closed=False;self._finishing=False
        self._environment_monitor=None;self._environment_temp=None;self.environment_file=None
        self._capture_started_utc=None
        self._native_temp=None;self._native_capture=None
        if self.enabled:
            if _PROCESS_ACTIVE is not None:raise RuntimeError('Only one active profiling runtime per process is supported')
            self._token=_ACTIVE.set(self);_PROCESS_ACTIVE=self
        if self.requested and self.config['environment']['enabled']:
            self._start_environment()
            if dist.is_available() and dist.is_initialized():dist.barrier()

    def _start_environment(self):
        """One sampler per host, or attach an outer pre-launch harness.

        Baseline collection precedes the first PyTorch capture. Monitoring is
        an independent process and never inserts a CUDA synchronization.
        """
        participant={'pid':os.getpid(),'rank':self.rank,'hostname':self.hostname,
                     'device':str(self.device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'ascend_visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                     'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                     'threads':torch.get_num_threads()}
        if self.device.type=='cuda':
            try:
                value=getattr(torch.cuda.get_device_properties(self.device),'uuid',None)
                if value is not None:participant['gpu_uuid']=str(value)
            except Exception as exc:participant['gpu_identity_error']=type(exc).__name__
        participants=[participant]
        if dist.is_available() and dist.is_initialized():
            participants=[None]*dist.get_world_size()
            dist.all_gather_object(participants,participant)
        local=[p for p in participants if p['hostname']==self.hostname]
        external=os.environ.get('TERRA_PROFILE_ENVIRONMENT_FILE')
        self.environment_file=external or str(self.output/('environment.'+self.hostname+'.'+self.run_id+'.json'))
        try:
            if external:
                registration=Path(self.environment_file+f'.rank{self.rank}.json')
                registration.parent.mkdir(parents=True,exist_ok=True)
                registration.write_text(json.dumps(participant,indent=2)+'\n',encoding='utf-8')
            elif self.rank==min(p['rank'] for p in local):
                from terra.profiling.environment import start_monitor
                # The engine still owns creation/refusal of the training output
                # directory. Keep the baseline outside it until close().
                self._environment_temp=tempfile.TemporaryDirectory(prefix='terra-profile-environment-')
                self._environment_monitor=start_monitor(Path(self._environment_temp.name)/'environment.json',participants=local,
                    data_paths=self.metadata.get('data_paths',[]),
                    interval_s=self.config['environment']['interval_s'],
                    baseline_samples=self.config['environment']['baseline_samples'])
        except Exception as exc:
            # Missing telemetry must be explicit, but must not stop training.
            warnings.warn('Environment telemetry unavailable: '+type(exc).__name__+': '+str(exc))
            self.metadata['environment_error']=type(exc).__name__+': '+str(exc)

    def is_recording(self):
        if not self.enabled or self._prof is None:return False
        ProfilerAction=self._profiler_api().ProfilerAction
        return self._prof.current_action in (ProfilerAction.RECORD,ProfilerAction.RECORD_AND_SAVE)

    def _profiler_api(self):
        if self.device.type=='npu':
            import torch_npu.profiler
            return torch_npu.profiler
        return torch.profiler

    @contextmanager
    def range(self,name,**metadata):
        inherited=dict(_METADATA.get());inherited.update(metadata)
        token=_METADATA.set(inherited)
        update_token=None
        try:
            if not self.is_recording() or (name.startswith('terra/io/') and not self.config['capture_io']):
                yield;return
            if name=='terra/profile/update':update_token=_UPDATE_ACTIVE.set(True)
            label=f'{name}#{next(self._events):08d}'
            details={**self.base,**inherited}
            start=time.perf_counter_ns()
            with torch.profiler.record_function(label):yield
            self._annotations[label]={'name':name,'metadata':details,'wall_duration_ns':time.perf_counter_ns()-start}
        finally:
            if update_token is not None:_UPDATE_ACTIVE.reset(update_token)
            _METADATA.reset(token)

    def _native_profiler_options(self,api):
        """Enable the native Ascend task/flow export, without HBM claims.

        Initializing the native trace handler establishes the CANN raw path.
        Its callback is intentionally unused: _export owns the existing Terra
        trace/sidecar names. Scratch stays outside the engine-owned output
        directory, which may not exist yet during startup capture.
        """
        required=('_ExperimentalConfig','ProfilerLevel','AiCMetrics','ExportType',
                  'tensorboard_trace_handler')
        missing=[name for name in required if not hasattr(api,name)]
        if missing:
            raise RuntimeError('NPU native profiling requires torch_npu.profiler APIs: '+', '.join(missing))
        if self._native_temp is None:
            self._native_temp=tempfile.TemporaryDirectory(prefix='terra-native-profile-')
        api.tensorboard_trace_handler(self._native_temp.name,
            worker_name=f'rank{self.rank:05d}',analyse_flag=False)
        experimental=api._ExperimentalConfig(profiler_level=api.ProfilerLevel.Level1,
            aic_metrics=api.AiCMetrics.AiCoreNone,l2_cache=False,record_op_args=False,
            export_type=api.ExportType.Text)
        self._native_capture={'format':'ascend_native','profiler_level':'Level1',
            'aic_metrics':'AiCoreNone','export_type':'Text',
            'hbm_counters_available':False,
            'raw_trace':'unmodified export_chrome_trace JSON',
            'scratch_retained':False}
        return {'experimental_config':experimental}

    def _start(self,phase):
        if not self.enabled or self._prof is not None:return
        self._phase=phase;self.base['phase']=phase
        self._capture_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        api=self._profiler_api()
        activities=[api.ProfilerActivity.CPU]
        if self.device.type=='cuda':activities.append(api.ProfilerActivity.CUDA)
        if self.device.type=='npu':activities.append(api.ProfilerActivity.NPU)
        kwargs={key:self.config[key] for key in ('record_shapes','profile_memory','with_stack')}
        if self.device.type=='npu':kwargs.update(self._native_profiler_options(api))
        if phase=='train':
            original=api.schedule(**self.config['schedule'])
            offset=self._steady_steps
            schedule=lambda index:original(index+offset)
        else:schedule=None
        self._prof=api.profile(activities=activities,schedule=schedule,
            on_trace_ready=self._export,**kwargs)
        self._prof.__enter__()

    def _export(self,prof):
        if not self._annotations:return
        if not self.output.parent.is_dir():return
        self.output.mkdir(parents=True,exist_ok=True)
        self._cycle+=1
        stem=f'{self.hostname}.rank{self.rank:04d}.{self.run_id}.{self._phase}.cycle{self._cycle:03d}.trace'
        path=self.output/(stem+('.json' if self.device.type=='npu' else '.json.gz'))
        prof.export_chrome_trace(str(path))
        sidecar=self.output/(stem+'.metadata.json')
        record={'format':'terra_pytorch_profile_v1','run_id':self.run_id,'hostname':self.hostname,
                'global_rank':self.rank,'world_size':self.world_size,'stage':self.stage,
                'device':str(self.device),'capture_phase':self._phase,'cycle':self._cycle,
                'torch_version':torch.__version__,'cuda_build':torch.version.cuda,
                'accelerator_backend':self.device.type,
                'native_capture':self._native_capture,
                'environment_file':self.environment_file,
                'capture_started_utc':self._capture_started_utc,
                'capture_exported_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'configuration':self.config,'metadata':self.metadata,
                'annotations':self._annotations,'trace_file':str(path),
                'io_semantics':'torch.load application wall time includes cached file access, deserialization and allocation; not physical disk service time',
                'clock_semantics':'CPU/device timestamps come from the same profiler trace; no cross-host absolute clock alignment assumed',
                'update_semantics':'recorded train_update ends with a device drain when complete_updates=true; internal phases remain asynchronous'}
        sidecar.write_text(json.dumps(record,indent=2)+'\n')
        self._traces.append(str(path));self._annotations={}

    def _stop(self):
        if self._prof is None:return
        prof=self._prof
        try:prof.__exit__(None,None,None)
        finally:self._prof=None

    def start_startup(self):
        if self.config['capture_startup']:self._start('startup')

    def start_training(self,metadata=None):
        self.metadata.update(metadata or {})
        self._stop()
        if self.config['capture_first_step'] and not self._first_done:self._start('first_use')
        else:self._start('train')

    def set_step(self,global_step,epoch=None):
        self.base['global_step']=int(global_step)
        if epoch is not None:self.base['epoch']=int(epoch)

    def iter_loader(self,loader):
        iterator=iter(loader)
        while True:
            with profile_range('terra/data_wait'):
                try:batch=next(iterator)
                except StopIteration:return
            yield batch

    def after_step(self):
        if not self.enabled:return
        if self._phase=='first_use':
            self._stop();self._first_done=True;self._start('train')
        elif self._phase=='train' and self._prof is not None:
            self._prof.step();self._steady_steps+=1

    @contextmanager
    def suspend(self):
        """Exclude validation/checkpoints from sampling; never synchronize CUDA."""
        was_running=self._prof is not None
        phase=self._phase
        if was_running:self._stop()
        try:yield
        finally:
            if was_running and not self._closed:self._start(phase)

    def close(self):
        global _PROCESS_ACTIVE
        if self._closed:return
        try:self._stop()
        finally:
            if self._environment_monitor is not None:
                try:
                    self._environment_monitor.stop()
                    source=Path(self._environment_temp.name)/'environment.json'
                    if source.is_file():
                        target=Path(self.environment_file);target.parent.mkdir(parents=True,exist_ok=True)
                        document=json.loads(source.read_text(encoding='utf-8'))
                        journal=Path(str(source)+'.samples.jsonl')
                        if journal.is_file():
                            saved=Path(str(target)+'.samples.jsonl');shutil.copyfile(journal,saved)
                            document['raw_samples_path']=str(saved)
                        else:document['raw_samples_path']=None
                        target.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
                except Exception as exc:warnings.warn('Environment telemetry finalization failed: '+str(exc))
                self._environment_monitor=None
            if self._environment_temp is not None:
                self._environment_temp.cleanup();self._environment_temp=None
            if self._native_temp is not None:
                self._native_temp.cleanup();self._native_temp=None
            if self._token is not None:_ACTIVE.reset(self._token);self._token=None
            if _PROCESS_ACTIVE is self:_PROCESS_ACTIVE=None
            self._closed=True

    def finish(self,metadata=None):
        self.metadata.update(metadata or {})
        self.close()
        if not self.requested:return None
        traces=[self._traces]
        if dist.is_available() and dist.is_initialized():
            traces=[None]*dist.get_world_size()
            dist.all_gather_object(traces,self._traces)
        paths=[p for group in traces for p in group]
        result=None;error=None
        if self.rank==0:
            try:
                self.output.mkdir(parents=True,exist_ok=True)
                (self.output/('index.'+self.run_id+'.json')).write_text(json.dumps({'traces':paths,'configuration':self.config},indent=2)+'\n')
                if paths:
                    from terra.profiling.report import summarize_paths
                    result=summarize_paths(paths,self.output/('summary.'+self.run_id),
                        render_pdf=self.config['export_pdf'],preview_png=True,
                        max_plot_ranks=self.config['max_plot_ranks'])
                else:
                    warnings.warn('Profiling enabled but no active annotated window was captured')
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        if dist.is_available() and dist.is_initialized():
            message=[error];dist.broadcast_object_list(message,src=0);error=message[0]
        if error:raise RuntimeError('Profile summary failed: '+error)
        return result
