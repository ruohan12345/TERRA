"""Read-only, sampled host telemetry, deliberately outside PyTorch tracing.

No CUDA/PyTorch imports or synchronization. Counters measure host activity, not
physical memory bandwidth or proof of isolation. Missing observations are None.
"""
from pathlib import Path
import argparse
import csv
import datetime
import glob
import json
import math
import os
import socket
import subprocess
import sys
import time


FORMAT = 'terra_profile_environment_v1'


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _read(path):
    return Path(path).read_text(encoding='utf-8', errors='replace')


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def parse_cpu_stat(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and (parts[0] == 'cpu' or parts[0][3:].isdigit() and parts[0].startswith('cpu')):
            values = [int(v) for v in parts[1:]]
            # guest counters are already included in user/nice.
            result[parts[0]] = values[:8] + [0] * max(0, 8-len(values))
    return result


def cpu_usage(previous, current, affinity=None):
    def combine(keys):
        if not previous or not keys or any(k not in previous or k not in current for k in keys):
            return None, None
        deltas = [sum(current[k][i]-previous[k][i] for k in keys) for i in range(8)]
        total = sum(deltas)
        if total <= 0 or min(deltas) < 0:
            return None, None
        return 100 * (total-deltas[3]-deltas[4])/total, 100*deltas[4]/total
    busy, wait = combine(['cpu'])
    local_busy, local_wait = combine(['cpu'+str(i) for i in affinity]) if affinity else (None, None)
    return {'busy_pct':busy, 'iowait_pct':wait, 'affinity_busy_pct':local_busy,
            'affinity_iowait_pct':local_wait}


def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        parts = value.split()
        if parts:
            values[key] = int(parts[0]) * (1024 if len(parts)>1 and parts[1]=='kB' else 1)
    return {'total_bytes':values.get('MemTotal'), 'available_bytes':values.get('MemAvailable'),
            'swap_total_bytes':values.get('SwapTotal'), 'swap_free_bytes':values.get('SwapFree'),
            'cached_bytes':values.get('Cached'), 'dirty_bytes':values.get('Dirty'),
            'writeback_bytes':values.get('Writeback')}


def parse_pressure(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields:
            result[fields[0]] = {key:_number(value) for key,value in
                                (field.split('=',1) for field in fields[1:] if '=' in field)}
    return result


def parse_diskstats(text, devices=None):
    result = {}
    for line in text.splitlines():
        p = line.split()
        if len(p)<14 or p[2].startswith(('loop','ram')) or devices is not None and p[2] not in devices:
            continue
        result[p[2]] = {'read_bytes':int(p[5])*512, 'write_bytes':int(p[9])*512,
                        'io_time_ms':int(p[12]), 'weighted_io_time_ms':int(p[13]),
                        'in_flight':int(p[11]), 'major':int(p[0]), 'minor':int(p[1])}
    return result


def disk_rates(previous, current, elapsed):
    result = []
    for name, row in sorted(current.items()):
        data = {'device':name, **row, 'read_bytes_per_s':None, 'write_bytes_per_s':None,
                'busy_pct':None}
        old = previous.get(name) if previous else None
        if old and elapsed and elapsed>0:
            for key in ('read_bytes','write_bytes'):
                change = row[key]-old[key]
                data[key+'_per_s'] = change/elapsed if change>=0 else None
            change = row['io_time_ms']-old['io_time_ms']
            data['busy_pct'] = 100*change/(elapsed*1000) if change>=0 else None
        result.append(data)
    return result


def parse_process_stat(text):
    end = text.rfind(')')
    prefix, tail = text[:end+1], text[end+2:].split()
    return {'pid':int(prefix.split(' ',1)[0]), 'name':prefix[prefix.index('(')+1:-1],
            'ppid':int(tail[1]), 'cpu_ticks':int(tail[11])+int(tail[12]),
            'start_ticks':int(tail[19])}


def job_processes(processes, roots):
    result = set(int(pid) for pid in roots)
    while True:
        extra = {pid for pid,row in processes.items() if row['ppid'] in result} - result
        if not extra:
            return result
        result.update(extra)


def process_rates(previous, current, elapsed, roots, clock_ticks=100):
    ours = job_processes(current, roots)
    rows = []
    for pid,row in current.items():
        output = {key:row.get(key) for key in ('pid','name','ppid')}
        output.update(is_job=pid in ours, cpu_pct=None, read_bytes_per_s=None, write_bytes_per_s=None)
        old = previous.get(pid) if previous else None
        if old and old['start_ticks']==row['start_ticks'] and elapsed and elapsed>0:
            change = row['cpu_ticks']-old['cpu_ticks']
            output['cpu_pct'] = 100*change/(clock_ticks*elapsed) if change>=0 else None
            for key in ('read_bytes','write_bytes'):
                if old.get(key) is not None and row.get(key) is not None:
                    change = row[key]-old[key]
                    output[key+'_per_s'] = change/elapsed if change>=0 else None
        rows.append(output)
    return rows, ours


def parse_gpu_csv(text):
    names = ('index','name','uuid','pci_bus_id','memory_total_mib','memory_used_mib',
             'utilization_gpu_pct','utilization_memory_pct','power_w','temperature_c')
    result = []
    for fields in csv.reader(text.splitlines()):
        if len(fields)!=len(names):
            continue
        row = dict(zip(names,(v.strip() for v in fields)))
        for key in names:
            if key not in ('name','uuid','pci_bus_id'):
                row[key] = _number(row[key])
        result.append(row)
    return result


def resolve_selected_gpus(inventory, participants, selected_devices=None):
    selectors = list(selected_devices or [])
    unknown = []
    if not selectors:
        for participant in participants:
            if participant.get('gpu_uuid'):
                selectors.append(str(participant['gpu_uuid']))
                continue
            device = str(participant.get('device',''))
            if not device.startswith('cuda'):
                continue
            logical = int(device.split(':',1)[1]) if ':' in device else 0
            visible = participant.get('cuda_visible_devices')
            if visible is not None:
                visible = [s.strip() for s in str(visible).split(',') if s.strip()]
                if logical>=len(visible):
                    unknown.append(device+' not in visible GPU list')
                    continue
                if visible[logical].startswith('GPU-'):
                    selectors.append(visible[logical])
                else:
                    unknown.append('Numeric CUDA_VISIBLE_DEVICES cannot establish physical nvidia-smi mapping without a verified UUID')
            else:
                unknown.append('CUDA logical ordinal '+str(logical)+' has no verified physical GPU UUID')
    matched = set()
    for selector in selectors:
        selector = str(selector).strip()
        matches = [row for row in inventory if row.get('index') is not None and str(int(row['index']))==selector or
                   row['uuid'].startswith(selector) and selector.startswith('GPU-')]
        if len(matches)==1:
            matched.add(matches[0]['uuid'])
        else:
            unknown.append('Cannot uniquely resolve GPU selector '+selector)
    return sorted(matched), unknown


def _stat_processes():
    result = {}
    denied = 0
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            row = parse_process_stat(_read(path))
        except (OSError, ValueError, IndexError):
            continue
        row.update(read_bytes=None, write_bytes=None)
        try:
            values = dict(line.split(':',1) for line in _read(path.parent/'io').splitlines() if ':' in line)
            row.update(read_bytes=int(values['read_bytes']), write_bytes=int(values['write_bytes']))
        except (OSError, ValueError, KeyError):
            denied += 1
        # comm is a basename, unlike cmdline which could contain credentials.
        result[row['pid']] = row
    return result, denied


def _run(args):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=3,
                          stdin=subprocess.DEVNULL)
    if proc.returncode:
        raise OSError(args[0]+' exit '+str(proc.returncode))
    return proc.stdout


def _gpu_snapshot():
    errors = []
    try:
        inventory = parse_gpu_csv(_run(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu','--format=csv,noheader,nounits']))
    except (OSError, subprocess.SubprocessError) as exc:
        inventory = []
        errors.append('GPU inventory unavailable: '+str(exc))
    processes = None
    if inventory:
        try:
            processes = []
            for fields in csv.reader(_run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits']).splitlines()):
                if len(fields)==3 and fields[0].strip().isdigit():
                    processes.append({'pid':int(fields[0]), 'gpu_uuid':fields[1].strip(),
                                      'used_memory_mib':_number(fields[2].strip())})
        except (OSError, subprocess.SubprocessError) as exc:
            processes = None
            errors.append('GPU process list unavailable: '+str(exc))
    return inventory, processes, errors


def _unescape_mount(value):
    for code,char in (('040',' '),('011','\t'),('012','\n'),('134','\\')):
        value = value.replace('\\'+code,char)
    return value


def storage_locations(paths, mountinfo=None):
    try:
        lines = (mountinfo if mountinfo is not None else _read('/proc/self/mountinfo')).splitlines()
    except OSError:
        lines = []
    mounts = []
    for line in lines:
        left, separator, right = line.partition(' - ')
        a,b = left.split(),right.split()
        if separator and len(a)>=5 and len(b)>=2:
            mounts.append({'mountpoint':_unescape_mount(a[4]), 'filesystem':b[0],
                           'source':_unescape_mount(b[1]), 'device_id':a[2]})
    result = []
    for raw in paths:
        path = str(Path(raw).resolve())
        matches = [m for m in mounts if path==m['mountpoint'] or
                   path.startswith(m['mountpoint'].rstrip('/')+'/')]
        row = max(matches,key=lambda m:len(m['mountpoint'])) if matches else {}
        result.append({'path':path, **row, 'network_filesystem':row.get('filesystem','').lower().startswith(('nfs','cifs','lustre','ceph','fuse','dpc')) if row else None})
    return result


def _cgroup_cpu():
    try:
        relative = next(line.split(':',2)[2] for line in _read('/proc/self/cgroup').splitlines() if line.startswith('0::'))
        base = Path('/sys/fs/cgroup')/relative.lstrip('/')
        # Some containers expose a namespace-relative cgroup root.
        if not (base/'cpu.max').is_file():
            base = Path('/sys/fs/cgroup')
        quotas = []
        cgroup_root = Path('/sys/fs/cgroup')
        while base==cgroup_root or cgroup_root in base.parents:
            try:
                quota,period = _read(base/'cpu.max').split()[:2]
                if quota!='max':
                    quotas.append(int(quota)/int(period))
            except OSError:
                pass
            if base==cgroup_root:
                break
            base = base.parent
        return min(quotas) if quotas else None
    except (OSError, ValueError, StopIteration, IndexError):
        return None


class _Sampler:
    def __init__(self, participants, roots, affinity=None):
        self.participants = participants
        self.roots = list(roots)
        self.previous = {}
        self.affinity = affinity
        self.root_identities = {}

    def snapshot(self):
        timestamp = time.monotonic()
        elapsed = timestamp-self.previous['time'] if self.previous else None
        result = {'utc':_utc(), 'monotonic_s':timestamp, 'interval_s':elapsed,
                  'errors':[], 'cpu':{}, 'memory':{}, 'pressure':{}, 'disks':[],
                  'processes':[], 'gpu_processes':None, 'gpus':[]}
        current = {'time':timestamp}
        try:
            cpus = parse_cpu_stat(_read('/proc/stat'))
            affinity = sorted(set(c for p in self.participants for c in (p.get('cpu_affinity') or [])))
            if not affinity:
                affinity = self.affinity or sorted(os.sched_getaffinity(0))
            result['cpu'] = cpu_usage(self.previous.get('cpus'),cpus,affinity)
            current['cpus'] = cpus
            load = _read('/proc/loadavg').split()
            result['cpu'].update(total_cores=len(cpus)-1, affinity_cores=len(affinity),
                                 affinity_cpu_ids=affinity, cgroup_quota_cores=_cgroup_cpu(),
                                 load1=float(load[0]), runnable=int(load[3].split('/')[0]))
        except (OSError, AttributeError, ValueError, IndexError) as exc:
            result['errors'].append('CPU counters unavailable: '+str(exc))
        try:
            result['memory'] = parse_meminfo(_read('/proc/meminfo'))
        except (OSError, ValueError) as exc:
            result['errors'].append('Memory counters unavailable: '+str(exc))
        for resource in ('cpu','memory','io'):
            try:
                result['pressure'][resource] = parse_pressure(_read('/proc/pressure/'+resource))
            except (OSError, ValueError):
                result['pressure'][resource] = None
        try:
            devices = {p.name for p in Path('/sys/block').iterdir()}
            disks = parse_diskstats(_read('/proc/diskstats'),devices)
            result['disks'] = disk_rates(self.previous.get('disks'),disks,elapsed)
            current['disks'] = disks
        except (OSError, ValueError) as exc:
            result['errors'].append('Disk counters unavailable: '+str(exc))
        processes = {}
        try:
            processes,denied = _stat_processes()
            roots = self.roots+[os.getpid()]+[p['pid'] for p in self.participants if p.get('pid')]
            valid_roots = []
            for pid in roots:
                if pid not in processes:
                    continue
                birth = processes[pid]['start_ticks']
                self.root_identities.setdefault(pid,birth)
                if self.root_identities[pid]==birth:
                    valid_roots.append(pid)
            rows,ours = process_rates(self.previous.get('processes'),processes,elapsed,valid_roots,os.sysconf('SC_CLK_TCK'))
            # Keep all job processes plus strongest external CPU/IO consumers.
            external = [r for r in rows if not r['is_job']]
            selected = sorted(external,key=lambda r:r['cpu_pct'] or 0,reverse=True)[:12]
            selected += sorted(external,key=lambda r:(r['read_bytes_per_s'] or 0)+(r['write_bytes_per_s'] or 0),reverse=True)[:12]
            result['processes'] = list({r['pid']:r for r in rows if r['is_job'] or r in selected}.values())
            result['process_io_unreadable_count'] = denied
            result['processes_observed_count'] = len(processes)
            current['processes'] = processes
        except (OSError, ValueError, AttributeError) as exc:
            ours = set()
            result['errors'].append('Process counters unavailable: '+str(exc))
        gpus,gpu_processes,errors = _gpu_snapshot()
        result.update(gpus=gpus,gpu_processes=gpu_processes)
        result['errors'].extend(errors)
        if gpu_processes is not None:
            for process in gpu_processes:
                # NVML may expose host PIDs while /proc uses a container namespace.
                # An invisible PID is unknown, not automatically another job.
                process['is_job'] = process['pid'] in ours if process['pid'] in processes else None
        result['collection_wall_ms'] = (time.monotonic()-timestamp)*1000
        self.previous = current
        return result


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def _maximum(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _phase_summary(samples, selected_uuids):
    cpus = [s.get('cpu',{}) for s in samples]
    result = {'sample_count':len(samples),
              'cpu_busy_mean_pct':_mean([c.get('busy_pct') for c in cpus]),
              'cpu_busy_max_pct':_maximum([c.get('busy_pct') for c in cpus]),
              'cpu_iowait_max_pct':_maximum([c.get('iowait_pct') for c in cpus]),
              'affinity_cpu_busy_mean_pct':_mean([c.get('affinity_busy_pct') for c in cpus]),
              'affinity_cpu_busy_max_pct':_maximum([c.get('affinity_busy_pct') for c in cpus]),
              'cpu_total_cores':_maximum([c.get('total_cores') for c in cpus]),
              'cpu_affinity_cores':_maximum([c.get('affinity_cores') for c in cpus]),
              'cgroup_quota_cores':_maximum([c.get('cgroup_quota_cores') for c in cpus]),
              'memory_available_min_bytes':None, 'selected_gpus':[],
              'external_cpu_top':[], 'external_io_top':[]}
    available = [s.get('memory',{}).get('available_bytes') for s in samples]
    available = [v for v in available if v is not None]
    result['memory_available_min_bytes'] = min(available) if available else None
    other_jobs={}
    for sample in samples:
        for process in sample.get('gpu_processes') or []:
            if process.get('is_job') is not True and process['gpu_uuid'] not in selected_uuids:
                other_jobs[(process['gpu_uuid'],process['pid'])]=process
    result['other_gpu_jobs']=list(other_jobs.values())
    for uuid in selected_uuids:
        observations = [g for s in samples for g in s.get('gpus',[]) if g['uuid']==uuid]
        identity = observations[0] if observations else {}
        foreign = sorted({p['pid'] for s in samples for p in s.get('gpu_processes') or []
                          if p['gpu_uuid']==uuid and p.get('is_job') is False})
        unknown = sorted({p['pid'] for s in samples for p in s.get('gpu_processes') or []
                          if p['gpu_uuid']==uuid and p.get('is_job') is None})
        utilization_count = sum(g.get('utilization_gpu_pct') is not None for g in observations)
        process_count = sum(s.get('gpu_processes') is not None for s in samples)
        complete = bool(samples) and len(observations)==len(samples) and utilization_count==len(samples) and process_count==len(samples) and not unknown
        maximum_utilization = _maximum([g.get('utilization_gpu_pct') for g in observations])
        result['selected_gpus'].append({'uuid':uuid, 'index':identity.get('index'), 'name':identity.get('name'),
                'utilization_mean_pct':_mean([g.get('utilization_gpu_pct') for g in observations]),
                'utilization_max_pct':maximum_utilization,
                'memory_controller_mean_pct':_mean([g.get('utilization_memory_pct') for g in observations]),
                'memory_used_max_mib':_maximum([g.get('memory_used_mib') for g in observations]),
                'memory_total_mib':identity.get('memory_total_mib'), 'foreign_pids':foreign,
                'unknown_process_pids':unknown,
                'observed_sample_count':len(observations), 'utilization_sample_count':utilization_count,
                'process_list_sample_count':process_count, 'coverage_complete':complete,
                'low_load':bool(maximum_utilization<=5 and not foreign) if complete else None})
    external = [p for s in samples for p in s.get('processes',[]) if not p.get('is_job',False)]
    for key,fields in (('external_cpu_top',('cpu_pct',)),('external_io_top',('read_bytes_per_s','write_bytes_per_s'))):
        ordered = sorted(external,key=lambda p:sum(p.get(f) or 0 for f in fields),reverse=True)
        unique = {}
        for row in ordered:
            if row['pid'] not in unique and any(row.get(f) is not None for f in fields):
                unique[row['pid']] = row
        result[key] = list(unique.values())[:5]
    for direction in ('read','write'):
        # Max individual block-device rate avoids adding md/dm + underlying disks twice.
        result['disk_'+direction+'_bytes_per_s_max'] = _maximum([
            d.get(direction+'_bytes_per_s') for s in samples for d in s.get('disks',[])])
    result['disk_busy_max_pct'] = _maximum([d.get('busy_pct') for s in samples for d in s.get('disks',[])])
    return result


def summarize_environment(document):
    baseline = document.get('baseline_samples',[])
    during = document.get('samples',[])
    selected = document.get('selected_gpu_uuids',[])
    before,run = _phase_summary(baseline,selected),_phase_summary(during,selected)
    errors = sorted({str(e) for s in baseline+during for e in s.get('errors',[])})
    selected_processes_known = bool(selected) and bool(baseline) and bool(during) and all(
        s.get('gpu_processes') is not None for s in baseline+during)
    gpu_complete = bool(selected) and bool(baseline) and bool(during) and all(
        g['coverage_complete'] for phase in (before,run) for g in phase['selected_gpus'])
    process_io_complete = all(not s.get('process_io_unreadable_count',0) for s in baseline+during)
    gpu_foreign = any(g['foreign_pids'] for phase in (before,run) for g in phase['selected_gpus'])
    other_gpu_jobs=bool(before['other_gpu_jobs'] or run['other_gpu_jobs'])
    cpu_foreign = any((p.get('cpu_pct') or 0)>=50 for phase in (before,run) for p in phase['external_cpu_top'])
    io_foreign = any((p.get('read_bytes_per_s') or 0)+(p.get('write_bytes_per_s') or 0)>=1024**2
                     for phase in (before,run) for p in phase['external_io_top'])
    enough_cpu = before['cpu_busy_mean_pct'] is not None and run['cpu_busy_mean_pct'] is not None
    classification = 'interference_observed' if gpu_foreign or cpu_foreign or io_foreign or other_gpu_jobs else (
        'no_obvious_interference_observed' if enough_cpu and not document.get('mapping_errors') and
        process_io_complete and (not selected or gpu_complete) else 'unknown')
    limitations = [
        'Sampled observations can miss short jobs; no claim of exclusive hardware access or zero interference.',
        'Host CPU, disk and memory counters are shared-host observations; external activity does not prove causal slowdown.',
        'GPU memory utilization is controller activity, not measured memory bandwidth. CPU memory bandwidth is unavailable.',
        'Disk rate is the maximum observed individual block-device rate, not dataset physical read bandwidth.',
        'Cache state is uncontrolled; filesystem and device caches are not flushed.',
        'Process CPU 100% means one logical CPU; host CPU percentages use all observed host logical CPUs.',
        'All phases use their own sampled wall clock. Samples are not exact per-operation PyTorch trace timings.'
        ,'The monitor adds read-only process queries and JSON writes; collection_wall_ms records sampled collector overhead.'
    ]
    if any(s.get('network_filesystem') or s.get('filesystem') in ('dpc','dpcfs') for s in document.get('storage',[])):
        limitations.append('Dataset is on a network/distributed filesystem: local disk counters do not measure remote server contention or physical data reads.')
    if any(s.get('process_io_unreadable_count',0) for s in baseline+during):
        limitations.append('Some process IO counters are unreadable; external IO process attribution is incomplete.')
    if any(g.get('unknown_process_pids') for phase in (before,run) for g in phase['selected_gpus']):
        limitations.append('Some GPU process PIDs are not visible in /proc; PID namespace differences or short-lived processes make job membership unknown.')
    return {'baseline':before, 'during':run, 'classification':classification,
            'baseline_gpu_low_load':all(g['low_load'] for g in before['selected_gpus'])
                    if before['selected_gpus'] and all(g['coverage_complete'] for g in before['selected_gpus']) else None,
            'thresholds':{'gpu_low_load_utilization_pct':5, 'external_process_cpu_pct':50,
                          'external_process_io_bytes_per_s':1024**2},
            'evidence':{'external_gpu_process':gpu_foreign,'external_cpu_activity':cpu_foreign,
                        'external_io_activity':io_foreign,'other_gpus_have_jobs':other_gpu_jobs},
            'coverage':{'gpu_mapping_available':bool(selected),'gpu_process_list_available':selected_processes_known,
                        'gpu_observation_complete':gpu_complete, 'process_io_complete':process_io_complete,
                        'observed_interval_mean_s':_mean([s.get('interval_s') for s in baseline+during]),
                        'observed_interval_max_s':_maximum([s.get('interval_s') for s in baseline+during]),
                        'collector_wall_mean_ms':_mean([s.get('collection_wall_ms') for s in baseline+during]),
                        'collector_wall_max_ms':_maximum([s.get('collection_wall_ms') for s in baseline+during]),
                        'cpu_baseline_and_during_available':enough_cpu,'sample_errors':errors},
            'limitations':limitations}


class MonitorHandle:
    def __init__(self, process, path):
        self.process, self.path = process, Path(path)

    def update_participants(self, participants):
        _atomic_json(str(self.path)+'.participants.json',participants)

    def stop(self):
        Path(str(self.path)+'.stop').touch()
        try:
            self.process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        return self.path


def start_monitor(output_path, *, participants=None, data_paths=None, interval_s=1.,
                  baseline_samples=3, selected_devices=None, job_root_pids=None):
    """Collect baseline before returning; monitor continues in a separate process.

    Call outside the measured region. Caller owns exception handling if process
    creation is unavailable. No GPU context, CUDA call, profiler, or cache flush.
    """
    if os.name!='posix' or not Path('/proc/stat').is_file():
        raise OSError('Linux /proc telemetry is unavailable on this platform')
    if not isinstance(interval_s,(int,float)) or isinstance(interval_s,bool) or not math.isfinite(interval_s) or interval_s<.1:
        raise ValueError('environment interval_s must be finite and >= 0.1')
    if not isinstance(baseline_samples,int) or isinstance(baseline_samples,bool) or baseline_samples<2:
        raise ValueError('environment baseline_samples must be >= 2')
    path = Path(output_path).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    config = {'participants':participants or [], 'data_paths':data_paths or [],
              'interval_s':float(interval_s), 'baseline_samples':baseline_samples,
              'selected_devices':selected_devices, 'job_root_pids':job_root_pids or [],
              'owner_pid':os.getpid()}
    config_path = Path(str(path)+'.config.json')
    _atomic_json(config_path,config)
    # Direct file execution keeps torch and third-party imports out of the sampler.
    process = subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(path)],
                               stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    handle = MonitorHandle(process,path)
    deadline = time.monotonic()+max(25,baseline_samples*(interval_s+8)+5)
    ready = Path(str(path)+'.ready')
    while time.monotonic()<deadline:
        if ready.is_file():
            return handle
        if process.poll() is not None:
            raise RuntimeError('Environment monitor exited before baseline; see '+str(path))
        time.sleep(.05)
    handle.stop()
    raise TimeoutError('Environment monitor baseline did not finish; see '+str(path))


def _registrations(path, initial):
    hostname = socket.gethostname()
    combined = {str(p.get('hostname',''))+':'+str(p.get('rank',p.get('pid'))):p for p in initial
                if p.get('hostname',hostname)==hostname}
    for source in [str(path)+'.participants.json']+glob.glob(str(path)+'.rank*.json'):
        try:
            value = json.loads(_read(source))
            for participant in value if isinstance(value,list) else [value]:
                if participant.get('hostname',hostname)!=hostname:
                    continue
                combined[str(participant.get('hostname',''))+':'+str(participant.get('rank',participant.get('pid')))] = participant
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return list(combined.values())


def _worker(path):
    path = Path(path)
    config = json.loads(_read(str(path)+'.config.json'))
    document = {'format':FORMAT, 'hostname':socket.gethostname(), 'started_utc':_utc(),
                'finished_utc':None, 'sampler_pid':os.getpid(), 'interval_s':config['interval_s'],
                'participants':config['participants'], 'inventory':[], 'selected_gpu_uuids':[],
                'baseline_samples':[], 'samples':[], 'storage':storage_locations(config['data_paths']),
                'mapping_errors':[], 'sampling_semantics':'Periodic read-only host counters; baseline completes before measured workload begins.'}
    raw_path = Path(str(path)+'.samples.jsonl')
    document['raw_samples_path'] = str(raw_path)
    journal = raw_path.open('a',encoding='utf-8')
    roots = config['job_root_pids']+[config['owner_pid']]
    sampler = _Sampler(config['participants'],roots)
    def sample(baseline):
        document['participants'] = _registrations(path,config['participants'])
        sampler.participants = document['participants']
        snapshot = sampler.snapshot()
        document['inventory'] = snapshot['gpus'] or document['inventory']
        selected,errors = resolve_selected_gpus(document['inventory'],document['participants'],config['selected_devices'])
        document['selected_gpu_uuids'],document['mapping_errors'] = selected,errors
        document['baseline_samples' if baseline else 'samples'].append(snapshot)
        journal.write(json.dumps({'phase':'baseline' if baseline else 'during', 'snapshot':snapshot},
                                 ensure_ascii=False,separators=(',',':'))+'\n')
        journal.flush()
        # Avoid O(samples^2) cumulative JSON writes competing with measured IO.
        # A full baseline is available before ready; a compact progress summary
        # every 30 samples supports report generation before the wrapper exits.
        if baseline or len(document['samples'])%30==0:
            document['summary'] = summarize_environment(document)
            progress = document if baseline else dict(document,samples=[],
                samples_in_raw_file=len(document['samples']), sample_storage='append-only journal; final JSON includes all samples')
            _atomic_json(path,progress)
    try:
        for index in range(config['baseline_samples']):
            if index:
                time.sleep(config['interval_s'])
            sample(True)
        Path(str(path)+'.ready').touch()
        stop = Path(str(path)+'.stop')
        while not stop.exists():
            # Poll stop often; interval controls starts without a busy loop.
            deadline = time.monotonic()+config['interval_s']
            while time.monotonic()<deadline and not stop.exists():
                time.sleep(min(.1,max(.001,deadline-time.monotonic())))
            if not stop.exists():
                sample(False)
            try:
                os.kill(config['owner_pid'],0)
            except OSError:
                document['owner_exited'] = True
                break
    except Exception as exc:
        document['monitor_error'] = type(exc).__name__+': '+str(exc)
    finally:
        journal.close()
        document['finished_utc'] = _utc()
        document['summary'] = summarize_environment(document)
        _atomic_json(path,document)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',required=True)
    _worker(parser.parse_args().worker)
