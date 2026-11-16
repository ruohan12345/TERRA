"""Reusable Chinese figures and interval analysis of independent rank traces.

No accelerator/model imports. CPU submission/wait scopes and device activity
are separate views; unobserved activity is never called physical idle or HBM.
"""
import math
import os
from pathlib import Path

from .intervals import union_intervals, intersect_intervals, interval_duration


CPU_PHASES = (
    ('final_drain', '末尾设备同步'), ('finite_objective', '目标有限性检查'),
    ('finite_gradients', '梯度有限性检查'), ('loss', '损失计算'),
    ('gradient_sync', '梯度收尾／等待'), ('gradient_clip', '梯度裁剪'),
    ('optimizer', '优化器更新'), ('unscale', '梯度反缩放'),
    ('backward', '反向提交／等待'), ('forward', '前向提交／等待'),
)
DEVICE_LANES = (
    ('compute', '设备计算（含打包／转换）', '#427EAA'),
    ('gradient', '梯度通信包络', '#E39B43'),
    ('halo', 'Halo 通信包络', '#9271B4'),
    ('other_or_unknown_communication', '其他通信（含来源未知）', '#A7B2BF'),
    ('visible_transport_copy', '可见通信搬运／独立拷贝', '#2C9C90'),
)
HOST_KINDS = {'cpu_op', 'cpu_runtime'}


def capture_window(events):
    """Adding CPU dispatch detail must not enlarge the legacy capture window."""
    bounded=[e for e in events if e['kind'] not in HOST_KINDS] or events
    return min(e['start_ns'] for e in bounded),max(e['end_ns'] for e in bounded)


def device_technology(metadata, device=None, adapter=None):
    context=metadata.get('metadata',{})
    case=context.get('case',metadata.get('case',{}))
    runtime=case.get('runtime',metadata.get('runtime',{}))
    names=[metadata.get('device',''),metadata.get('accelerator_backend',''),runtime.get('device',''),
           runtime.get('backend',''),device or '',(adapter or {}).get('name','')]
    text=' '.join(str(x).lower() for x in names)
    if any(token in text for token in ('npu','ascend','hccl')):
        return {'type':'NPU','backend':'Ascend','label':'NPU／Ascend','device':device}
    if any(token in text for token in ('cuda','nccl','gpu')) or metadata.get('cuda_build'):
        return {'type':'GPU','backend':'CUDA','label':'GPU／CUDA','device':device}
    if runtime.get('device')=='cpu' or metadata.get('device')=='cpu':
        return {'type':'CPU','backend':None,'label':'CPU（未采集加速器）','device':device}
    return {'type':'unknown','backend':None,'label':'加速器类型未知','device':device}


def file_io_events(events, metadata):
    """Application file acquisition; exclude fake/resident/copy assumptions.

    Canonical frame_acquire parents take precedence over contained legacy
    load_pt children. This measures cached application wall, never disk bytes.
    """
    mode=profile_context(metadata)['input_mode']
    parents=[e for e in events if e['kind']=='cpu_acquire' and e.get('metadata',{}).get('source')!='fake']
    selected=list(parents)
    for event in events:
        if event['kind']!='cpu_load' or event.get('metadata',{}).get('source')=='fake':
            continue
        if not any(p.get('pid')==event.get('pid') and p.get('tid')==event.get('tid') and
                   p['start_ns']<=event['start_ns']<=event['end_ns']<=p['end_ns'] for p in parents):
            selected.append(event)
    # A real file marker with explicit path/backend evidence wins over a broad
    # case label; otherwise fake/resident captures must never become file I/O.
    if mode in ('fake','device_resident','coordinate'):
        selected=[e for e in selected if any(e.get('metadata',{}).get(k) for k in
                  ('path','file_path','file_size_bytes','read_backend'))]
    status=('observed' if selected else 'not_applicable_fake' if mode=='fake' else
            'not_applicable_resident' if mode in ('device_resident','coordinate') else 'unknown')
    return {'events':selected,'status':status,'input_mode':mode,
            'semantics':'Application frame_acquire / uncovered legacy torch.load interval union, including page cache, deserialization and allocation; not pure disk service'}


def profile_context(metadata):
    """Recorded case metadata may be embedded by an external/ordinary harness."""
    context = metadata.get('metadata', {})
    case = context.get('case', metadata.get('case', {}))
    runtime = case.get('runtime', metadata.get('runtime', {}))
    work, model = case.get('workload', {}), case.get('model', context.get('model', {}))
    shape = work.get('shape_bchw')
    data_shape = context.get('data', {}).get('shape_chw')
    precision = context.get('precision', runtime.get('precision', metadata.get('precision')))
    if isinstance(precision, dict):
        precision = precision.get('precision')
    return {'precision': precision, 'shape_bchw': shape, 'shape_chw': shape[1:] if shape else data_shape,
            'rollout_steps': context.get('rollout_steps', work.get('rollout_steps')),
            'dmp_degree': case.get('parallel', context.get('parallel', {})).get('dmp_degree'),
            'world_size':case.get('topology',{}).get('world_size',metadata.get('world_size')),
            'input_mode': work.get('data_mode', context.get('data', {}).get('source', 'unknown')),
            'model_depth':model.get('depth'),
            'model_label': 'D%s' % model['embedding_dim'] if model.get('embedding_dim') else None}


def _difference(left, right):
    result = []
    right = union_intervals(right)
    for start, end in union_intervals(left):
        cursor = start
        for a, b in right:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                result.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def parse_zoom_ms(value):
    """A rank-local window relative to the selected update/capture start."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(':')
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('zoom_ms must be START:END in milliseconds')
    try:
        start, end = map(float, value)
    except (TypeError, ValueError) as error:
        raise ValueError('zoom_ms must contain two finite numbers') from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError('zoom_ms requires 0 <= START < END')
    return start, end


def _bounds(event):
    return event['start_ns'], event['end_ns']


def _phase_key(event):
    name = event['name']
    if name == 'terra/profile/final_drain':
        return 'final_drain'
    if name == 'terra/train/finite_check':
        operation = event.get('metadata', {}).get('operation')
        return 'finite_' + operation if operation in ('objective', 'gradients') else None
    if name.startswith('terra/train/'):
        key = name.rsplit('/', 1)[-1]
        if key in dict(CPU_PHASES):
            return key
    return None


def _cpu_partition(events, window, thread, file_events=()):
    grouped = {key: [] for key, _ in CPU_PHASES}
    for event in events:
        if event['kind'] != 'cpu_phase' or (event.get('pid'), event.get('tid')) != thread:
            continue
        key = _phase_key(event)
        if key:
            grouped[key].append(_bounds(event))
    file_intervals=intersect_intervals([_bounds(e) for e in file_events],[window])
    covered, rows = list(file_intervals), []
    if file_intervals:
        rows.append({'key':'file_io','label':'真实文件获取（应用墙钟）',
                     'wall_ms':interval_duration(file_intervals)/1e6,'intervals_ns':file_intervals,'recorded':True})
    for key, label in CPU_PHASES:
        current = _difference(intersect_intervals(grouped[key], [window]), covered)
        covered = union_intervals(covered + current)
        rows.append({'key': key, 'label': label, 'wall_ms': interval_duration(current) / 1e6,
                     'intervals_ns': current, 'recorded': bool(intersect_intervals(grouped[key],[window]))})
    current = _difference([window], covered)
    rows.append({'key': 'other', 'label': '其他主机范围／未覆盖间隙',
                 'wall_ms': interval_duration(current) / 1e6, 'intervals_ns': current, 'recorded': True})
    return rows


def _host_analysis(events,window,file_info):
    data={key:[] for key in ('cpu_op','cpu_runtime','file_input','file_target','file_other','cpu_prepare','cpu_h2d_submit')}
    counts={key:0 for key in data}
    for event in events:
        key=event['kind']
        if key in ('cpu_buffer','cpu_prepare','cpu_debug_checks','cpu_metadata','cpu_materialize','cpu_map'):
            key='cpu_prepare'
        if key not in data:
            continue
        cut=intersect_intervals([_bounds(event)],[window])
        if cut:
            data[key].extend(cut);counts[key]+=1
    for event in file_info['events']:
        key={'input':'file_input','target':'file_target'}.get(event.get('metadata',{}).get('role'),'file_other')
        cut=intersect_intervals([_bounds(event)],[window])
        if cut:
            data[key].extend(cut);counts[key]+=1
    data={key:union_intervals(value) for key,value in data.items()}
    # Prep is a separate lane from actual acquisition, not nested IO again.
    files=union_intervals(data['file_input']+data['file_target']+data['file_other'])
    data['cpu_prepare']=_difference(data['cpu_prepare'],files)
    return {'intervals_ns':data,'union_ms':{key:interval_duration(value)/1e6 for key,value in data.items()},
            'counts':counts,'file_io':{'status':file_info['status'],'wall_ms':interval_duration(files)/1e6,
                'input_wall_ms':interval_duration(data['file_input'])/1e6,
                'target_wall_ms':interval_duration(data['file_target'])/1e6,
                'other_wall_ms':interval_duration(data['file_other'])/1e6,
                'input_observed':bool(data['file_input']),'target_observed':bool(data['file_target']),
                'observed_in_selected_window':bool(files),'semantics':file_info['semantics']},
            'scope':'CPU operator/runtime lanes are multi-thread wall unions containing waits; not CPU busy time and not additive with device lanes'}


def _device_analysis(events, device, window, zoom_ms):
    if device == 'npu:ambiguous':
        return {'device':device,'available':False,'overlap_available':False,'intervals_ns':{k:[] for k,_,_ in DEVICE_LANES},
                'counts':{},'union_ms':{k:None for k,_,_ in DEVICE_LANES},
                'gradient_halo_overlap_ms':None,'gradient_compute_overlap_ms':None,
                'compute_visible_transport_overlap_ms':None,'observed_uncovered_ms':None,
                'zoom_interval_ns':list(window),'zoom_ms_relative_to_update':[0,(window[1]-window[0])/1e6],
                'zoom_selection':'device_identity_ambiguous','hbm_counters_available':False}
    selected = [e for e in events if e['kind'].startswith('gpu_') and str(e.get('device')) == device]
    data = {key: [] for key in ('compute', 'gradient', 'halo', 'other_communication', 'unknown_communication',
                                 'communication_transport', 'explicit_copy', 'h2d', 'unknown_device')}
    counts = {key: 0 for key in data}
    for event in selected:
        cut = intersect_intervals([_bounds(event)], [window])
        if not cut:
            continue
        kind, metadata = event['kind'], event.get('metadata', {})
        if kind == 'gpu_compute':
            key = 'compute'
        elif kind == 'gpu_comm':
            category = metadata.get('communication_category') if metadata.get('source_flow_exact') is True else 'unknown'
            key = {'gradient': 'gradient', 'halo': 'halo', 'other': 'other_communication'}.get(category, 'unknown_communication')
        elif kind == 'gpu_transport':
            key = 'communication_transport'
        elif kind in ('gpu_h2d', 'gpu_copy_other'):
            key = 'explicit_copy'
            if kind=='gpu_h2d':
                data['h2d'].extend(cut);counts['h2d']+=1
        else:
            key = 'unknown_device'
        data[key].extend(cut)
        counts[key] += 1
    data = {key: union_intervals(value) for key, value in data.items()}
    data['communication'] = union_intervals(data['gradient'] + data['halo'] + data['other_communication'] + data['unknown_communication'])
    data['other_or_unknown_communication'] = union_intervals(data['other_communication'] + data['unknown_communication'])
    data['visible_transport_copy'] = union_intervals(data['communication_transport'] + data['explicit_copy'])
    data['other_visible_transport_copy'] = _difference(data['visible_transport_copy'],data['h2d'])
    observed = union_intervals([part for intervals in data.values() for part in intervals])
    gh = intersect_intervals(data['gradient'], data['halo'])
    gc = intersect_intervals(data['gradient'], data['compute'])
    visible_compute = intersect_intervals(data['visible_transport_copy'], data['compute'])
    duration = window[1] - window[0]
    requested = parse_zoom_ms(zoom_ms)
    if requested is not None:
        if requested[1] * 1e6 > duration + 1:
            raise ValueError('zoom_ms exceeds the selected update/capture duration')
        zoom = [window[0] + int(requested[0] * 1e6), window[0] + int(requested[1] * 1e6)]
        reason = 'explicit_relative_ms'
    else:
        candidates, reason = (gh, 'gradient_halo_intersection') if gh else (gc, 'gradient_compute_intersection')
        if not candidates:
            candidates, reason = (intersect_intervals(data['communication'], data['compute']), 'compute_communication_intersection')
        if not candidates:
            candidates, reason = (data['gradient'] or data['communication'] or data['compute'], 'observed_activity_fallback')
        if candidates:
            a, b = max(candidates, key=lambda pair: pair[1] - pair[0])
            width = min(duration, max(duration * .04, (b - a) * 8))
            left = max(window[0], min((a + b - width) / 2, window[1] - width))
            zoom = [int(left), int(left + width)]
        else:
            zoom, reason = list(window), 'activity_unavailable'
    return {'device': device, 'available': bool(observed),'overlap_available':bool(observed),'intervals_ns': data, 'counts': counts,
            'union_ms': {key: interval_duration(value) / 1e6 if observed else None for key, value in data.items()},
            'gradient_halo_overlap_ms': interval_duration(gh) / 1e6 if observed else None,
            'gradient_compute_overlap_ms': interval_duration(gc) / 1e6 if observed else None,
            'compute_visible_transport_overlap_ms': interval_duration(visible_compute) / 1e6 if observed else None,
            'observed_uncovered_ms': (duration - interval_duration(observed)) / 1e6 if observed else None,
            'zoom_interval_ns': zoom, 'zoom_ms_relative_to_update': [(x - window[0]) / 1e6 for x in zoom],
            'zoom_selection': reason, 'hbm_counters_available': False}


def analyze_training_bundle(bundle, zoom_ms=None):
    """Normalize CPU/device views separately for this single trace clock."""
    zoom_ms = parse_zoom_ms(zoom_ms)
    events = bundle.events
    captured = capture_window(events)
    file_info=file_io_events(events,bundle.metadata)
    active = [e for e in events if e['kind'] == 'cpu_phase' and e['name'] == 'terra/profile/active_window']
    markers = [e for e in events if e['kind'] == 'cpu_phase' and e['name'] in (
        'terra/profile/update', 'terra/train/update', 'terra/train/update_attempt')]
    markers.sort(key=lambda e: (e['start_ns'], e['end_ns']))
    updates = []
    devices = sorted({str(e.get('device')) for e in events if e['kind'].startswith('gpu_')})
    for index, marker in enumerate(markers):
        window = _bounds(marker)
        bounded = not active or any(a['start_ns'] <= window[0] and a['end_ns'] >= window[1] for a in active)
        complete = bounded and marker.get('metadata', {}).get('complete', True) is not False
        thread = (marker.get('pid'), marker.get('tid'))
        drain = [e for e in events if e['name'] == 'terra/profile/final_drain'
                 and (e.get('pid'), e.get('tid')) == thread and window[0] <= e['start_ns'] <= e['end_ns'] <= window[1]]
        completed_device = complete and bool(drain)
        updates.append({'index': index, 'label': marker.get('label', marker['name']), 'metadata': marker.get('metadata', {}),
            'start_ns': window[0], 'end_ns': window[1], 'wall_ms': (window[1] - window[0]) / 1e6,
            'cpu_update_complete': complete, 'device_completion_observed': completed_device,
            'completion_evidence': 'final_drain_marker' if drain else 'not_observed',
            'main_cpu_thread': list(thread), 'cpu_partition': _cpu_partition(events, window, thread,file_info['events']),
            'host_layers':_host_analysis(events,window,file_info),
            'devices': [_device_analysis(events, device, window, None) for device in devices]})
    complete = [row for row in updates if row['cpu_update_complete']]
    selected = (complete or updates)[-1] if updates else None
    if selected is not None:
        window = (selected['start_ns'], selected['end_ns'])
        # Optional explicit zoom applies to the selected step only.
        selected = dict(selected, devices=[_device_analysis(events, device, window, zoom_ms) for device in devices])
    else:
        # Older CPU/CUDA/IO traces retain a clearly labelled capture view.
        phases = [e for e in events if e['kind'] == 'cpu_phase']
        first = phases[0] if phases else events[0]
        selected = {'index': None, 'start_ns': captured[0], 'end_ns': captured[1],
            'wall_ms': (captured[1] - captured[0]) / 1e6, 'cpu_update_complete': False,
            'device_completion_observed': False, 'completion_evidence': 'update_boundary_missing',
            'main_cpu_thread': [first.get('pid'), first.get('tid')],
            'cpu_partition': _cpu_partition(events, captured, (first.get('pid'), first.get('tid')),file_info['events']),
            'host_layers':_host_analysis(events,captured,file_info),
            'devices': [_device_analysis(events, device, captured, zoom_ms) for device in devices]}
    for update in updates:
        update['file_io']=update['host_layers']['file_io']
    selected['file_io']=selected['host_layers']['file_io']
    return {'format': 'terra_training_profile_figures_v1', 'clock_scope': 'one_trace_rank_cycle_only',
            'updates': updates, 'selected_update': selected, 'selection_policy': 'last_bounded_cpu_update; no cross-trace averaging',
            'recorded_context':profile_context(bundle.metadata),
            'technology':device_technology(bundle.metadata,adapter=getattr(bundle,'device_adapter',{})),
            'capture_window_ns':list(captured),
            'device_adapter': getattr(bundle, 'device_adapter', {}),
            'limitations': ['CPU阶段是主线程提交与等待墙钟，不能等同设备计算或网络耗时。',
                '通信包络包含等待；各设备行有重叠，不能累加。未覆盖区间不等于物理空闲。',
                '可见数据搬运仅包括记录到的通信传输与显式拷贝，不代表算子内部完整HBM访存。',
                '缺少输入／标签／文件／H2D轨道表示未知；驻留输入不代表真实文件I/O。']}


def timeline_lanes(bundle, analysis, device):
    """Layered static lanes; events/flows retain full per-thread detail in HTML."""
    selected=analysis['selected_update'];host=selected['host_layers']
    technology=device_technology(bundle.metadata,device['device'],getattr(bundle,'device_adapter',{}))
    prefix=technology['type'] if technology['type'] in ('GPU','NPU') else '类型未知加速器'
    specs=[('cpu_op','CPU算子（含等待）','#536D89'),('cpu_runtime','CPU runtime／launch（含等待）','#B39158'),
           ('file_input','CPU输入文件获取','#7354A5'),('file_target','CPU标签文件获取','#BE557B'),
           ('cpu_prepare','CPU数据准备（不重复文件获取）','#8C9B78'),
           ('cpu_h2d_submit','CPU H2D提交','#D9BC67')]
    if host['intervals_ns']['file_other']:
        specs.insert(4,('file_other','CPU其他文件获取','#997FAD'))
    rows=[]
    for key,label,color in specs:
        rows.append({'key':key,'label':label,'color':color,'layer':'CPU',
            'intervals_ns':host['intervals_ns'][key],'union_ms':host['union_ms'][key],
            'file_io_status':host['file_io']['status'] if key.startswith('file_') else None})
    specs=[('compute',prefix+'实际计算（含打包／转换）','#427EAA'),
           ('h2d',prefix+'实际H2D拷贝','#32A891'),('gradient',prefix+'梯度通信包络','#E39B43'),
           ('halo',prefix+' Halo通信包络','#9271B4'),
           ('other_or_unknown_communication',prefix+'其他通信（含来源未知）','#A7B2BF'),
           ('other_visible_transport_copy',prefix+'其他可见搬运／拷贝','#2C9C90')]
    for key,label,color in specs:
        rows.append({'key':key,'label':label,'color':color,'layer':prefix,
            'intervals_ns':device['intervals_ns'].get(key,[]),'union_ms':device['union_ms'].get(key)})
    return rows


def configure_matplotlib_font():
    """Use a real CJK face, with an explicit path override for offline hosts."""
    import matplotlib
    from matplotlib import font_manager
    requested = os.environ.get('TERRA_PLOT_FONT')
    if requested:
        if not Path(requested).is_file():
            raise ValueError('TERRA_PLOT_FONT must name an existing CJK font file')
        font_manager.fontManager.addfont(requested)
        families = ['DejaVu Sans', font_manager.FontProperties(fname=requested).get_name()]
    else:
        from .environment_report import chinese_font
        families = chinese_font().get_family()
    matplotlib.rcParams.update({'font.family': families, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                                  'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False})
    return {'status': 'available', 'families': families, 'explicit_path': requested}


def _context_label(bundle, selected):
    metadata = bundle.metadata
    context=profile_context(metadata)
    precision=context['precision'].upper() if isinstance(context['precision'],str) else '精度未知'
    technology=device_technology(metadata,adapter=getattr(bundle,'device_adapter',{}))['label']
    workload=[]
    if context['model_label']:
        workload.append(context['model_label']+(' / %s层' % context['model_depth'] if context['model_depth'] is not None else ''))
    if context['shape_bchw']:
        workload.append('B%s' % context['shape_bchw'][0])
    if context['shape_chw']:
        workload.append('C×H×W='+'×'.join(str(n) for n in context['shape_chw']))
    if context['world_size'] is not None:
        workload.append('P%s' % context['world_size'])
    workload.extend([precision,technology])
    return ' · '.join(workload)+'\n'+'节点 %s · rank %s · 周期 %s · %s' % (metadata.get('hostname', '未知'),
        metadata.get('global_rank', metadata.get('rank', '未知')), metadata.get('cycle', '未知'),
        '完整更新 %.2f ms' % selected['wall_ms'] if selected['cpu_update_complete'] else '完整更新边界未知')


def cpu_display_rows(selected):
    by_key={r['key']:r for r in selected['cpu_partition']}
    labels=(('forward','前向'),('loss','损失'),('backward','反向'),
            ('gradient_sync','梯度收尾／等待'),('gradient_clip','梯度裁剪'),('optimizer','优化器更新'))
    rows=[{'key':k,'label':label,'wall_ms':by_key[k]['wall_ms'],'recorded':by_key[k]['recorded']} for k,label in labels]
    used={k for k,_ in labels}
    if 'file_io' in by_key:
        rows.insert(0,{'key':'file_io','label':'文件获取 I/O（应用墙钟）',
                      'wall_ms':by_key['file_io']['wall_ms'],'recorded':True})
        used.add('file_io')
    remaining=[r for r in selected['cpu_partition'] if r['key'] not in used]
    rows.append({'key':'other','label':'其他','wall_ms':sum(r['wall_ms'] for r in remaining),
                 'recorded':any(r['recorded'] for r in remaining)})
    if abs(sum(r['wall_ms'] for r in rows)-selected['wall_ms'])>1e-6:
        raise ValueError('Host phase display does not cover the update exactly')
    return rows


def render_training_figures(bundle, analysis, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    font = configure_matplotlib_font()
    selected = analysis['selected_update']
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, name):
        for suffix in ('.png', '.pdf', '.svg'):
            p = prefix.with_name(prefix.name + '.' + name + suffix)
            if p.exists():
                raise FileExistsError(p)
            fig.savefig(p, dpi=160, facecolor='white')
            paths.append(str(p))
        plt.close(fig)
    rows=cpu_display_rows(selected)
    fig, ax = plt.subplots(figsize=(11.6, max(5.7, len(rows) * .38 + 2.1)))
    fig.subplots_adjust(left=.27, right=.95, top=.79, bottom=.20)
    maximum = max((r['wall_ms'] for r in rows), default=1)
    for index, row in enumerate(rows):
        color = '#7354A5' if row['key']=='file_io' else '#E39B43' if row['key'] == 'gradient_sync' else '#A7B2BF' if row['key'] == 'other' else '#427EAA'
        ax.barh(index, row['wall_ms'], height=.6, color=color)
        ax.text(row['wall_ms'] + maximum * .015, index,
                '%.2f ms' % row['wall_ms'] if row['recorded'] else '未观察到／未知',va='center',fontsize=10)
    ax.set_yticks(range(len(rows)), [r['label'] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, max(maximum * 1.25, .001)); ax.set_xlabel('主机更新墙钟（毫秒）'); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    fig.text(.045, .945, '单步主机耗时：CPU阶段与文件I/O' if selected['cpu_update_complete'] else '主机阶段（完整更新边界未知）', fontsize=20, weight='bold')
    fig.text(.045, .875, _context_label(bundle, selected), fontsize=10.5)
    completion = '本步包含末尾设备同步。' if selected['device_completion_observed'] else '未观察到末尾同步；不声称设备已完成该步。'
    io_note={'not_applicable_resident':'文件I/O不适用（驻留输入）','not_applicable_fake':'文件I/O不适用（合成输入）'}.get(selected['file_io']['status'],'文件I/O未观察到，时间未知')
    first_note=('文件获取优先从嵌套阶段扣除，含缓存／反序列化，不是纯磁盘时间。' if selected['file_io']['observed_in_selected_window'] else
                io_note+'；CPU阶段含提交、检查与等待，不是GPU/NPU纯计算。')
    fig.text(.045, .10,first_note,fontsize=10)
    fig.text(.045, .045, completion + ' 其他含检查、反缩放、末尾同步与间隙；CPU与设备图分别解读。', fontsize=9.5)
    save(fig, 'cpu_phases')
    for index, device in enumerate(selected['devices']):
        lanes=timeline_lanes(bundle,analysis,device)
        technology=device_technology(bundle.metadata,device['device'],getattr(bundle,'device_adapter',{}))
        fig, axes = plt.subplots(2, 1, figsize=(14.4, 12.2))
        fig.subplots_adjust(left=.30, right=.88, top=.84, bottom=.20, hspace=.56)
        origin, end = selected['start_ns'], selected['end_ns']
        limits = [(origin, end), tuple(device['zoom_interval_ns'])]
        for panel, (ax, limit) in enumerate(zip(axes, limits)):
            for lane, item in enumerate(lanes):
                cuts = intersect_intervals(item['intervals_ns'], [limit])
                ax.broken_barh([((a-origin)/1e6, (b-a)/1e6) for a,b in cuts], (lane-.28,.56), facecolors=item['color'])
                if panel == 0:
                    suffix = ('%.2f ms' % item['union_ms'] if cuts else
                              '不适用：驻留输入' if item.get('file_io_status')=='not_applicable_resident' else
                              '不适用：合成输入' if item.get('file_io_status')=='not_applicable_fake' else '未捕获／未知')
                    ax.text(1.015, lane, suffix, transform=ax.get_yaxis_transform(), va='center', fontsize=9)
            ax.set_yticks(range(len(lanes)), [r['label'] for r in lanes],fontsize=9.5); ax.invert_yaxis()
            cpu_count=sum(r['layer']=='CPU' for r in lanes)
            ax.axhline(cpu_count-.5,color='#697989',alpha=.4,lw=.8)
            ax.set_xlim((limit[0]-origin)/1e6, max((limit[1]-origin)/1e6,.001)); ax.grid(axis='x',alpha=.18)
            ax.set_xlabel('距本步／所选采集窗口开始（毫秒）')
            ax.set_title('CPU提交 → '+technology['label']+'实际执行（同轴完整更新）' if panel == 0 and selected['cpu_update_complete'] else '独立采集窗口' if panel == 0 else '局部放大（自动选择重叠区间）' if device['zoom_selection'] != 'explicit_relative_ms' else '局部放大（指定范围）', loc='left', fontsize=11)
        a,b = device['zoom_ms_relative_to_update']
        axes[0].add_patch(Rectangle((a,-.45),b-a,len(lanes)-.1,facecolor='#BCCBDA',edgecolor='#627D99',alpha=.15))
        fig.text(.045,.95,'CPU提交、'+technology['label']+'执行与I/O时间轴',fontsize=19,weight='bold')
        fig.text(.045,.905,_context_label(bundle,selected)+' · '+device['device'],fontsize=10)
        overlap_text=('梯度／Halo包络重叠 %.3f ms；梯度／实际计算重叠 %.3f ms。' % (device['gradient_halo_overlap_ms'], device['gradient_compute_overlap_ms'])
                      if device['overlap_available'] else '设备活动或身份未知：不合并不同设备时间轴，不计算重叠。')
        fig.text(.045,.14,overlap_text,fontsize=10.5)
        fig.text(.045,.10,'CPU算子／runtime为多线程墙钟并集，含等待；与'+technology['label']+'实际执行不能相加。',fontsize=10)
        io=selected['host_layers']['file_io']
        io_text=('文件获取 %.3f ms：应用frame_acquire并集，含缓存；不是纯磁盘耗时。' % io['wall_ms']
                 if io['observed_in_selected_window'] else '文件I/O不适用：驻留／合成输入。' if io['status'].startswith('not_applicable') else '本步未观察到真实文件I/O：未知，不能用拷贝代替。')
        fig.text(.045,.065,io_text,fontsize=10)
        fig.text(.045,.03,'通信包络含等待；可见SDMA／拷贝不是完整HBM访存。缺失轨道明确未知。',fontsize=9.5)
        save(fig, 'device%d_overlap' % index)
    return {'files': paths, 'font': font, 'selected_update_index': selected['index'],
            'cpu_display_rows':[{k:r[k] for k in ('key','label','wall_ms')} for r in rows],
            'technology':analysis['technology'],'file_io':selected['host_layers']['file_io'],
            'selected_update_device_completion_observed': selected['device_completion_observed']}
