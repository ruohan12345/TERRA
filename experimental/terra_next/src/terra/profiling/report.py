"""IO-first overview from original PyTorch events, with explicit timing limits.

All overlaps are rank-local and device-local interval intersections. Separate
trace files/capture phases never share an assumed clock. PDF/CSV generation
runs offline, outside the profiler callback and measured training window.
"""
import csv,json,math
from pathlib import Path

from .intervals import union_intervals,intersect_intervals,interval_duration,difference_duration
from .trace import discover_traces,read_trace
from .environment_report import load_environment_record,unique_environments,render_environment_intro
from .figures import (analyze_training_bundle,render_training_figures,configure_matplotlib_font,
                      parse_zoom_ms,profile_context,capture_window,device_technology)
from .hotspots import analyze_hotspots

LANES=(('cpu_op','CPU 算子（含等待）','#536D89'),
       ('cpu_runtime','CPU runtime／launch（含等待）','#B39158'),
       ('cpu_input_load','CPU 输入获取','#7354a5'),
       ('cpu_target_load','CPU 标签获取','#be557b'),
       ('cpu_other_load','CPU 掩码／清单获取','#8b8794'),
       ('cpu_load','CPU 完整 .pt 加载（子范围）','#b6a1d0'),
       ('cpu_map','CPU mmap 打开（子范围）','#b69238'),
       ('cpu_materialize','CPU 局部物化（含 I/O）','#538b8e'),
       ('cpu_buffer','CPU 缓冲／补齐','#a6a0ba'),
       ('cpu_debug_checks','CPU 可选数值检查','#b87373'),
       ('cpu_metadata','CPU 元信息／其他准备','#aeb4b8'),
       ('cpu_h2d_submit','CPU H2D 调用（可能等待）','#d9bc67'),
       ('gpu_h2d','设备 H2D 拷贝','#32a891'),
       ('gpu_compute','设备计算／打包／转换','#377bb5'),
       ('gpu_comm','设备通信包络（含等待）','#e68b3c'),
       ('gpu_transport','可见通信传输','#42a89a'),
       ('gpu_copy_other','设备其他拷贝／填充','#8c9eab'),
       ('gpu_unknown','设备活动未分类','#bb647c'))


def _intervals(events,kinds):
    return [(e['start_ns'],e['end_ns']) for e in events if e['kind'] in kinds]


def _ms(value):return round(value/1e6,6)


def _ratio(numerator,denominator):return numerator/denominator if denominator>0 else None


def _acquisition_events(events):
    """Parent acquisitions, with old unwrapped load markers as a fallback.

    A parent's full-load/materialize children are details, not additional IO
    time. Keeping uncovered old loads permits mixed-version partial captures.
    """
    parents=[e for e in events if e['kind']=='cpu_acquire']
    result=list(parents)
    for event in events:
        if event['kind']!='cpu_load':continue
        covered=any(p.get('pid')==event.get('pid') and p.get('tid')==event.get('tid') and
                    p['start_ns']<=event['start_ns'] and p['end_ns']>=event['end_ns']
                    for p in parents)
        if not covered:result.append(event)
    return result


def _load_statistics(events,*,acquisitions=False):
    loads=_acquisition_events(events) if acquisitions else [e for e in events if e['kind']=='cpu_load']
    result={}
    for role in sorted(set(str(e['metadata'].get('role','unknown')) for e in loads)):
        selected=[e for e in loads if str(e['metadata'].get('role','unknown'))==role]
        intervals=[(e['start_ns'],e['end_ns']) for e in selected]
        result[role]={'frame_calls':len(selected),'wall_ms':_ms(interval_duration(intervals)),
                      'sum_call_ms':_ms(sum(b-a for a,b in intervals)),
                      'file_size_bytes_sum':sum(int(e['metadata'].get('file_size_bytes',0) or 0) for e in selected),
                      'requested_region_bytes_sum':sum(int(e['metadata'].get('requested_region_bytes',0) or 0) for e in selected),
                      'materialized_bytes_sum':sum(int(e['metadata'].get('materialized_bytes',0) or 0) for e in selected),
                      'byte_semantics':'Sum of file sizes referenced by calls, NOT physical disk bytes read'}
    return result


def _slow_frames(events):
    result=[]
    for e in sorted(events,key=lambda e:e['end_ns']-e['start_ns'],reverse=True)[:12]:
        result.append({'duration_ms':_ms(e['end_ns']-e['start_ns']),
                       'role':e['metadata'].get('role','unknown'),'lead':e['metadata'].get('lead'),
                       'frame_id':e['metadata'].get('frame_id'),'sample_id':e['metadata'].get('sample_id'),
                       'file_size_bytes':e['metadata'].get('file_size_bytes'),
                       'read_backend':e['metadata'].get('read_backend','legacy_naive'),
                       'path':e['metadata'].get('path',e['metadata'].get('file_path'))})
    return result


def summarize_bundle(bundle,*,zoom_ms=None):
    metadata,events=bundle.metadata,bundle.events
    context=metadata.get('metadata',{})
    source=metadata.get('source',metadata.get('source_metadata',context.get('data',{})))
    start,end=capture_window(events)
    acquire_events=_acquisition_events(events)
    acquisitions=[(e['start_ns'],e['end_ns']) for e in acquire_events]
    loads=_intervals(events,{'cpu_load'})
    prep=_intervals(events,{'cpu_prepare','cpu_buffer','cpu_debug_checks','cpu_metadata'})
    submits=_intervals(events,{'cpu_h2d_submit'})
    gpu=[e for e in events if e['kind'].startswith('gpu_')]
    expected_device=str(metadata.get('device',metadata.get('runtime',{}).get('device','unknown')))
    cuda_expected=expected_device.startswith(('cuda','npu'))
    warnings=[]
    if not gpu:warnings.append('未捕获设备活动；设备耗时、未覆盖和重叠均未知，不是零。')
    if cuda_expected and not gpu:warnings.append('请求了加速器但缺少设备事件：采集不完整。')
    if gpu and submits and not any(e['kind']=='gpu_h2d' for e in gpu):
        warnings.append('有CPU H2D调用但缺少设备H2D拷贝事件，覆盖情况未知。')
    if not acquisitions:warnings.append('未记录帧获取标记；驻留输入、合成数据或元信息等待不代表真实存储I/O。')
    if bundle.malformed_events:warnings.append(f'已省略{bundle.malformed_events}条格式异常的选定事件。')
    devices=[]
    for device in sorted(set(str(e['device']) for e in gpu)):
        if device=='npu:ambiguous':
            warnings.append('原生NPU设备身份不明确；保留原始事件，但不合并时间轴或计算设备重叠。')
            continue
        device_events=[e for e in gpu if str(e['device'])==device]
        compute=_intervals(device_events,{'gpu_compute'})
        comm=_intervals(device_events,{'gpu_comm'})
        h2d=_intervals(device_events,{'gpu_h2d'})
        all_gpu=[(e['start_ns'],e['end_ns']) for e in device_events]
        cn=interval_duration(intersect_intervals(compute,comm))
        hc=interval_duration(intersect_intervals(h2d,compute))
        lg=interval_duration(intersect_intervals(acquisitions,all_gpu))
        comm_time=interval_duration(comm);h2d_time=interval_duration(h2d);load_time=interval_duration(acquisitions)
        active=interval_duration(all_gpu)
        devices.append({'device':device,'window_ms':_ms(end-start),'compute_ms':_ms(interval_duration(compute)),
                        'communication_ms':_ms(comm_time),'h2d_ms':_ms(h2d_time),
                        'h2d_activity_present':bool(h2d),
                        'other_copy_ms':_ms(interval_duration(_intervals(device_events,{'gpu_copy_other'}))),
                        'unknown_ms':_ms(interval_duration(_intervals(device_events,{'gpu_unknown'}))),
                        'gpu_active_ms':_ms(active),'gpu_idle_ms':_ms((end-start)-active),
                        'observed_uncovered_ms':_ms((end-start)-active),
                        'visible_transport_ms':_ms(interval_duration(_intervals(device_events,{'gpu_transport'}))),
                        'hbm_counters_available':False,
                        'compute_comm_overlap_ms':_ms(cn),'communication_overlap_ratio':_ratio(cn,comm_time),
                        'communication_without_compute_ms':_ms(difference_duration(comm,compute)),
                        'h2d_compute_overlap_ms':_ms(hc),'h2d_compute_overlap_ratio':_ratio(hc,h2d_time),
                        'load_gpu_overlap_ms':_ms(lg),'load_gpu_overlap_ratio':_ratio(lg,load_time),
                        'load_while_gpu_idle_ms':_ms(load_time-lg),
                        'acquisition_gpu_overlap_ms':_ms(lg),'acquisition_gpu_overlap_ratio':_ratio(lg,load_time),
                        'acquisition_while_gpu_idle_ms':_ms(load_time-lg)})
    if len(devices)>1:warnings.append('同一trace含多个设备；每个设备单独计算重叠。')
    slow=_slow_frames(acquire_events)
    backends=sorted(set(str(e['metadata']['read_backend']) for e in acquire_events if e['metadata'].get('read_backend')))
    if not backends and isinstance(source,dict) and source.get('read_backend'):backends=[source['read_backend']]
    if not backends and loads:backends=['legacy_naive']
    training_profile=analyze_training_bundle(bundle,zoom_ms=zoom_ms)
    training_profile['hotspots']=analyze_hotspots(bundle,training_profile)
    return {'trace':bundle.path,'run_id':metadata.get('run_id'),'hostname':metadata.get('hostname','unknown'),
            'global_rank':metadata.get('global_rank',metadata.get('rank','unknown')),
            'capture_phase':metadata.get('capture_phase','unknown'),'stage':metadata.get('stage','unknown'),
            'cycle':metadata.get('cycle'),
            'global_steps':sorted(set(e['metadata']['global_step'] for e in events if isinstance(e['metadata'].get('global_step'),int))),
            'source_metadata':source,'workload_metadata':context,
            'read_backends':backends,
            'device_adapter':getattr(bundle,'device_adapter',{}),
            'technology':device_technology(metadata,adapter=getattr(bundle,'device_adapter',{})),
            'dispatch_flows':getattr(bundle,'dispatch_flows',[]),
            'training_profile':training_profile,
            'environment':load_environment_record(metadata,bundle.path),
            'check_values':source.get('check_values') if isinstance(source,dict) else None,
            'acquisition_semantics':'frame_acquire parent wall time; uncovered old load_pt fallback; children are not added',
            'start_ns':start,'end_ns':end,'window_ms':_ms(end-start),
            'file_load_wall_ms':_ms(interval_duration(loads)),
            'frame_acquire_wall_ms':_ms(interval_duration(acquisitions)),
            'cpu_data_ready_wall_ms':_ms(interval_duration(acquisitions+prep)),
            'mmap_open_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_map'}))),
            'region_materialize_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_materialize'}))),
            'manifest_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_manifest'}))),
            'cpu_prepare_wall_ms':_ms(interval_duration(prep)),
            'cpu_buffer_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_buffer'}))),
            'cpu_debug_check_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_debug_checks'}))),
            'cpu_metadata_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_metadata'}))),
            'cpu_other_prepare_wall_ms':_ms(interval_duration(_intervals(events,{'cpu_prepare'}))),
            'cpu_h2d_call_wall_ms':_ms(interval_duration(submits)),
            'h2d_tensor_bytes':sum(int(e['metadata'].get('tensor_bytes',0) or 0) for e in events if e['kind']=='cpu_h2d_submit'),
            'file_loads_by_role':_load_statistics(events),
            'frame_acquisitions_by_role':_load_statistics(events,acquisitions=True),
            'slowest_frame_loads':_slow_frames([e for e in events if e['kind']=='cpu_load']),
            'slowest_frame_acquisitions':slow,
            'gpu':devices,'data_quality':{'status':'AMBIGUOUS_DEVICE_IDENTITY' if any(str(e['device'])=='npu:ambiguous' for e in gpu) else 'GPU_CAPTURE_MISSING' if cuda_expected and not gpu else
                                         'CPU_ONLY' if not gpu else 'OBSERVED',
                                         'gpu_activity_present':bool(gpu),'file_load_present':bool(loads),
                                         'frame_acquisition_present':bool(acquisitions),
                                         'selected_events':len(events),'malformed_events':bundle.malformed_events,
                                         'warnings':warnings}}


def _display(value,suffix=' ms'):
    return '未知' if value is None else f'{value:,.2f}{suffix}'


def _rank_title(row):
    steps=row.get('global_steps',[])
    step_text=str(steps) if len(steps)<=4 else f'{steps[0]} ... {steps[-1]}'
    backend='/'.join(row.get('read_backends',[])) or '读取后端未知'
    phase={'startup':'初始化','first_use':'首次更新','train':'训练'}.get(row['capture_phase'],row['capture_phase'])
    return f"rank {row['global_rank']} | {row['hostname']} | {row['technology']['label']} | {backend} | {phase} | 更新 {step_text}"


def _timeline(ax,bundle,row):
    start=row['start_ns'];events=bundle.events
    acquisitions=_acquisition_events(events)
    primary=row['gpu'][0]['device'] if row['gpu'] else None
    for index,(kind,label,color) in enumerate(LANES):
        if kind=='cpu_input_load':
            selected=[e for e in acquisitions if e['metadata'].get('role')=='input']
        elif kind=='cpu_target_load':
            selected=[e for e in acquisitions if e['metadata'].get('role')=='target']
        elif kind=='cpu_other_load':
            selected=[e for e in events if e['kind']=='cpu_manifest']+[
                e for e in acquisitions if e['metadata'].get('role') not in ('input','target')]
        elif kind=='cpu_metadata':
            selected=[e for e in events if e['kind'] in ('cpu_metadata','cpu_prepare')]
        else:
            selected=[e for e in events if e['kind']==kind and
                      (not kind.startswith('gpu_') or str(e['device'])==primary)]
        spans=union_intervals((e['start_ns'],e['end_ns']) for e in selected)
        ax.broken_barh([((a-start)/1e6,(b-a)/1e6) for a,b in spans],(index-.3,.6),facecolors=color,linewidth=0)
        if not selected:
            ax.text(.995,index,'未观察到／未知',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.5,color='#777777')
        if kind=='cpu_target_load':
            # Keep labels only where the actual frame interval has room. The
            # plot never enlarges short IO spans or alters the metric data.
            eligible=[e for e in selected if e['metadata'].get('lead') is not None and
                      (e['end_ns']-e['start_ns'])/1e6>=max(row['window_ms']*.032,.001)]
            for e in sorted(eligible,key=lambda e:e['end_ns']-e['start_ns'],reverse=True)[:12]:
                center=((e['start_ns']+e['end_ns'])/2-start)/1e6
                ax.text(center,index,'L'+str(e['metadata']['lead']),ha='center',va='center',
                        fontsize=7,color='white',fontweight='bold',clip_on=True)
    technology=row['technology']['type']
    device_label=technology if technology in ('GPU','NPU') else '加速器类型未知'
    ax.set_yticks(range(len(LANES)),[label.replace('设备',device_label) for _,label,_ in LANES],fontsize=7.5)
    ax.invert_yaxis();ax.set_xlim(0,max(row['window_ms'],.001));ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('距本次采集开始的毫秒数（本rank时钟）',fontsize=9)
    ax.set_title(_rank_title(row),fontsize=10,loc='left',fontweight='bold')
    if not row['gpu']:
        ax.text(.99,.03,'未捕获设备活动：设备各行均未知',ha='right',va='bottom',
                transform=ax.transAxes,color='#a33138',fontsize=8)


def _make_pdf(bundles,rows,path,*,max_plot_ranks=2,preview_png=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    configure_matplotlib_font()
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    # Never present startup/first-use and steady-state as one merged clock.
    phases=[]
    for row in rows:
        phase=row['capture_phase']
        if phase not in phases:phases.append(phase)
    # Initialization and first-use are distinct timelines on one overview
    # page. Steady training gets its own page; the appendix is shared.
    non_train=[p for p in phases if p!='train']
    non_train.sort(key=lambda p:(p!='first_use',p!='startup',p))
    phase_groups=([non_train] if non_train else [])+([['train']] if 'train' in phases else [])
    previews=[];page=0
    with PdfPages(path) as pdf:
        environment_previews,page=render_environment_intro(pdf,
            [{'label':f"测试环境 {index+1}", 'record':record}
             for index,record in enumerate(unique_environments(rows))],
            preview_prefix=path.with_suffix('') if preview_png else None)
        previews.extend(environment_previews)
        for group in phase_groups:
            phase=' / '.join(group)
            candidates=[i for i,r in enumerate(rows) if r['capture_phase'] in group]
            # Pick a representative trace and the slowest window, visibly
            # identify their ranks rather than inventing an averaged timeline.
            by_time=sorted(candidates,key=lambda i:rows[i]['window_ms'])
            if len(group)>1:
                selected=[]
                for capture in group:
                    current=sorted((i for i in candidates if rows[i]['capture_phase']==capture),key=lambda i:rows[i]['window_ms'])
                    selected.append(current[-1])
            else:
                selected=[by_time[len(by_time)//2]]
                if by_time[-1] not in selected:selected.append(by_time[-1])
            selected=selected[:max_plot_ranks]
            fig=plt.figure(figsize=(15,11 if len(selected)>1 else 8.5));grid=fig.add_gridspec(len(selected)+1,1,height_ratios=[.9]+[2.1]*len(selected),hspace=.6)
            ax=fig.add_subplot(grid[0]);ax.axis('off')
            ax.text(0,1,'TERRA：I/O、设备活动与通信概览',fontsize=17,fontweight='bold',va='top')
            selection='各阶段最长窗口' if len(group)>1 else '中位窗口与最长窗口'
            phase_label=' / '.join({'startup':'初始化','first_use':'首次更新','train':'训练'}.get(p,p) for p in group)
            ax.text(0,.61,f'采集：{phase_label} | {len(candidates)}份独立rank／周期trace | {selection}',fontsize=10,va='top')
            reference=rows[selected[0]];context=reference.get('workload_metadata',{});source=reference.get('source_metadata',{})
            recorded=profile_context(bundles[selected[0]].metadata)
            shape=source.get('shape_chw',source.get('shape',recorded['shape_chw'] or '未知')) if isinstance(source,dict) else recorded['shape_chw'] or '未知'
            mp=recorded['dmp_degree'] or '未知'
            check_mode=reference.get('check_values')
            checks='开启' if check_mode is True else '关闭' if check_mode is False else '历史记录／未知'
            ax.text(0,.38,f"输入CHW：{shape} | 预测步数K：{recorded['rollout_steps'] or '未知'} | 精度：{recorded['precision'] or '未知'} | MP进程数：{mp} | CPU数值检查：{checks}",fontsize=9,va='top')
            descriptions=[]
            for i in selected:
                r=rows[i];g=r['gpu'][0] if r['gpu'] else {}
                ratio=g.get('communication_overlap_ratio')
                roles=r['frame_acquisitions_by_role']
                descriptions.append(f"rank {r['global_rank']}：窗口 {_display(r['window_ms'])}；获取 {_display(r['frame_acquire_wall_ms'] if r['data_quality']['frame_acquisition_present'] else None)}；"
                                    f"输入 {_display(roles.get('input',{}).get('wall_ms'))}，标签 {_display(roles.get('target',{}).get('wall_ms'))}；"
                                    f"设备H2D {_display(g.get('h2d_ms') if g.get('h2d_activity_present') else None)}；设备未覆盖 {_display(g.get('observed_uncovered_ms'))}；"
                                    f"通信／计算重叠 {'未知' if ratio is None else format(ratio,'.1%')}")
            ax.text(0,.13,'\n'.join(descriptions),fontsize=8.5,va='top',linespacing=1.4)
            for index,i in enumerate(selected):_timeline(fig.add_subplot(grid[index+1]),bundles[i],rows[i])
            fig.subplots_adjust(left=.22,right=.98,top=.95,bottom=.10)
            fig.text(.02,.027,'获取包括打开／加载与局部物化；子范围有嵌套，不能相加。mmap缺页发生在访问时，不是纯磁盘耗时。',fontsize=8,color='#444444')
            fig.text(.02,.010,'CPU H2D调用不等于设备拷贝；缺失事件不是零成本，未覆盖不是物理空闲。缓存条件仅以已有证据说明。',fontsize=8,color='#444444')
            pdf.savefig(fig);page+=1
            if preview_png:
                preview=path.with_name(f'{path.stem}.page{page}.png');fig.savefig(preview,dpi=140);previews.append(str(preview))
            plt.close(fig)
        # One compact appendix covers all ranks without creating one page per
        # rank. Detailed full-resolution values remain in JSON/CSV.
        fig,axes=plt.subplots(2,1,figsize=(15,9),gridspec_kw={'height_ratios':[2.4,1]});axes[0].axis('off');axes[1].axis('off')
        axes[0].set_title('各rank采集统计与数据完整性',loc='left',fontsize=15,fontweight='bold',pad=15)
        headers=['rank／阶段','窗口ms','获取ms','准备ms','设备H2D ms','设备未覆盖ms','通信重叠','采集状态']
        cells=[]
        for r in rows[:22]:
            g=r['gpu'][0] if r['gpu'] else {};ratio=g.get('communication_overlap_ratio')
            cells.append([f"{r['global_rank']} / {r['capture_phase']}",_display(r['window_ms'],''),
                          _display(r['frame_acquire_wall_ms'] if r['data_quality']['frame_acquisition_present'] else None,''),_display(r['cpu_prepare_wall_ms'],''),
                          _display(g.get('h2d_ms') if g.get('h2d_activity_present') else None,''),_display(g.get('gpu_idle_ms'),''),
                          '未知' if ratio is None else format(ratio,'.1%'),{'OBSERVED':'已观测','CPU_ONLY':'仅CPU','GPU_CAPTURE_MISSING':'设备采集缺失'}.get(r['data_quality']['status'],'未知')])
        table=axes[0].table(cellText=cells,colLabels=headers,cellLoc='left',colLoc='left',loc='upper center',
                            colWidths=[.17,.09,.09,.09,.12,.11,.11,.22])
        table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.45)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r==0:cell.set_facecolor('#e6eef5');cell.set_text_props(fontweight='bold')
        notes=['口径与覆盖范围：',
               '- 每行是独立rank／周期，不跨节点或trace求绝对时间交集。',
               '- 获取采用frame_acquire父范围；旧trace缺少父范围时使用load_pt。',
               '- 完整加载／映射／物化是嵌套子范围，不能再加到获取耗时。',
               '- mmap局部物化含缺页I/O及拷贝／转换，打开映射的时间不是读取时间。',
               '- 缓冲／补齐属于准备；可选数值调试与轻量元信息检查分开记录。',
               '- 文件大小与H2D张量字节是逻辑元信息，不是实际存储／总线流量。',
               '- 设备计算包含打包、索引和转换；不全部属于模型稠密FLOPs。',
               '- 设备未覆盖不是物理空闲；暴露通信也不直接等于该通信造成的整步损失。',
               '- 原trace保留；详细事件和慢帧见summary.json与events.csv。']
        warnings=sorted(set(w for r in rows for w in r['data_quality']['warnings']))
        notes.extend('- '+w for w in warnings[:3])
        slow=sorted(((r,e) for r in rows for e in r['slowest_frame_acquisitions']),key=lambda item:item[1]['duration_ms'],reverse=True)
        for r,e in slow[:2]:
            file=Path(e['path']).name if e['path'] else 'frame '+str(e['frame_id'])
            if len(file)>45:file=file[:42]+'...'
            notes.append(f"- 较慢获取：rank {r['global_rank']} / {r['capture_phase']} / {e['role']} 预测步 {e['lead']}："
                         f"{_display(e['duration_ms'])}; {file}")
        if len(rows)>22:notes.append(f'- 此表展示22/{len(rows)}行，JSON包含所有rank／周期。')
        axes[1].text(0,1,'\n'.join(notes),va='top',fontsize=8,linespacing=1.45,wrap=True)
        fig.subplots_adjust(left=.035,right=.97,top=.94,bottom=.04,hspace=.12)
        pdf.savefig(fig);page+=1
        if preview_png:
            preview=path.with_name(f'{path.stem}.page{page}.png');fig.savefig(preview,dpi=140);previews.append(str(preview))
        plt.close(fig)
    return previews


def summarize_paths(trace_paths,output_dir,*,render_pdf=True,preview_png=False,max_plot_ranks=2,max_input_mb=512,
                    zoom_ms=None,extra_figures=True):
    """Analyze trace files/directories. No distributed or model calls occur.

    All normalized events are written as CSV. Summary JSON contains scalars,
    per-device overlap definitions, data quality, and slow-frame metadata.
    Existing output files are refused, keeping previous experiments intact.
    """
    if max_plot_ranks<1:raise ValueError('max_plot_ranks must be positive')
    zoom_ms=parse_zoom_ms(zoom_ms)
    paths=discover_traces(trace_paths);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    json_path=out/'summary.json';csv_path=out/'events.csv';pdf_path=out/'overview.pdf'
    hotspots_json=out/'hotspots.json';hotspots_csv=out/'hotspots.csv'
    if any(p.exists() for p in (json_path,csv_path,hotspots_json,hotspots_csv)) or (render_pdf and pdf_path.exists()):
        raise FileExistsError('Summary outputs already exist; choose a new output directory')
    bundles=[read_trace(p,max_input_mb=max_input_mb) for p in paths]
    rows=[summarize_bundle(bundle,zoom_ms=zoom_ms) for bundle in bundles]
    report={'format':'terra_pytorch_io_summary_v2','status':'COMPLETE',
            'timing_units':'milliseconds; normalized events.csv stores rank-local nanoseconds',
            'definitions':{'frame_load':'CPU wall time of original torch.load, including cache, deserialization and allocation',
                           'frame_acquire':'Parent frame_acquire wall time from open/load to local materialization; old traces fall back to load_pt',
                           'mmap_open':'Mapping creation only; most payload access may occur later in region_materialize',
                           'region_materialize':'Local data access and copy/conversion, including mmap page-fault IO; not pure disk service time',
                           'cpu_data_ready':'Interval union of acquisition and CPU preparation, not the sum of nested scopes',
                           'cpu_prepare':'Union of buffer/padding, optional debug checks, metadata and unspecified legacy preparation',
                           'cpu_debug_checks':'Optional CPU value scans/masking, enabled by data.check_values; not required in normal training',
                           'cpu_buffer':'Local buffer allocation/reuse and padding initialization/assembly; padding only initializes on new buffers',
                           'legacy_fields':'file_load_wall_ms/file_loads_by_role keep leaf-load semantics; load_gpu_overlap uses canonical acquisition intervals',
                           'gpu_compute':'Observed CUDA/NPU compute tasks, including packing/casts/indexing; native communication aliases excluded',
                           'communication_overlap':'union(device collective envelopes) intersect union(compute tasks) / union(collective envelopes); includes communication waits',
                           'h2d_compute_overlap':'union(GPU H2D) intersect union(non-NCCL kernels) / union(GPU H2D)',
                           'load_gpu_overlap':'union(CPU acquisition) intersect union(all observed GPU work) / union(CPU acquisition)',
                           'gpu_idle':'Legacy alias of observed_uncovered_ms: capture wall minus observed device-event union, NOT physical idle; unavailable if device activities missing',
                           'gpu_transport':'Visible native communication SDMA/transport tasks with aliases deduplicated; never a complete HBM traffic measurement',
                           'cpu_operator':'Real CPU operator scopes, including waits; multi-thread interval unions are not CPU busy time',
                           'cpu_runtime':'Recorded CPU CANN/CUDA runtime/driver API calls, including launch/sync; not actual accelerator execution',
                           'host_file_io_partition':'Inside each selected update, canonical application file acquisition has priority over nested forward/loss/other CPU phases; all host categories remain mutually exclusive and fill the step',
                           'dispatch_flows':'Only exact native async/correlation evidence; missing or ambiguous associations remain unknown, never nearest-time guesses',
                           'hotspots':'Per trace/device/causal phase, actual task elapsed sums and native HCOM wait-envelope sums have separate denominators; CPU self wall excludes nested same-thread operators, includes waits, and is not device time',
                           'cross_rank':'No absolute clock alignment or cross-rank interval intersections are assumed'},
            'traces':rows,'environments':unique_environments(rows),
            'paths':{'json':str(json_path),'csv':str(csv_path),'pdf':str(pdf_path) if render_pdf else None,
                     'hotspots_json':str(hotspots_json),'hotspots_csv':str(hotspots_csv)}}
    if any(r['data_quality']['status']=='GPU_CAPTURE_MISSING' for r in rows):report['status']='INCOMPLETE_GPU_CAPTURE'
    fields=['source_trace','run_id','hostname','global_rank','capture_phase','event_id','kind','name','label',
            'start_ns','end_ns','duration_ms','device','stream','pid','tid','correlation','metadata_json']
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for bundle,row in zip(bundles,rows):
            for event in bundle.events:
                item={key:event.get(key) for key in fields if key in event}
                item.update(source_trace=bundle.path,run_id=row['run_id'],hostname=row['hostname'],
                            global_rank=row['global_rank'],capture_phase=row['capture_phase'],
                            duration_ms=_ms(event['end_ns']-event['start_ns']),
                            metadata_json=json.dumps(event['metadata'],sort_keys=True))
                writer.writerow(item)
    hotspot_fields=['source_trace','global_rank','cycle','window_start_ns','window_end_ns',
                    'table_kind','phase','device','task_kind','name','display_name','count',
                    'communication_category',
                    'duration_sum_ns','duration_union_ns','inclusive_duration_sum_ns',
                    'denominator_ns','share_percent','evidence_counts_json','example_event_ids_json',
                    'source_range_counts_json','raw_name_counts_json']
    with hotspots_csv.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=hotspot_fields);writer.writeheader()
        for row in rows:
            hotspots=row['training_profile']['hotspots'];window=hotspots.get('window',{})
            for record in hotspots.get('rows',[]):
                item={key:record.get(key) for key in hotspot_fields if key in record}
                item.update(source_trace=row['trace'],global_rank=row['global_rank'],cycle=row.get('cycle'),
                            window_start_ns=window.get('start_ns'),window_end_ns=window.get('end_ns'),
                            evidence_counts_json=json.dumps(record.get('evidence_counts',{}),sort_keys=True),
                            example_event_ids_json=json.dumps(record.get('example_event_ids',[])),
                            source_range_counts_json=json.dumps(record.get('source_range_counts',{}),sort_keys=True),
                            raw_name_counts_json=json.dumps(record.get('raw_name_counts',{}),sort_keys=True))
                writer.writerow(item)
    hotspots_json.write_text(json.dumps({'format':'terra_operator_hotspot_collection_v1',
        'clock_scope':'Each trace/device is independent; elapsed sums are not step wall fractions',
        'traces':[{'trace':row['trace'],'global_rank':row['global_rank'],'cycle':row.get('cycle'),
                   'hotspots':row['training_profile']['hotspots']} for row in rows]},
        indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if render_pdf:
        report['paths']['previews']=_make_pdf(bundles,rows,pdf_path,max_plot_ranks=max_plot_ranks,preview_png=preview_png)
        if extra_figures:
            import re
            report['paths']['training_figures']=[]
            for index,(bundle,row) in enumerate(zip(bundles,rows)):
                rank=re.sub(r'[^A-Za-z0-9_.-]','_',str(row['global_rank']))
                cycle=re.sub(r'[^A-Za-z0-9_.-]','_',str(row.get('cycle','unknown')))
                prefix=out/('trace_%03d.rank%s.cycle%s' % (index,rank,cycle))
                figures=render_training_figures(bundle,row['training_profile'],prefix)
                from .hotspot_figures import render_hotspot_figures
                figures['hotspots']=render_hotspot_figures(bundle,row['training_profile'],row['training_profile']['hotspots'],prefix)
                from .timeline_html import write_timeline_html
                figures['timeline_html']=write_timeline_html(bundle,row['training_profile'],prefix)
                report['paths'].setdefault('interactive_timelines',[]).append(figures['timeline_html'])
                figures['trace']=bundle.path
                report['paths']['training_figures'].append(figures)
    json_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
