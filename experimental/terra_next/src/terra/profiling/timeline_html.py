"""Self-contained host/device trace viewer, exported by the existing harness.

All event times are converted to relative milliseconds only after subtracting
the rank-local integer origin. Associations come from explicit profiler flow
or correlation evidence; visual proximity never creates a dispatch edge.
"""
import html
import json
from pathlib import Path


def build_timeline_data(bundle, analysis):
    from .figures import device_technology, file_io_events, cpu_display_rows
    selected = analysis['selected_update']
    origin = min(e['start_ns'] for e in bundle.events)
    end = max(e['end_ns'] for e in bundle.events)
    technology = device_technology(bundle.metadata, adapter=getattr(bundle, 'device_adapter', {}))
    hotspots = analysis.get('hotspots')
    if hotspots is None:
        from .hotspots import analyze_hotspots
        hotspots = analyze_hotspots(bundle, analysis)
    label = technology['label']
    file_data = file_io_events(bundle.events, bundle.metadata)
    file_ids = {e.get('event_id', e['label']) for e in file_data['events']}
    tracks, values, track_lookup = [], [], {}
    phase_names = {'terra/train/forward', 'terra/train/loss', 'terra/train/backward',
                   'terra/train/gradient_sync', 'terra/train/gradient_clip',
                   'terra/train/unscale', 'terra/train/finite_check',
                   'terra/train/optimizer', 'terra/profile/final_drain'}
    colors = {'phase': '#64748b', 'cpu_op': '#765eb0', 'cpu_runtime': '#d39b2d',
              'file_input': '#2e9e75', 'file_target': '#299ab0', 'file_other': '#408b80',
              'prepare': '#77969e', 'submit': '#bc9b47', 'generate': '#999299',
              'compute': '#347dae', 'h2d': '#32a891', 'gradient': '#e79b3e',
              'halo': '#9271b4', 'communication': '#a0acba', 'transport': '#24a79a',
              'copy': '#7894a5', 'unknown': '#b76678'}

    def track(key, title, group, color, order):
        if key not in track_lookup:
            track_lookup[key] = len(tracks)
            tracks.append({'key': key, 'label': title, 'group': group, 'color': color,
                           'order': order})
        return track_lookup[key]

    for index, event in enumerate(bundle.events):
        kind, metadata = event['kind'], event.get('metadata', {})
        identity = event.get('event_id', 'event:%d' % index)
        event_file_id = event.get('event_id', event['label'])
        thread = '%s/%s' % (event.get('pid'), event.get('tid'))
        device = str(event.get('device'))
        stream = str(event.get('stream', event.get('tid')))
        group, color, order = 'CPU', None, 0
        if kind == 'cpu_phase' and event['name'] in phase_names:
            key, title, color = 'phase:' + thread, 'CPU 训练阶段（提交／等待） · 线程 ' + thread, colors['phase']
        elif kind in ('cpu_op', 'cpu_runtime'):
            key = kind + ':' + thread
            title = ('CPU 算子调用（可含等待）' if kind == 'cpu_op' else
                     'CPU 运行时调用／提交') + ' · 线程 ' + thread
            color, order = colors[kind], 1 if kind == 'cpu_op' else 2
        elif event_file_id in file_ids:
            role = metadata.get('role')
            role = role if role in ('input', 'target') else 'other'
            key, title = 'file_' + role, {'input': 'CPU 文件 I/O：输入读取',
                'target': 'CPU 文件 I/O：标签读取', 'other': 'CPU 文件 I/O：其他读取'}[role]
            color, order = colors['file_' + role], 3 + {'input': 0, 'target': .1, 'other': .2}[role]
        elif kind in ('cpu_prepare', 'cpu_buffer', 'cpu_materialize', 'cpu_metadata', 'cpu_debug_checks'):
            key, title, color, order = 'prepare', 'CPU 数据准备／局部物化', colors['prepare'], 4
        elif kind == 'cpu_h2d_submit':
            key, title, color, order = 'submit', 'CPU H2D 调用（可含等待）', colors['submit'], 5
        elif kind == 'cpu_fake_generate':
            key, title, color, order = 'generate', 'CPU 合成数据生成（非文件 I/O）', colors['generate'], 6
        elif kind.startswith('gpu_'):
            group, order = label, 10
            if kind == 'gpu_compute':
                category, text = 'compute', '实际计算核（含打包／转换）'
            elif kind == 'gpu_h2d':
                category, text = 'h2d', '实际 H2D 拷贝'
            elif kind == 'gpu_comm':
                category = metadata.get('communication_category', 'unknown')
                if metadata.get('source_flow_exact') is not True:
                    category = 'unknown'
                category, text = {'gradient': ('gradient', '梯度通信包络'),
                    'halo': ('halo', 'Halo 通信包络')}.get(category, ('communication', '其他／未知通信包络'))
            else:
                category, text = {'gpu_transport': ('transport', '通信搬运执行'),
                    'gpu_copy_other': ('copy', '独立拷贝／填充执行')}.get(kind, ('unknown', '未分类活动'))
            key = '%s:%s:%s:%s' % (device, category, stream, group)
            title, color = '%s %s · %s · 流 %s' % (label, text, device, stream), colors[category]
        else:
            continue
        track_index = track(key, title, group, color, order)
        values.append({'id': identity, 'track': track_index, 'kind': kind, 'name': event['name'],
            'start': (event['start_ns'] - origin) / 1e6,
            'end': (event['end_ns'] - origin) / 1e6,
            'duration': (event['end_ns'] - event['start_ns']) / 1e6,
            'pid': event.get('pid'), 'tid': event.get('tid'), 'device': event.get('device'),
            'stream': event.get('stream'), 'metadata': metadata})
    order = sorted(range(len(tracks)), key=lambda i: (tracks[i]['order'], tracks[i]['label']))
    remap = {old: new for new, old in enumerate(order)}
    tracks = [tracks[index] for index in order]
    for value in values:
        value['track'] = remap[value['track']]
    known = {e['id'] for e in values}
    flows = []
    for source in getattr(bundle, 'dispatch_flows', []):
        if source.get('exact') is not True:
            continue
        row = dict(source)
        for key in ('source_ts_ns', 'target_ts_ns'):
            row[key[:-3] + '_ms'] = (row[key] - origin) / 1e6 if row.get(key) is not None else None
        row['drawable'] = row.get('source_event_id') in known and row.get('target_event_id') in known
        flows.append(row)
    return {'format': 'terra_host_device_timeline_v1', 'trace': bundle.path,
        'hostname': bundle.metadata.get('hostname', '未知'),
        'rank': bundle.metadata.get('global_rank', bundle.metadata.get('rank', '未知')),
        'cycle': bundle.metadata.get('cycle'), 'technology': technology,
        'workload': analysis.get('recorded_context', {}),
        'clock_scope': 'single_trace_rank_only', 'origin_ns': str(origin),
        'range': [0, (end - origin) / 1e6],
        'step': [(selected['start_ns'] - origin) / 1e6, (selected['end_ns'] - origin) / 1e6],
        'step_wall_ms': selected['wall_ms'],
        'cpu_update_complete': selected['cpu_update_complete'],
        'device_completion_observed': selected['device_completion_observed'],
        'file_io_status': file_data['status'],
        'file_io': selected.get('file_io', {}),
        'cpu_breakdown': cpu_display_rows(selected),
        'hotspots': hotspots,
        'coverage': {'cpu_operators_observed': any(e['kind']=='cpu_op' for e in values),
                     'cpu_runtime_observed': any(e['kind']=='cpu_runtime' for e in values),
                     'actual_compute_observed': any(e['kind']=='gpu_compute' for e in values)},
        'tracks': tracks, 'events': values, 'flows': flows,
        'counts': {'events': len(values), 'exact_flows': len(flows),
                   'drawable_flows': sum(row['drawable'] for row in flows)},
        'semantics': 'CPU记录为算子调用、运行时提交与等待墙钟；GPU/NPU记录为实际执行。'
                     '文件I/O为应用读取，含缓存、反序列化和分配；通信包络含等待。'
                     '仅精确flow/correlation关联，缺失不按邻近猜测；各层不可累加为单步。'}


def write_timeline_html(bundle, analysis, prefix):
    payload = build_timeline_data(bundle, analysis)
    output = Path(prefix).with_name(Path(prefix).name + '.timeline.html')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    embedded = embedded.replace('&', r'\u0026').replace('<', r'\u003c').replace('>', r'\u003e')
    title = ('CPU 算子与 I/O 时间线' if payload['technology']['type']=='CPU' else
             'CPU → %s 算子与 I/O 时间线' % payload['technology']['label'])
    text = _HTML.replace('__TRACE_TITLE__', html.escape(title))
    output.write_text(text.replace('__TRACE_DATA__', embedded), encoding='utf-8')
    return {'html': str(output), **payload['counts'], 'clock_scope': payload['clock_scope'],
            'file_io_status': payload['file_io_status'], 'technology': payload['technology']}


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TRACE_TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#162536;font:14px "Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif}
header{background:white;padding:18px 24px 12px;border-bottom:1px solid #d9e0e8}h1{font-size:23px;margin:0 0 8px}
.muted{color:#596b7e;font-size:12px;line-height:1.65}.toolbar{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:12px 0 6px}
button,input,select{font:inherit;border:1px solid #cad4df;background:white;border-radius:5px;padding:6px 10px;color:#243c53}
button{cursor:pointer}button:hover{background:#eaf1f9}input{width:100px}label{font-size:12px}.badge{display:inline-block;background:#edf3fa;padding:3px 8px;border-radius:4px;margin-right:6px}
#viewport{height:65vh;min-height:420px;overflow:auto;background:white;border-bottom:1px solid #cad4df;position:relative}
canvas{display:block}#tip{position:fixed;pointer-events:none;max-width:580px;background:#162536;color:white;padding:8px 11px;border-radius:5px;white-space:pre-wrap;display:none;font-size:12px;z-index:3}
#details{background:white;margin:12px 18px;padding:14px 18px;border-radius:7px}#selection{font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.55 Consolas,monospace;max-height:230px;overflow:auto}
.note{padding:10px 22px;background:#eff5fa;color:#425b73;font-size:12px;line-height:1.65}
#breakdown{margin-top:10px;border-top:1px solid #e2e8ef;padding-top:9px}summary{cursor:pointer;font-weight:600}
#phasegrid{display:grid;grid-template-columns:1fr 1fr;gap:8px 28px;margin:10px 0 6px}.phase{display:grid;grid-template-columns:170px 1fr 100px;gap:8px;align-items:center;font-size:12px}.phase-track{height:8px;background:#e8edf3;border-radius:4px}.phase-fill{height:8px;border-radius:4px}.phase-value{text-align:right}
@media(max-width:950px){#phasegrid{grid-template-columns:1fr}}
#hotspots{margin-top:10px;border-top:1px solid #e2e8ef;padding-top:9px}
#hotspot-search{width:220px}.hotspot-table{width:100%;border-collapse:collapse;table-layout:fixed;margin:8px 0;font-size:12px}
.hotspot-table th,.hotspot-table td{padding:7px 9px;border-bottom:1px solid #e5eaf0;text-align:right}
.hotspot-table th:first-child,.hotspot-table td:first-child{text-align:left;width:60%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hotspot-table th{background:#f0f4f9}.hotspot-swatch{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:7px}
</style></head><body>
<header><h1>__TRACE_TITLE__</h1><div id="identity"></div><div id="stats" class="muted"></div>
<div class="toolbar"><button id="step">本步全览</button><button id="capture">完整采集</button>
<button id="zin">放大</button><button id="zout">缩小</button>
<label>开始（毫秒） <input id="start" type="number" step="any"></label>
<label>结束（毫秒） <input id="end" type="number" step="any"></label><button id="apply">应用范围</button>
<button id="related">仅查看所选算子的关联轨道</button><button id="all">恢复全部轨道</button>
<label><input id="arrows" type="checkbox" checked style="width:auto">显示所选算子的精确关联</label></div>
<div class="muted">单击 CPU 调用或 GPU/NPU 执行条查看来源与关联；拖动平移，Ctrl＋滚轮缩放，Shift＋滚轮平移。各线程／流独立排布，嵌套调用保留层级。</div>
<details id="breakdown" open><summary id="breakdown-title">单步耗时拆解：CPU 主机阶段与文件 I/O</summary><div id="phasegrid"></div><div id="breakdown-note" class="muted">各分项互斥并覆盖完整主机更新；含提交与等待，不与下方 GPU／NPU 实际执行时长相加。文件 I/O 从嵌套阶段扣除，原始标记保留。</div></details>
<details id="hotspots"><summary>前向／反向热点算子（含通信）</summary>
<div class="toolbar">
<label>阶段 <select id="hotspot-phase"><option value="forward">模型前向</option><option value="backward_with_finalize">反向与梯度规约收尾</option><option value="backward">仅反向</option><option value="gradient_finalize">仅梯度规约收尾</option><option value="unknown">阶段未知</option></select></label>
<label>口径 <select id="hotspot-kind"><option value="device_tasks">实际设备任务</option><option value="communication_envelopes">通信包络（含等待）</option><option value="cpu_self">CPU 独占调用墙钟</option></select></label>
<label>设备 <select id="hotspot-device"></select></label>
<input id="hotspot-search" type="search" placeholder="按算子名称筛选" aria-label="筛选热点算子">
</div><div id="hotspot-denominator" class="muted"></div>
<table class="hotspot-table"><thead><tr><th>算子／任务</th><th>累计耗时（毫秒）</th><th>占比</th><th>记录数</th></tr></thead><tbody id="hotspot-rows"></tbody></table>
<div id="hotspot-note" class="muted"></div></details></header>
<div id="viewport"><canvas id="timeline"></canvas></div><div id="tip"></div>
<div class="note" id="semantics"></div><section id="details"><div id="selection">选择任一算子查看提交与执行关系</div><pre id="detail"></pre></section>
<script id="trace-data" type="application/json">__TRACE_DATA__</script>
<script>
(function(){
'use strict';
const D=JSON.parse(document.getElementById('trace-data').textContent);
const el=id=>document.getElementById(id), canvas=el('timeline'), viewport=el('viewport'), ctx=canvas.getContext('2d');
const byId=new Map(D.events.map(e=>[e.id,e])), grouped=new Map();
let left=D.step[0],right=Math.max(D.step[1],D.step[0]+.001),selected=null,onlyTracks=null,hit=[],layout=[],drag=null;
const labelWidth=350,top=35,barHeight=17,gap=3;
for(const e of D.events){if(!grouped.has(e.track))grouped.set(e.track,[]);grouped.get(e.track).push(e);}
for(const events of grouped.values()){
 events.sort((a,b)=>a.start-b.start||b.end-a.end);
 const ends=[];
 for(const e of events){let depth=ends.findIndex(x=>x<=e.start);if(depth<0)depth=ends.length;ends[depth]=e.end;e.depth=depth;}
}
const ioText={observed:'已捕获真实文件读取',not_applicable_resident:'驻留输入：本次不适用文件 I/O',not_applicable_fake:'合成输入：本次不适用文件 I/O',unknown:'文件 I/O 覆盖未知'};
const workload=D.workload||{},shape=workload.shape_bchw,chw=workload.shape_chw;
const workloadLabel=[workload.model_label,workload.model_depth?workload.model_depth+'层':null,
 shape?'B'+shape[0]:null,chw?'C×H×W='+chw.join('×'):null,
 workload.world_size?'P'+workload.world_size:null,workload.precision?String(workload.precision).toUpperCase():null].filter(Boolean).join(' · ');
el('identity').textContent=workloadLabel+(workloadLabel?' · ':'')+'节点 '+D.hostname+' · rank '+D.rank+' · 周期 '+D.cycle+' · '+D.technology.label+(D.cpu_update_complete?' · 主机单步 ':' · 采集窗口 ')+D.step_wall_ms.toFixed(3)+' 毫秒'+(D.device_completion_observed?'（含步末同步）':'（设备完成边界未知）');
if(!D.cpu_update_complete){el('step').textContent='所选窗口';el('breakdown-title').textContent='主机阶段拆解：完整单步边界未知';el('breakdown-note').textContent='各分项覆盖所选采集窗口；缺少完整更新边界，不能作为完整单步耗时。CPU 主机时间与 GPU／NPU 实际执行时间分开解读。';}
const missing=[];
if(!D.coverage.cpu_operators_observed)missing.push('CPU 算子未捕获');
if(!D.coverage.cpu_runtime_observed)missing.push('CPU 运行时调用未捕获');
if(D.technology.type!=='CPU'&&!D.coverage.actual_compute_observed)missing.push(D.technology.label+' 计算核未捕获');
const io=D.file_io,ioCost=D.file_io_status==='observed'&&io.observed_in_selected_window?
 '；本步文件获取 '+io.wall_ms.toFixed(3)+' 毫秒（输入 '+io.input_wall_ms.toFixed(3)+'／标签 '+io.target_wall_ms.toFixed(3)+'）':'';
el('stats').textContent=D.counts.events+' 个事件，'+D.counts.exact_flows+' 条精确提交关联（可绘制 '+D.counts.drawable_flows+' 条）；'+(ioText[D.file_io_status]||D.file_io_status)+ioCost+'。'+(missing.length?' 覆盖缺失：'+missing.join('；')+'。':'');
el('semantics').textContent=D.semantics;
const phaseMax=Math.max(.001,...D.cpu_breakdown.map(row=>row.wall_ms));
for(const row of D.cpu_breakdown){
 const block=document.createElement('div');block.className='phase';
 const name=document.createElement('span');name.textContent=row.label;
 const track=document.createElement('div');track.className='phase-track';
 const fill=document.createElement('div');fill.className='phase-fill';fill.style.width=(row.wall_ms/phaseMax*100)+'%';
 fill.style.background=row.key==='file_io'?'#765eb0':row.key==='gradient_sync'?'#e79b3e':row.key==='other'?'#a0acba':'#347dae';track.append(fill);
 const value=document.createElement('span');value.className='phase-value';value.textContent=row.recorded?row.wall_ms.toFixed(3)+' 毫秒':'未观察到／未知';
 block.append(name,track,value);el('phasegrid').append(block);
}
const hotspotRows=D.hotspots&&Array.isArray(D.hotspots.rows)?D.hotspots.rows:[];
if(hotspotRows.some(r=>r.table_kind==='device_tasks'&&r.task_kind==='gpu_comm')){
 const option=document.createElement('option');option.value='communication_kernels';option.textContent=D.technology.type==='GPU'?'GPU 通信内核':'实际通信内核';
 if(!hotspotRows.some(r=>r.table_kind==='communication_envelopes'))el('hotspot-kind').options[1].replaceWith(option);
 else el('hotspot-kind').append(option);
}
const phaseNames={loss:'损失计算',optimizer:'优化器更新',checks:'检查',finite_check:'有限性检查',gradient_clip:'梯度裁剪',unscale:'梯度反缩放',data_io:'数据获取',other:'其他阶段',other_update:'更新内其他阶段',final_drain:'末尾设备同步'};
const presetPhases=new Set([...el('hotspot-phase').options].map(o=>o.value));
for(const phase of [...new Set(hotspotRows.map(r=>r.phase))].sort()){
 if(presetPhases.has(phase))continue;
 const option=document.createElement('option');option.value=phase;option.textContent=phaseNames[phase]||phase;el('hotspot-phase').append(option);
}
function hotspotDevice(row){return row.device===null||row.device===undefined?'CPU':String(row.device);}
function rowsForHotspotKind(kind){return hotspotRows.filter(r=>kind==='communication_kernels'?r.table_kind==='device_tasks'&&r.task_kind==='gpu_comm':r.table_kind===kind);}
function refreshHotspotDevices(){
 const previous=el('hotspot-device').value,kind=el('hotspot-kind').value;
 const devices=[...new Set(rowsForHotspotKind(kind).map(hotspotDevice))].sort();
 el('hotspot-device').replaceChildren();
 for(const device of devices){const option=document.createElement('option');option.value=device;option.textContent=device;el('hotspot-device').append(option);}
 if(devices.includes(previous))el('hotspot-device').value=previous;
 renderHotspots();
}
function renderHotspots(){
 const kind=el('hotspot-kind').value,phase=el('hotspot-phase').value,device=el('hotspot-device').value;
 if(kind==='cpu_self'&&D.hotspots.cpu_self_policy&&D.hotspots.cpu_self_policy.available===false){
  el('hotspot-rows').replaceChildren();const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='源记录存在交叉 CPU 调用区间，当前独占调用墙钟无法可靠计算。';tr.append(td);el('hotspot-rows').append(tr);el('hotspot-denominator').textContent='独占调用口径不可用，详见导出 JSON 中的 cpu_self_policy。';el('hotspot-note').textContent='未将重叠调用重复累计为可靠独占时间。';return;
 }
 const phases=phase==='backward_with_finalize'?new Set(['backward','gradient_finalize']):new Set([phase]);
 const all=rowsForHotspotKind(kind).filter(r=>hotspotDevice(r)===device);
 const selectedRows=all.filter(r=>phases.has(r.phase)),groups=new Map();
 for(const row of selectedRows){
  const name=row.display_name||row.name,key=String(row.task_kind||'')+'\u0000'+name;
  if(!groups.has(key))groups.set(key,{name,task_kind:row.task_kind,duration:0,count:0});
  const group=groups.get(key);group.duration+=Number(row.duration_sum_ns)||0;group.count+=Number(row.count)||0;
 }
 const rows=[...groups.values()].sort((a,b)=>b.duration-a.duration||a.name.localeCompare(b.name));
 const denominator=rows.reduce((s,r)=>s+r.duration,0),total=all.reduce((s,r)=>s+Number(r.duration_sum_ns||0),0);
 const unknown=all.filter(r=>r.phase==='unknown').reduce((s,r)=>s+Number(r.duration_sum_ns||0),0);
 const search=el('hotspot-search').value.trim().toLowerCase(),filtered=rows.filter(r=>r.name.toLowerCase().includes(search));
 const visible=filtered.slice(0,20);
 if(!search&&filtered.length>20)visible.push({name:'其余 '+(filtered.length-20)+' 个算子',task_kind:'other',duration:filtered.slice(20).reduce((s,r)=>s+r.duration,0),count:filtered.slice(20).reduce((s,r)=>s+r.count,0)});
 el('hotspot-rows').replaceChildren();
 for(const row of visible){
  const tr=document.createElement('tr'),name=document.createElement('td'),swatch=document.createElement('span');
  swatch.className='hotspot-swatch';swatch.style.background=['gpu_transport','gpu_comm'].includes(row.task_kind)?'#e79b3e':String(row.task_kind).includes('copy')||row.task_kind==='gpu_h2d'?'#8c9eab':'#427eaa';name.append(swatch,document.createTextNode(row.name));name.title=row.name;tr.append(name);
  for(const text of [(row.duration/1e6).toFixed(6),denominator>0?(row.duration/denominator*100).toFixed(2)+'%':'未知',String(row.count)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}
  el('hotspot-rows').append(tr);
 }
 if(!visible.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent='当前阶段／口径没有已归属记录；不能据此视为零成本。';tr.append(td);el('hotspot-rows').append(tr);}
 const definition=kind==='device_tasks'?'实际计算核、去重通信搬运和显式拷贝任务':kind==='communication_envelopes'?'通信包络（含等待）':kind==='communication_kernels'?'实际通信内核（可含核内等待）':'扣除同线程嵌套 CPU 子调用后的独占调用墙钟';
 el('hotspot-denominator').textContent='分母：本阶段 '+definition+' 累计 '+(denominator/1e6).toFixed(6)+' 毫秒。'+(total>0?' 全步该口径阶段归属覆盖 '+((total-unknown)/total*100).toFixed(2)+'%；未知 '+(unknown/1e6).toFixed(6)+' 毫秒。':' 缺少该口径记录。');
 el('hotspot-note').textContent=(kind==='device_tasks'?'各流任务可能重叠；100% 是累计任务耗时构成，不能当作单步墙钟占比。通信任务记录数不等于 collective 次数。':kind==='communication_envelopes'?'此表含等待，使用独立分母；不能与实际任务时间相加。':kind==='communication_kernels'?'通信内核已包含于实际设备任务；此视图仅使用通信内核子集作为分母，两表不能相加。':'CPU 调用包含等待；多个线程的独占调用时间之和不等于 CPU 忙碌时间。')+(phase==='backward_with_finalize'?' 此视图合并反向与梯度规约收尾；原统计表保留独立阶段。':'')+(search?' 名称筛选保留完整阶段分母，筛选结果占比不必合计100%。':'');
}
el('hotspot-kind').onchange=refreshHotspotDevices;el('hotspot-phase').onchange=renderHotspots;el('hotspot-device').onchange=renderHotspots;el('hotspot-search').oninput=renderHotspots;refreshHotspotDevices();
function connections(event){
 if(!event)return [];
 return D.flows.filter(f=>f.source_event_id===event.id||f.target_event_id===event.id||f.source_op_event_id===event.id||f.runtime_event_id===event.id);
}
function relatedIds(event){
 const ids=new Set(event?[event.id]:[]);
 for(const f of connections(event))for(const k of ['source_event_id','target_event_id','source_op_event_id','runtime_event_id'])if(f[k])ids.add(f[k]);
 return ids;
}
function span(a,b){
 if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a)return;
 const width=Math.min(Math.max(b-a,0.001),Math.max(D.range[1]-D.range[0],0.001));
 left=Math.max(D.range[0],Math.min(a,D.range[1]-width));right=left+width;draw();
}
function zoom(factor,anchor){const p=anchor===undefined?(left+right)/2:anchor;span(p-(p-left)*factor,p+(right-p)*factor);}
function draw(){
 const width=Math.max(960,viewport.clientWidth),plotWidth=width-labelWidth-18,scale=plotWidth/(right-left);
 const dpr=window.devicePixelRatio||1;let y=top;layout=[];
 for(let i=0;i<D.tracks.length;i++){
  if(onlyTracks&&!onlyTracks.has(i))continue;
  const events=(grouped.get(i)||[]).filter(e=>e.end>=left&&e.start<=right);
  const depth=Math.max(0,...events.map(e=>Math.min(e.depth,11)));
  const height=30+(depth+1)*(barHeight+gap);
  layout.push({index:i,y,height,events});y+=height;
 }
 canvas.width=width*dpr;canvas.height=Math.max(y,420)*dpr;canvas.style.width=width+'px';canvas.style.height=Math.max(y,420)+'px';
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,y);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,Math.max(y,420));
 ctx.font='12px "Microsoft YaHei",Arial';ctx.fillStyle='#34506a';ctx.fillText('CPU 线程／'+D.technology.label+' 执行流',12,22);
 const tickCount=Math.max(4,Math.floor(plotWidth/110));
 for(let i=0;i<=tickCount;i++){const x=labelWidth+i*plotWidth/tickCount;ctx.strokeStyle='#e8edf3';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle='#526b83';ctx.fillText((left+i*(right-left)/tickCount).toFixed(2),x-10,22);}
 hit=[];const related=relatedIds(selected),centers=new Map();
 for(const row of layout){
  const track=D.tracks[row.index];ctx.fillStyle=track.group==='CPU'?'#f7f5fc':'#f0f6fb';ctx.fillRect(0,row.y,width,row.height);
  ctx.fillStyle=track.group==='CPU'?'#654c94':'#28698f';ctx.fillRect(0,row.y,5,row.height-1);
  ctx.fillStyle='#263e54';ctx.font='12px "Microsoft YaHei",Arial';ctx.fillText(track.label.slice(0,47),12,row.y+19);
  ctx.save();ctx.beginPath();ctx.rect(labelWidth,row.y,plotWidth,row.height);ctx.clip();
  for(const e of row.events){
   const x=labelWidth+(e.start-left)*scale,w=Math.max(1,(e.end-e.start)*scale),ey=row.y+26+Math.min(e.depth,11)*(barHeight+gap);
   ctx.fillStyle=track.color;ctx.globalAlpha=selected&&!related.has(e.id)?.38:.9;ctx.fillRect(x,ey,w,barHeight);ctx.globalAlpha=1;
   if(selected&&related.has(e.id)){ctx.strokeStyle=e.id===selected.id?'#182a3a':'#e9b644';ctx.lineWidth=2;ctx.strokeRect(x,ey,w,barHeight);}
   if(w>46){ctx.fillStyle='#fff';ctx.font='11px Arial';ctx.fillText(e.name.slice(0,Math.floor(w/6)),x+3,ey+12);}
   hit.push({e,x:Math.max(labelWidth,x),y:ey,w:Math.max(1,Math.min(width-18,x+w)-Math.max(labelWidth,x)),h:barHeight});
   centers.set(e.id,{x,y:ey+barHeight/2,event:e});
  }
  ctx.restore();ctx.strokeStyle='#dce4ed';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,row.y+row.height);ctx.lineTo(width,row.y+row.height);ctx.stroke();
 }
 if(selected&&el('arrows').checked){
  for(const f of connections(selected).filter(x=>x.drawable).slice(0,80)){
   const a=centers.get(f.source_event_id),b=centers.get(f.target_event_id);if(!a||!b)continue;
   const sx=labelWidth+((f.source_ts_ms===null?a.event.start:f.source_ts_ms)-left)*scale;
   const tx=labelWidth+((f.target_ts_ms===null?b.event.start:f.target_ts_ms)-left)*scale;
   if(sx<labelWidth||sx>width-18||tx<labelWidth||tx>width-18)continue;
   ctx.strokeStyle='#c14936';ctx.fillStyle='#c14936';ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(sx,a.y);ctx.bezierCurveTo(sx+18,a.y,tx-18,b.y,tx,b.y);ctx.stroke();
   ctx.beginPath();ctx.moveTo(tx,b.y);ctx.lineTo(tx-7,b.y-4);ctx.lineTo(tx-7,b.y+4);ctx.closePath();ctx.fill();
  }
 }
 el('start').value=left.toFixed(4);el('end').value=right.toFixed(4);
}
function find(e){const box=canvas.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;return hit.filter(r=>x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h).sort((a,b)=>a.e.duration-b.e.duration)[0];}
function select(event){
 selected=event;const flows=connections(event);
 el('selection').textContent=event.name+' · '+event.duration.toFixed(6)+' 毫秒 · '+D.tracks[event.track].label;
 el('detail').textContent=JSON.stringify({事件:event,精确关联:flows,说明:flows.length?'红线来自原始flow/correlation；提交→开始间隔不是纯排队或CPU计算耗时。':'本事件缺少可用精确关联，未按相邻时间猜测来源。'},null,2);
 draw();
}
canvas.addEventListener('mousemove',e=>{
 if(drag){const dx=e.clientX-drag.x;if(Math.abs(dx)>3)drag.moved=true;if(drag.moved){const delta=-dx*(drag.right-drag.left)/(canvas.clientWidth-labelWidth-18);span(drag.left+delta,drag.right+delta);}el('tip').style.display='none';return;}
 const r=find(e),tip=el('tip');if(!r){tip.style.display='none';return;}
 tip.textContent=r.e.name+'\n'+D.tracks[r.e.track].label+'\n开始 '+r.e.start.toFixed(6)+' 毫秒 · 持续 '+r.e.duration.toFixed(6)+' 毫秒\n精确关联 '+connections(r.e).length+' 条';
 tip.style.display='block';tip.style.left=Math.max(4,Math.min(e.clientX+14,window.innerWidth-590))+'px';tip.style.top=Math.max(4,Math.min(e.clientY+16,window.innerHeight-130))+'px';
});
canvas.addEventListener('mousedown',e=>{if(e.button===0)drag={x:e.clientX,left,right,moved:false};});
window.addEventListener('mouseup',e=>{if(drag&&!drag.moved){const r=find(e);if(r)select(r.e);}drag=null;});
canvas.addEventListener('mouseleave',()=>{el('tip').style.display='none';});
canvas.addEventListener('wheel',e=>{
 if(!e.ctrlKey&&!e.shiftKey)return;e.preventDefault();
 if(e.shiftKey){const shift=e.deltaY*(right-left)/900;span(left+shift,right+shift);}
 else{const r=canvas.getBoundingClientRect(),t=Math.max(0,Math.min(1,(e.clientX-r.left-labelWidth)/(canvas.clientWidth-labelWidth-18)));zoom(Math.exp(e.deltaY*.002),left+t*(right-left));}
},{passive:false});
el('step').onclick=()=>span(...D.step);el('capture').onclick=()=>span(...D.range);el('zin').onclick=()=>zoom(.5);el('zout').onclick=()=>zoom(2);
el('apply').onclick=()=>span(Number(el('start').value),Number(el('end').value));el('arrows').onchange=draw;
el('related').onclick=()=>{if(!selected)return;const ids=relatedIds(selected);onlyTracks=new Set([...ids].map(id=>byId.get(id)).filter(Boolean).map(e=>e.track));draw();};
el('all').onclick=()=>{onlyTracks=null;draw();};window.addEventListener('resize',draw);draw();
window.TERRA_TIMELINE={data:D,selectEvent:id=>{if(byId.has(id))select(byId.get(id));},setRange:span};
})();
</script></body></html>
'''
