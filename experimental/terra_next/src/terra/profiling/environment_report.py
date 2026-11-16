"""Chinese environment evidence pages, shared by single-run and A/B reports.

This module never samples the current machine while interpreting an old trace.
Only a linked, recorded harness document can establish historical conditions.
"""
import json
import unicodedata
from pathlib import Path


def _missing(path=None, reason=None):
    return {'status': 'UNAVAILABLE', 'path': path,
            'reason': reason or '历史记录未采集，无法判断空载或其他程序干扰',
            'summary': {}, 'participants': [], 'inventory': [], 'storage': []}


def load_environment_record(metadata, trace_path=None):
    """Load finalized evidence, retaining a small self-contained summary.

    A basename fallback permits moving the environment document alongside a
    downloaded trace. Missing or malformed evidence remains explicitly unknown.
    """
    source = metadata.get('environment_file')
    if not source:
        return _missing()
    candidates = [Path(source)]
    if trace_path is not None:
        candidates.append(Path(trace_path).parent / Path(str(source).replace('\\', '/')).name)
    document = None
    error = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size > 32 * 1024 * 1024:
                raise ValueError('Environment document exceeds 32 MiB')
            document = json.loads(candidate.read_text(encoding='utf-8'))
            if not isinstance(document, dict) or document.get('format') != 'terra_profile_environment_v1':
                raise ValueError('Unsupported environment format')
            break
        except (ValueError, OSError, TypeError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            document = None
    if document is None:
        # A completed embedded record is useful when traces are exported alone.
        embedded = metadata.get('environment_record')
        if isinstance(embedded, dict) and embedded.get('status') == 'AVAILABLE':
            return dict(embedded, path=str(source), document_available=False)
        return _missing(str(source), '环境记录无法读取；无法判断测试时是否空载' +
                        (('（' + error + '）') if error else ''))
    summary = document.get('summary')
    if str(document.get('status', '')).lower() == 'unavailable' or (
            isinstance(summary, dict) and str(summary.get('status', '')).lower() == 'unavailable'):
        reason = document.get('reason') or (summary or {}).get('reason') or document.get('monitor_error') or '采集失败'
        return _missing(str(source), '环境采集不可用：' + str(reason) + '；无法判断测试时是否空载或存在干扰')
    if not isinstance(summary, dict):
        from .environment import summarize_environment
        summary = summarize_environment(document)
    return {'status': 'AVAILABLE', 'path': str(source), 'document_available': True,
            'hostname': document.get('hostname'), 'started_utc': document.get('started_utc'),
            'server_name': document.get('server_name'), 'server_alias': document.get('server_alias'),
            'finished_utc': document.get('finished_utc'),
            'interval_s': document.get('interval_s', document.get('sample_interval_s')),
            'participants': document.get('participants', []),
            'inventory': document.get('inventory', []),
            'selected_gpu_uuids': document.get('selected_gpu_uuids', []),
            'storage': document.get('storage', []), 'summary': summary}


def unique_environments(rows):
    """One record per recorded node/run, not one duplicate per trace cycle."""
    records = []
    seen = set()
    for row in rows:
        record = row.get('environment') or _missing()
        identity = (record.get('path'), row.get('hostname'), row.get('run_id'))
        if identity not in seen:
            seen.add(identity)
            records.append(record)
    return records


def _number(value, unit='', digits=1):
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return '未采集'
    return f'{value:.{digits}f}' + unit


def _gib(value):
    return _number(value / 1024**3, ' GiB') if isinstance(value, (float, int)) else '未采集'


def _rate(value):
    return _number(value / 1024**2, ' MiB/s') if isinstance(value, (float, int)) else '未采集'


def _processes(rows, *, io=False):
    if not rows:
        return '未观察到活动（受权限、采样间隔和阈值限制）'
    parts = []
    for process in rows[:3]:
        suffix = ('读 ' + _rate(process.get('read_bytes_per_s')) + ' / 写 ' +
                  _rate(process.get('write_bytes_per_s'))) if io else (
                      'CPU ' + _number(process.get('cpu_pct'), '%'))
        parts.append(f"PID {process.get('pid', '?')} {process.get('name', '?')}：{suffix}")
    if len(rows) > 3:
        parts.append(f'另有 {len(rows)-3} 项，见环境 JSON')
    return '；'.join(parts)


def _server_name(record):
    explicit=record.get('server_name') or record.get('server_alias')
    if explicit:return str(explicit)
    hostname=record.get('hostname')
    try:
        aliases=json.loads((Path(__file__).resolve().parents[2]/'configs/profiling_server_aliases.json').read_text(encoding='utf-8'))
        return aliases.get(hostname) or hostname or '未知'
    except (OSError,ValueError,TypeError):return hostname or '未知'


def _gpu_index(value):
    try:
        number=float(value)
        if number.is_integer():return str(int(number))
    except (TypeError,ValueError,OverflowError):pass
    return '未知' if value is None else str(value)


def environment_lines(record):
    """Only equipment, GPU exclusivity and possible CPU interference go in PDF.

    Detailed measurements remain in the recorded JSON. Empty foreign-PID lists
    do not prove exclusivity when process membership or coverage is unknown.
    """
    if record.get('status')!='AVAILABLE':
        return ['测试设备：未知，缺少对应环境记录。',
                'GPU 独占：无法确认。','CPU 干扰：无法判断。']
    summary=record.get('summary',{})
    baseline=summary.get('baseline',{});during=summary.get('during',{})
    phases=(baseline,during)
    selected=record.get('selected_gpu_uuids',[])
    inventory={g.get('uuid'):g for g in record.get('inventory',[])}
    gpu_rows={g.get('uuid'):g for phase in phases for g in phase.get('selected_gpus',[])}
    identities=selected or list(gpu_rows)
    devices=[{**inventory.get(uuid,{}),**gpu_rows.get(uuid,{})} for uuid in identities]
    indices=list(dict.fromkeys(_gpu_index(g.get('index')) for g in devices))
    indices.sort(key=lambda value:(not value.isdecimal(),int(value) if value.isdecimal() else value))
    names=list(dict.fromkeys(g.get('name') for g in devices if g.get('name')))
    model=' / '.join(names) or 'GPU（型号未知）'
    if devices:
        equipment=f"测试设备：{_server_name(record)} 服务器的 {'、'.join(indices)} 号 GPU，共 {len(devices)} 张 {model}。"
    else:equipment=f"测试设备：{_server_name(record)} 服务器；GPU 型号和编号未知。"
    measured=[g for phase in phases for g in phase.get('selected_gpus',[])]
    foreign=any(g.get('foreign_pids') for g in measured)
    unknown=any(g.get('unknown_process_pids') for g in measured)
    coverage=summary.get('coverage',{})
    covered=bool(identities) and all(set(identities)<={g.get('uuid') for g in phase.get('selected_gpus',[])} for phase in phases)
    complete=covered and (coverage.get('gpu_observation_complete') is True or
        (coverage.get('gpu_observation_complete') is None and all(g.get('coverage_complete') is True for g in measured)))
    complete=complete and coverage.get('gpu_process_list_available') is not False and coverage.get('gpu_mapping_available') is not False
    before=baseline.get('selected_gpus',[])
    initially_idle=bool(before) and all(g.get('coverage_complete') is True and
        g.get('utilization_max_pct')==0 and g.get('memory_used_max_mib')==0 and
        not g.get('foreign_pids') and not g.get('unknown_process_pids') for g in before)
    if foreign:exclusive='GPU 独占：否，观察到所用 GPU 上有其他作业。'
    elif complete and not unknown:exclusive='GPU 独占：采样期间未观察到与其他作业共用。'
    else:
        exclusive='GPU 独占：无法确认全程独占'
        exclusive+='；测试前所用 GPU 空闲。' if initially_idle else '。'
    evidence=summary.get('evidence',{})
    threshold=summary.get('thresholds',{}).get('external_process_cpu_pct',50)
    observed_cpu=bool(evidence.get('external_cpu_activity')) or any(
        (p.get('cpu_pct') or 0)>=threshold for phase in phases for p in phase.get('external_cpu_top',[]))
    other_jobs=bool(evidence.get('other_gpus_have_jobs')) or any(phase.get('other_gpu_jobs') for phase in phases)
    if observed_cpu:cpu='CPU 干扰：有可能，同机存在其他 CPU 计算任务。'
    elif other_jobs:cpu='CPU 干扰：有可能，同机还有其他计算作业。'
    elif coverage.get('cpu_baseline_and_during_available') is True and coverage.get('process_io_complete') is True and not coverage.get('sample_errors'):
        cpu='CPU 干扰：采样期间未发现明显的其他任务竞争。'
    else:cpu='CPU 干扰：无法判断，现有记录不足。'
    return [equipment,exclusive,cpu]


def chinese_font():
    """Find a real CJK font; never silently create unreadable tofu glyphs."""
    from matplotlib import font_manager
    candidates = ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
                  'Source Han Sans SC', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Arial Unicode MS')
    for name in candidates:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            # Droid's CJK fallback font intentionally lacks many Latin glyphs.
            # A filename pins all characters to that single file; a family list
            # lets Matplotlib use DejaVu for Latin and the CJK face for Chinese.
            family = font_manager.FontProperties(fname=path).get_name()
            return font_manager.FontProperties(family=['DejaVu Sans', family])
        except ValueError:
            pass
    for path in ('C:/Windows/Fonts/msyh.ttc',
                 '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            return font_manager.FontProperties(family=['DejaVu Sans', family])
    raise RuntimeError('中文环境说明页缺少中文字体；请安装 Noto Sans CJK 或 Droid Sans Fallback 后重新导出 PDF。JSON/CSV 可继续使用。')


def _wrap(text, width=174):
    lines = []
    current = ''
    used = 0
    for char in str(text):
        char_width = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used + char_width > width or char == '\n':
            lines.append(current)
            current = ''
            used = 0
            if char == '\n':
                continue
        current += char
        used += char_width
    if current:
        lines.append(current)
    return lines or ['']


def render_environment_intro(pages, entries, *, preview_prefix=None):
    """One short card per distinct environment; identical A/B cards are merged."""
    import matplotlib.pyplot as plt
    font=chinese_font()
    cards=[];lookup={}
    for entry in entries or [{'label':'本次测试','record':_missing()}]:
        lines=tuple(environment_lines(entry.get('record') or _missing()))
        label=str(entry.get('label','本次测试')).replace(' 测试环境','')
        if lines in lookup:
            if label not in lookup[lines]['labels']:lookup[lines]['labels'].append(label)
        else:
            card={'labels':[label],'lines':lines};lookup[lines]=card;cards.append(card)
    previews=[];page_number=0
    for offset in range(0,len(cards),4):
        group=cards[offset:offset+4]
        wrapped=[[part for line in card['lines'] for part in _wrap(line,145)] for card in group]
        height=max(180,72+sum(28+22*len(lines)+16 for lines in wrapped))
        fig=plt.figure(figsize=(15,height/72));fig.patch.set_facecolor('white')
        fig.text(.055,1-34/height,'TERRA 测试环境',fontproperties=font,fontsize=20,color='#15384b')
        y=height-72
        for card,lines in zip(group,wrapped):
            fig.text(.055,y/height,' / '.join(card['labels']),fontproperties=font,fontsize=12,color='#465865')
            y-=28
            for line in lines:
                fig.text(.055,y/height,line,fontproperties=font,fontsize=13,color='#252f34')
                y-=22
            y-=16
        pages.savefig(fig);page_number+=1
        if preview_prefix is not None:
            prefix=Path(preview_prefix);target=prefix.with_name(prefix.name+f'.page{page_number}.png')
            fig.savefig(target,dpi=140);previews.append(str(target))
        plt.close(fig)
    return previews,page_number
