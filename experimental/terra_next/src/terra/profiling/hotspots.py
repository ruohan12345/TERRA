"""Source-attributed operator hotspots; no torch, plotting, or clock guesses.

Device execution time never decides forward/backward ownership. Elapsed sums
are activity-time denominators and may overlap; they are not step wall time.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import re

from .intervals import interval_duration, intersect_intervals, union_intervals


_PHASES = {'terra/train/forward': 'forward', 'terra/train/backward': 'backward',
           'terra/train/gradient_sync': 'gradient_finalize', 'terra/train/loss': 'loss',
           'terra/train/optimizer': 'optimizer', 'terra/train/gradient_clip': 'gradient_clip',
           'terra/train/unscale': 'unscale', 'terra/train/finite_check': 'finite_check'}
_INSTANCE = re.compile(r'__\d+(?:_\d+)*$')


def normalize_name(name):
    """Remove numeric HCOM instance suffixes only, never invent kernel families."""
    return _INSTANCE.sub('', str(name))


class _Scopes:
    def __init__(self, events):
        grouped = defaultdict(list)
        for event in events:
            grouped[event.get('pid'), event.get('tid')].append(event)
        self.index = {}
        for key, values in grouped.items():
            values.sort(key=lambda event: (event['start_ns'], -event['end_ns']))
            maximum, high = [], -float('inf')
            for event in values:
                high = max(high, event['end_ns']); maximum.append(high)
            self.index[key] = values, [event['start_ns'] for event in values], maximum

    def containing(self, pid, tid, start, end=None):
        if start is None:
            return None
        values = self.index.get((pid, tid))
        if values is None:
            return None
        rows, starts, maximum = values
        i, candidates = bisect_right(starts, start) - 1, []
        while i >= 0 and maximum[i] > start:
            row = rows[i]
            if row['start_ns'] <= start < row['end_ns'] and (end is None or end <= row['end_ns']):
                candidates.append(row)
            i -= 1
        return min(candidates, key=lambda e: e['end_ns'] - e['start_ns']) if candidates else None


def _subtract(intervals, children):
    output = []
    for start, end in union_intervals(intervals):
        cursor = start
        for a, b in union_intervals(children):
            if b <= cursor or a >= end:
                continue
            if a > cursor:
                output.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            output.append((cursor, end))
    return output


def _window(events, analysis):
    selected = (analysis or {}).get('selected_update')
    if selected:
        return (selected['start_ns'], selected['end_ns']), 'selected_training_update'
    updates = [e for e in events if e['name'] == 'terra/profile/update']
    if updates:
        last = max(updates, key=lambda e: e['start_ns'])
        return (last['start_ns'], last['end_ns']), 'last_recorded_update'
    selected = [e for e in events if e['kind'] not in ('cpu_op', 'cpu_runtime')] or events
    return (min(e['start_ns'] for e in selected), max(e['end_ns'] for e in selected)), 'capture_window_without_update'


def analyze_hotspots(bundle, analysis=None):
    """Return full rows/totals for actual tasks, HCOM envelopes and CPU self.

    NPU HCOM envelopes are a separate table from their transport children.
    A CPU self call excludes nested CPU operators on the same pid/tid, not
    runtime calls on another virtual lane. It still includes host waiting.
    """
    original_events = bundle.events
    real_operator_ids = {e['event_id'] for e in original_events
                         if e['kind'] == 'cpu_op' and e.get('event_id')}
    # Older public callers construct TraceBundle by hand. Local ids support
    # self-time bookkeeping only; they must never become dispatch identities.
    events = [dict(event, event_id='analysis-cpu:' + str(index))
              if event['kind'] == 'cpu_op' and not event.get('event_id') else event
              for index, event in enumerate(original_events)]
    if not events:
        return {'format': 'terra_operator_hotspots_v1', 'rows': [], 'totals': [],
                'coverage': {}, 'available': False, 'limitations': ['No events']}
    window, selection = _window(events, analysis)
    by_id = {e.get('event_id'): e for e in events if e.get('event_id')}
    operator_events = [e for e in events if e['kind'] == 'cpu_op']
    markers = [e for e in events if e['kind'] == 'cpu_phase' and e['name'] in _PHASES]
    phase_scopes = _Scopes(markers)
    backward_scopes = _Scopes(e for e in operator_events
                             if e['name'].startswith('autograd::engine::evaluate_function:'))
    update_scopes = _Scopes(e for e in events if e['name'] == 'terra/profile/update')

    def cpu_phase(event):
        marker = phase_scopes.containing(event.get('pid'), event.get('tid'), event['start_ns'], event['end_ns'])
        if marker:
            return _PHASES[marker['name']], 'same_thread_terra_phase', marker.get('event_id')
        backward = backward_scopes.containing(event.get('pid'), event.get('tid'), event['start_ns'], event['end_ns'])
        if backward:
            return 'backward', 'same_thread_autograd_evaluate_function', backward.get('event_id')
        update = update_scopes.containing(event.get('pid'), event.get('tid'), event['start_ns'], event['end_ns'])
        if update:
            return 'other_update', 'same_thread_update_scope', update.get('event_id')
        return 'unknown', 'no_same_thread_phase_scope', None

    operator_phases = {e.get('event_id'): cpu_phase(e) for e in operator_events if e.get('event_id')}
    exact_flows = defaultdict(list)
    for flow in getattr(bundle, 'dispatch_flows', []):
        if flow.get('exact') is True:
            exact_flows[flow.get('target_event_id')].append(flow)

    def phase_at_source(pid, tid, timestamp):
        marker = phase_scopes.containing(pid, tid, timestamp)
        if marker:
            return _PHASES[marker['name']], 'exact_cpu_source_in_terra_phase', marker.get('event_id')
        marker = backward_scopes.containing(pid, tid, timestamp)
        if marker:
            return 'backward', 'exact_cpu_source_in_autograd_scope', marker.get('event_id')
        marker = update_scopes.containing(pid, tid, timestamp)
        if marker:
            return 'other_update', 'exact_cpu_source_in_update', marker.get('event_id')
        return 'unknown', 'source_scope_unavailable', None

    def device_phase(event):
        metadata = event.get('metadata', {})
        source = metadata.get('source_metadata', {})
        tag = metadata.get('source_range', '')
        # These are recorded enqueue semantics, not device execution positions.
        if metadata.get('source_flow_exact') is True:
            if tag in ('terra/comm/update/bucket_mp_sum', 'terra/comm/update/bucket_dp_average'):
                during = source.get('during_backward')
                if isinstance(during, bool):
                    return ('backward' if during else 'gradient_finalize'), 'recorded_gradient_bucket_mode', None
            direction = source.get('direction')
            if tag.startswith('terra/comm/') and direction in ('forward', 'backward'):
                return direction, 'recorded_communication_direction', None
        candidates = []
        for flow in exact_flows.get(event.get('event_id'), []):
            for key in ('source_op_event_id', 'source_event_id'):
                identifier = flow.get(key)
                if identifier in real_operator_ids and identifier in operator_phases and operator_phases[identifier][0] != 'unknown':
                    phase, evidence, _ = operator_phases[identifier]
                    candidates.append((phase, 'exact_dispatch_operator/' + evidence, identifier))
                    break
            else:
                value = phase_at_source(flow.get('source_pid'), flow.get('source_tid'), flow.get('source_ts_ns'))
                if value[0] != 'unknown':
                    candidates.append(value)
        if metadata.get('source_flow_exact') is True:
            identifier = metadata.get('source_op_event_id')
            if identifier in real_operator_ids and identifier in operator_phases and operator_phases[identifier][0] != 'unknown':
                phase, evidence, _ = operator_phases[identifier]
                candidates.append((phase, 'exact_async_operator/' + evidence, identifier))
            else:
                value = phase_at_source(metadata.get('source_cpu_pid'), metadata.get('source_cpu_tid'),
                                        metadata.get('source_cpu_timestamp_ns'))
                if value[0] != 'unknown':
                    candidates.append(value)
        phases = {item[0] for item in candidates}
        if len(phases) == 1:
            return candidates[0]
        return 'unknown', 'conflicting_exact_sources' if phases else 'no_exact_phase_source', None

    parent_index = {}
    for event in events:
        if event['kind'] != 'gpu_comm' or event.get('correlation') is None:
            continue
        key = str(event.get('device')), event['correlation']
        parent_index[key] = event if key not in parent_index else None
    groups = {}
    coverage = defaultdict(lambda: Counter())
    event_phase_evidence = Counter()

    def add(event, table, phase_info, intervals, *, name=None, inclusive=None):
        elapsed = interval_duration(intervals)
        phase, evidence, source_id = phase_info
        device = 'cpu' if table == 'cpu_self' else str(event.get('device'))
        name = normalize_name(event['name']) if name is None else name
        communication_category = (event.get('metadata', {}).get('communication_category', 'unknown')
                                  if event['kind'] in ('gpu_comm', 'gpu_transport') else None)
        prefix = {'gradient': '梯度桶', 'halo': 'Halo', 'other': '其他通信', 'unknown': '来源未知通信'}.get(communication_category)
        display_name = prefix + ' / ' + name if prefix else name
        key = table, phase, device, event['kind'], name, communication_category
        row = groups.setdefault(key, {'table_kind': table, 'phase': phase, 'device': device,
             'task_kind': event['kind'], 'name': name, 'display_name': display_name,
             'communication_category': communication_category, 'count': 0,
             'duration_sum_ns': 0, 'inclusive_duration_sum_ns': 0, 'evidence_counts': Counter(),
             'source_range_counts': Counter(), 'raw_name_counts': Counter(),
             'example_event_ids': [], '_intervals': defaultdict(list)})
        row['count'] += 1; row['duration_sum_ns'] += elapsed
        row['inclusive_duration_sum_ns'] += elapsed if inclusive is None else inclusive
        row['evidence_counts'][evidence] += 1
        row['source_range_counts'][event.get('metadata', {}).get('source_range', 'unmatched')] += 1
        row['raw_name_counts'][event['name']] += 1
        if len(row['example_event_ids']) < 5:
            row['example_event_ids'].append(event.get('event_id'))
        thread = (event.get('pid'), event.get('tid')) if table == 'cpu_self' else device
        row['_intervals'][thread].extend(intervals)
        coverage[table]['count'] += 1; coverage[table]['duration_sum_ns'] += elapsed
        coverage[table]['known_phase_count'] += int(phase != 'unknown')
        coverage[table]['known_phase_duration_sum_ns'] += elapsed if phase != 'unknown' else 0
        event_phase_evidence[table + '/' + evidence] += 1

    for event in events:
        if not event['kind'].startswith('gpu_'):
            continue
        clipped = intersect_intervals([(event['start_ns'], event['end_ns'])], [window])
        if not clipped:
            continue
        native = event.get('metadata', {}).get('device_adapter') == 'ascend_native_v1'
        table = 'communication_envelopes' if native and event['kind'] == 'gpu_comm' else 'device_tasks'
        name = normalize_name(event['name'])
        if event['kind'] == 'gpu_transport':
            parent = parent_index.get((str(event.get('device')), event.get('correlation')))
            child = event.get('metadata', {}).get('child_task_name') or event['name']
            if parent:
                category = parent.get('metadata', {}).get('communication_category', 'unknown')
                name = normalize_name(parent['name']) + ' [' + category + '] / ' + child
            else:
                name = normalize_name(child) + ' [parent unknown]'
        add(event, table, device_phase(event), clipped, name=name)

    # Direct nested operators suffice: their intervals already cover descendants.
    # Crossing (not properly nested) records are reported, never silently fixed.
    by_thread = defaultdict(list)
    for event in operator_events:
        by_thread[event.get('pid'), event.get('tid')].append(event)
    children = defaultdict(list); crossings = 0
    for values in by_thread.values():
        stack = []
        for event in sorted(values, key=lambda e: (e['start_ns'], -e['end_ns'])):
            while stack and stack[-1]['end_ns'] <= event['start_ns']:
                stack.pop()
            while stack and event['end_ns'] > stack[-1]['end_ns']:
                crossings += 1; stack.pop()
            if stack:
                children[stack[-1]['event_id']].append((event['start_ns'], event['end_ns']))
            stack.append(event)
    for event in operator_events:
        clipped = intersect_intervals([(event['start_ns'], event['end_ns'])], [window])
        if clipped:
            self_intervals = _subtract(clipped, children[event['event_id']])
            add(event, 'cpu_self', operator_phases[event['event_id']], self_intervals,
                inclusive=interval_duration(clipped))

    totals = defaultdict(lambda: {'denominator_ns': 0, 'count': 0})
    for row in groups.values():
        key = row['table_kind'], row['phase'], row['device']
        totals[key]['denominator_ns'] += row['duration_sum_ns']; totals[key]['count'] += row['count']
    rows = []
    for row in groups.values():
        row['duration_union_ns'] = sum(interval_duration(values) for values in row.pop('_intervals').values())
        denominator = totals[row['table_kind'], row['phase'], row['device']]['denominator_ns']
        row.update(denominator_ns=denominator, share_percent=100 * row['duration_sum_ns'] / denominator if denominator else None)
        row['evidence_counts'] = dict(row['evidence_counts'])
        row['source_range_counts'] = dict(row['source_range_counts'])
        row['raw_name_counts'] = dict(row['raw_name_counts'])
        rows.append(row)
    rows.sort(key=lambda row: (row['table_kind'], row['phase'], row['device'], -row['duration_sum_ns'], row['name']))
    totals = [{'table_kind': table, 'phase': phase, 'device': device, **values}
              for (table, phase, device), values in sorted(totals.items())]
    coverage = {table: {**dict(value),
                'unknown_phase_count': value['count'] - value['known_phase_count'],
                'unknown_phase_duration_sum_ns': value['duration_sum_ns'] - value['known_phase_duration_sum_ns'],
                'known_phase_count_fraction': value['known_phase_count'] / value['count'] if value['count'] else None,
                'known_phase_duration_fraction': value['known_phase_duration_sum_ns'] / value['duration_sum_ns'] if value['duration_sum_ns'] else None}
                for table, value in coverage.items()}
    return {'format': 'terra_operator_hotspots_v1', 'available': True,
            'window': {'start_ns': window[0], 'end_ns': window[1], 'wall_ns': window[1] - window[0], 'selection': selection},
            'rows': rows, 'totals': totals, 'coverage': coverage, 'evidence_counts': dict(event_phase_evidence),
            'cpu_self_policy': {'available': crossings == 0,
                'invalid_reason': None if crossings == 0 else 'crossing CPU operator intervals; nested self-time is not reliable',
                'subtract': 'fully nested cpu_op intervals on the same pid/tid only',
                'runtime_subtracted': False, 'crossing_operator_intervals': crossings,
                'union_scope': 'sum of per-thread unions; includes waits, not CPU busy time'},
            'limitations': ['Phase uses CPU-side scopes, autograd nodes, explicit communication direction or exact dispatch sources; never GPU wall placement.',
                'device_tasks excludes native HCOM wait envelopes; those have a separate communication_envelopes denominator.',
                'A table denominator is sum of event elapsed times (CPU table: self call wall). Activities can overlap; percentages are not step wall fractions.',
                'Kernel/task names are retained; only numeric __instance suffixes are removed. No GEMM/Conv families are invented.',
                'Unknown phase and update/finalize work remain explicit. CPU nested self times are not added to device time.']}
