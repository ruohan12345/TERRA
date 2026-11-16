"""Ascend native Chrome events -> the shared profiling event vocabulary.

No torch/torch_npu dependency. Hardware execution, collective envelopes and
transport tasks are distinct layers; wait/control records are not computation.
CPU enqueue sources are joined by exact async_npu flows, never nearest time.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation


NAME = "ascend_native_v1"
COMPUTE = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU"}
WAIT_CONTROL = {"Notify Wait", "EVENT_WAIT", "NOTIFY_WAIT", "NOTIFY_WAIT_SQE",
                "Write Value", "EVENT_RECORD", "NOTIFY_RECORD", "NOTIFY_RECORD_SQE",
                "WRITE_VALUE_SQE", "PROFILING_DISABLE"}
NATIVE_PROCESSES = {"Ascend Hardware", "Communication", "CANN", "Overlap Analysis", "AI Core Freq"}


def is_profiler_envelope(name):
    return name == "ProfilerStep" or str(name).startswith("ProfilerStep#")


def time_ns(value):
    try:
        value = Decimal(str(value)) * 1000
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid native timestamp") from exc
    if not value.is_finite():
        raise ValueError("Nonfinite native timestamp")
    return int(value)


def is_native_npu(rows):
    return any(isinstance(row, dict) and (
        row.get("ph") == "M" and row.get("name") == "process_name"
        and row.get("args", {}).get("name") == "Ascend Hardware") for row in rows)


def _plain(value):
    """Event metadata must remain JSON serializable after Decimal JSON parsing."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _interval(row):
    start, duration = time_ns(row["ts"]), time_ns(row["dur"])
    if duration < 0:
        raise ValueError("Negative native duration")
    return start, start + duration


def _task_key(row, hardware):
    args = row.get("args", {})
    fields = ("Physic Stream Id", "Task Id", "Subtask Id") if hardware else ("stream id", "task id", "context id")
    if any(field not in args for field in fields):
        return None
    return tuple(args[field] for field in fields)


def adapt_native_npu(rows, metadata):
    """Return normalized device events and provenance for one rank-local trace.

    skip_indices includes native device/control layers; accepted_indices is the
    subset represented by normalized events, so the caller can retain counts.
    """
    processes = {row.get("pid"): row.get("args", {}).get("name") for row in rows
                 if isinstance(row, dict) and row.get("ph") == "M" and row.get("name") == "process_name"}
    device = str(metadata.get("device", ""))
    device_known = device.startswith("npu")
    if not device_known:
        device = "npu:unknown"
    hardware_pids = {pid for pid, name in processes.items() if name == "Ascend Hardware"}
    ambiguous_device = not device_known and len(hardware_pids) > 1
    if ambiguous_device:
        device = "npu:ambiguous"
    skip = {index for index, row in enumerate(rows) if isinstance(row, dict)
            and processes.get(row.get("pid")) in NATIVE_PROCESSES}
    valid, malformed = [], 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("ph") != "X":
            continue
        try:
            bounds = _interval(row)
        except (KeyError, TypeError, ValueError, InvalidOperation):
            if index in skip:
                malformed += 1
            continue
        if bounds[1] > bounds[0]:
            valid.append((index, row, bounds))
    annotations = metadata.get("annotations", {})
    if isinstance(annotations, list):
        annotations = {item["label"]: item for item in annotations}
    ranges = defaultdict(list)
    for _, row, bounds in valid:
        annotation = annotations.get(row.get("name"))
        if annotation:
            ranges[row.get("pid"), row.get("tid")].append((bounds[0], bounds[1], annotation))
    range_index = {}
    for key, values in ranges.items():
        values.sort(key=lambda item: item[:2])
        high, maximum = -1, []
        for start, end, _ in values:
            high = max(high, end)
            maximum.append(high)
        range_index[key] = values, [item[0] for item in values], maximum
    operators = defaultdict(list)
    operator_events = {}
    for index, row, bounds in valid:
        if (processes.get(row.get("pid")) not in NATIVE_PROCESSES and row.get("cat") == "cpu_op"
                and row.get("name") not in annotations and not str(row.get("name", "")).startswith("terra/")
                and not is_profiler_envelope(row.get("name"))):
            item = {"event_id": "row:" + str(index), "name": row.get("name"),
                    "pid": row.get("pid"), "tid": row.get("tid"),
                    "start_ns": bounds[0], "end_ns": bounds[1], "kind": "cpu_op"}
            operators[item["pid"], item["tid"]].append(item)
            operator_events[item["event_id"]] = item
    operator_index = {}
    for key, values in operators.items():
        values.sort(key=lambda item: item["start_ns"])
        high, maximum = -1, []
        for item in values:
            high = max(high, item["end_ns"])
            maximum.append(high)
        operator_index[key] = values, [item["start_ns"] for item in values], maximum

    def enclosing_operator(pid, tid, timestamp):
        indexed = operator_index.get((pid, tid))
        if indexed is None:
            return None
        values, starts, maximum = indexed
        i, candidates = bisect_right(starts, timestamp) - 1, []
        while i >= 0 and maximum[i] > timestamp:
            item = values[i]
            if item["start_ns"] <= timestamp < item["end_ns"]:
                candidates.append(item)
            i -= 1
        return min(candidates, key=lambda item: item["end_ns"] - item["start_ns"]) if candidates else None

    flow_starts = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and row.get("cat") == "async_npu" and row.get("ph") == "s":
            flow_starts[row.get("id")].append(row)
    flow_finishes = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("cat") != "async_npu" or row.get("ph") != "f":
            continue
        sources = flow_starts.get(row.get("id"), [])
        if len(sources) != 1:  # Ambiguous/reused ids are never joined by proximity.
            continue
        try:
            key = row.get("pid"), row.get("tid"), time_ns(row["ts"])
            time_ns(sources[0]["ts"])
        except (KeyError, TypeError, ValueError, InvalidOperation):
            continue
        if key in flow_finishes:
            flow_finishes[key] = None
        else:
            flow_finishes[key] = sources[0]

    def origin(row, start):
        source = flow_finishes.get((row.get("pid"), row.get("tid"), start))
        details = {"source_flow_exact": source is not None, "source_range": "unmatched",
                   "source_metadata": {}, "source_cpu_timestamp_ns": None,
                   "source_flow_id": None, "source_op_event_id": None, "source_op_name": None}
        if source is None:
            return details
        timestamp = time_ns(source["ts"])
        details.update(source_cpu_timestamp_ns=timestamp, source_cpu_pid=source.get("pid"),
                       source_cpu_tid=source.get("tid"), source_flow_id=source.get("id"))
        operator = enclosing_operator(source.get("pid"), source.get("tid"), timestamp)
        if operator:
            details.update(source_op_event_id=operator["event_id"], source_op_name=operator["name"],
                           source_op_start_ns=operator["start_ns"], source_op_end_ns=operator["end_ns"])
        indexed = range_index.get((source.get("pid"), source.get("tid")))
        if indexed:
            values, starts, maximum = indexed
            i, containing = bisect_right(starts, timestamp) - 1, []
            while i >= 0 and maximum[i] > timestamp:
                a, b, annotation = values[i]
                if a <= timestamp < b:
                    containing.append((b - a, i, annotation))
                i -= 1
            if containing:
                annotation = min(containing, key=lambda item: item[:2])[2]
                details["source_range"] = annotation.get("name", "unmatched")
                details["source_metadata"] = _plain(annotation.get("metadata", {}))
        return details

    def category(details):
        tag = details["source_range"]
        if tag in ("terra/comm/update/bucket_mp_sum", "terra/comm/update/bucket_dp_average"):
            return "gradient"
        if tag == "terra/comm/dmp/halo":
            return "halo"
        return "other" if tag.startswith("terra/comm/") else "unknown"

    events, accepted, dispatch_flows = [], set(), []

    def emit(index, row, bounds, kind, details):
        args = row.get("args", {})
        event = {"event_id": "row:" + str(index), "kind": kind, "name": str(row.get("name", "")),
                       "label": str(row.get("name", "")), "start_ns": bounds[0], "end_ns": bounds[1],
                       "pid": row.get("pid"), "tid": row.get("tid"), "device": device if kind.startswith("gpu_") else None,
                       "stream": args.get("Physic Stream Id", args.get("stream id", row.get("tid"))),
                       "correlation": args.get("connection_id"),
                       "metadata": {"device_adapter": NAME, **details}}
        events.append(event)
        accepted.add(index)
        return event

    runtimes = defaultdict(list)
    for index, row, bounds in valid:
        if processes.get(row.get("pid")) != "CANN":
            continue
        args, name = row.get("args", {}), str(row.get("name", ""))
        if args.get("level") not in ("acl", "node", "communication"):
            continue
        role = ("synchronize" if "synchron" in name.lower() or "wait" in name.lower() else
                "launch" if args.get("Mode") == "launch" or args.get("level") == "node" else
                "collective_submit" if args.get("level") == "communication" else "api")
        event = emit(index, row, bounds, "cpu_runtime", {"native_process": "CANN",
                     "runtime_role": role, "runtime_target_device": device,
                     "runtime_level": args.get("level"), "native_args": _plain(args)})
        connection = args.get("connection_id")
        if connection is not None:
            runtimes[connection].append(event)

    def link_dispatch(event, row):
        details = event["metadata"]
        runtime_candidates = runtimes.get(row.get("args", {}).get("connection_id"), [])
        runtime = runtime_candidates[0] if len(runtime_candidates) == 1 else None
        operator = operator_events.get(details.get("source_op_event_id"))
        source = runtime or operator
        exact_async = bool(details.get("source_flow_exact"))
        if runtime is None and not exact_async:
            details.update(dispatch_exact=False, dispatch_evidence="unknown", runtime_event_id=None)
            return
        evidence = "native_connection_id_exact" if runtime else "native_async_npu_exact"
        timestamp = runtime["start_ns"] if runtime else details.get("source_cpu_timestamp_ns")
        details.update(dispatch_exact=True, dispatch_evidence=evidence,
                       runtime_event_id=runtime["event_id"] if runtime else None)
        if runtime:
            details.update(launch_start_ns=runtime["start_ns"], launch_end_ns=runtime["end_ns"],
                           launch_cpu_pid=runtime["pid"], launch_cpu_tid=runtime["tid"],
                           launch_name=runtime["name"])
        dispatch_flows.append({"id": "native:" + event["event_id"],
            "flow_id": details.get("source_flow_id"), "evidence": evidence, "exact": True,
            "source_event_id": source["event_id"] if source else None,
            "target_event_id": event["event_id"], "source_ts_ns": timestamp,
            "target_ts_ns": event["start_ns"],
            "source_pid": source["pid"] if source else details.get("source_cpu_pid"),
            "source_tid": source["tid"] if source else details.get("source_cpu_tid"),
            "target_device": event["device"], "target_stream": event["stream"],
            "source_name": source["name"] if source else "unresolved exact CPU source point",
            "target_name": event["name"], "source_op_event_id": details.get("source_op_event_id"),
            "source_op_name": details.get("source_op_name"),
            "source_timestamp_semantics": "runtime_call_start" if runtime else "raw_async_npu_point",
            "runtime_event_id": runtime["event_id"] if runtime else None,
            "operator_evidence": "native_async_npu_exact_and_same_thread_containment" if operator else None})

    parents, coverage = {}, Counter()
    for index, row, bounds in valid:
        if processes.get(row.get("pid")) != "Communication" or not str(row.get("name", "")).lower().startswith(("hcom_", "hccl_", "nccl_")):
            continue
        details = origin(row, bounds[0])
        details.update(communication_category=category(details), includes_wait=True,
                       native_process="Communication", native_args=_plain(row.get("args", {})))
        coverage["hcom_envelopes"] += 1
        coverage["hcom_exact_flow_matches"] += int(details["source_flow_exact"])
        coverage["hcom_source_range_matches"] += int(details["source_range"] != "unmatched")
        event = emit(index, row, bounds, "gpu_comm", details)
        link_dispatch(event, row)
        connection = row.get("args", {}).get("connection_id")
        if connection is not None:
            if connection in parents:
                parents[connection] = None
            else:
                parents[connection] = details

    children = defaultdict(list)
    for index, row, bounds in valid:
        if processes.get(row.get("pid")) == "Communication" and row.get("name") in ("Memcpy", "Reduce_Inline"):
            key = _task_key(row, False)
            if key is not None:
                children[key, bounds].append((index, row, bounds))
    matched_children, hardware_types = set(), Counter()
    for index, row, bounds in valid:
        if processes.get(row.get("pid")) != "Ascend Hardware":
            continue
        args = row.get("args", {})
        task_type = args.get("Task Type")
        hardware_types[task_type or "unspecified"] += 1
        if task_type in WAIT_CONTROL:
            coverage["excluded_wait_control_records"] += 1
            continue
        if task_type in COMPUTE:
            details = origin(row, bounds[0])
            details.update(task_type=task_type, native_process="Ascend Hardware")
            event = emit(index, row, bounds, "gpu_compute", details)
            link_dispatch(event, row)
            continue
        key = _task_key(row, True)
        aliases = children.get((key, bounds), [])
        if task_type not in ("SDMA", "SDMA_SQE") and not aliases:
            emit(index, row, bounds, "gpu_unknown", {"task_type": task_type,
                 "classification": "unrecognized native hardware task"})
            coverage["unknown_hardware_records"] += 1
            continue
        parent = parents.get(args.get("connection_id"))
        if aliases or task_type == "SDMA" or parent is not None:
            child = aliases[0][1] if aliases else None
            child_args = child.get("args", {}) if child else {}
            details = dict(parent) if parent else {"source_flow_exact": False, "source_range": "unmatched",
                      "source_metadata": {}, "source_cpu_timestamp_ns": None, "communication_category": "unknown"}
            # A transport task is execution, not its parent's wait envelope.
            details.update(includes_wait=False, task_type=task_type, native_process="Ascend Hardware",
                           communication_child_matched=bool(aliases), alias_key=list(key) if key else None,
                           child_task_name=child.get("name") if child else None,
                           transport_type=child_args.get("transport type", "SDMA"),
                           link_type=child_args.get("link type", "unknown"), size_bytes=child_args.get("size(Byte)"))
            details["source_association_scope"] = "parent_collective" if parent else "unknown"
            event = emit(index, row, bounds, "gpu_transport", details)
            link_dispatch(event, row)
            coverage["transport_hardware_records"] += 1
            if aliases:
                matched_children.add((key, bounds))
                coverage["exact_transport_aliases"] += 1
                if row.get("name") == "MEMCPY_ASYNC":
                    coverage["memcpy_async_also_communication_children"] += 1
            else:
                coverage["unmatched_transport_hardware_records"] += 1
        elif row.get("name") == "MEMCPY_ASYNC":
            details = origin(row, bounds[0])
            details.update({"task_type": task_type,
                 "communication_category": "unknown", "communication_child_matched": False,
                 "direction": "unknown", "native_process": "Ascend Hardware",
                 "classification": "MEMCPY_ASYNC without matching HCOM child or parent connection"})
            event = emit(index, row, bounds, "gpu_copy_other", details)
            link_dispatch(event, row)
            coverage["non_hcom_memcpy_async_records"] += 1
        else:
            emit(index, row, bounds, "gpu_unknown", {"task_type": task_type,
                 "classification": "unrecognized SDMA_SQE operation"})
            coverage["unknown_hardware_records"] += 1
    for key, aliases in children.items():
        if key in matched_children:
            continue
        index, row, bounds = aliases[0]
        args = row.get("args", {})
        # Preserve observed transport details when the hardware alias is absent,
        # but do not invent a parent/semantic group from interval containment.
        emit(index, row, bounds, "gpu_transport", {"source_flow_exact": False,
             "source_range": "unmatched", "source_metadata": {}, "source_cpu_timestamp_ns": None,
             "communication_category": "unknown", "includes_wait": False,
             "communication_child_matched": False, "hardware_alias_missing": True,
             "native_process": "Communication", "alias_key": list(key[0]),
             "child_task_name": row.get("name"), "transport_type": args.get("transport type", "unknown"),
             "link_type": args.get("link type", "unknown"), "size_bytes": args.get("size(Byte)")})
        coverage["transport_child_without_hardware_alias"] += 1
    counts = Counter(event["kind"] for event in events)
    counter_names = Counter(row.get("name") for row in rows if isinstance(row, dict) and row.get("ph") == "C")
    device_events = sum(count for kind, count in counts.items() if kind.startswith("gpu_"))
    diagnostics = {"name": NAME, "version": 1, "detected": True,
                   "status": "native_activity_observed" if device_events else "native_device_activity_missing",
                   "device_id": device, "device_identity_known": device_known,
                   "device_identity_ambiguous": ambiguous_device,
                   "overlap_available": bool(device_events) and not ambiguous_device,
                   "coverage": {**dict(coverage), "communication_child_records": sum(map(len, children.values())),
                                "communication_child_unique_tasks": len(children)},
                   "normalized_counts": dict(counts), "hardware_task_type_counts": dict(hardware_types),
                   "compute_activity_present": bool(counts["gpu_compute"]),
                   "communication_activity_present": bool(counts["gpu_comm"]),
                   "transport_activity_present": bool(counts["gpu_transport"]),
                   "dispatch_flow_count": len(dispatch_flows),
                   "dispatch_evidence_counts": dict(Counter(flow["evidence"] for flow in dispatch_flows)),
                   "hbm_counters_available": False, "observed_counter_names": dict(counter_names),
                   "classification": {"compute": "Actual hardware AI tasks, including packing/vector/layout kernels",
                       "communication": "HCOM parent envelopes, including wait; not physical network time",
                       "transport": "SDMA/copy/inline-reduction task execution; hardware and Communication aliases counted once",
                       "copy": "Unassociated MEMCPY_ASYNC; direction unknown",
                       "hbm": "No decoded HBM byte/bandwidth counters; SDMA is not total kernel memory traffic"},
                   "limitations": ["One trace has one rank-local clock; never align different hosts by timestamp.",
                       "Missing/ambiguous async_npu flow or source ranges produce unknown semantic categories.",
                       "No nearest-time matching, physical idle, HBM utilization or network bandwidth is inferred."]}
    return {"events": events, "skip_indices": skip, "accepted_indices": accepted,
            "device_adapter": diagnostics, "dispatch_flows": dispatch_flows,
            "malformed_events": malformed}
