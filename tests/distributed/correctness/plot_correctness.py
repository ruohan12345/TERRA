import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


CASE_CONFIG_FILES = {
    "seq_ddp": "reference_seq_ddp.yaml",
    "para_m2_n4_s1_t1": "reference_para_m2_n4_s1_t1_ddp.yaml",
    "para_m2_n2_s1_t2": "reference_para_m2_n2_s1_t2_ddp.yaml",
    "para_m2_n1_s2_t2": "reference_para_m2_n1_s2_t2_ddp.yaml",
}

STALE_OUTPUTS = (
    "correctness_overview.png",
    "output_sum_delta_vs_baseline.png",
    "loss_rmse_vs_baseline.png",
    "output_sum_rmse_vs_baseline.png",
    "avg_step_time.png",
    "peak_memory.png",
)


def remove_stale_outputs(result_dir):
    for filename in STALE_OUTPUTS:
        path = result_dir / filename
        if path.exists():
            path.unlink()


def maybe_float(value):
    if value in (None, ""):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = dict(row)
            item["iter"] = int(row["iter"])
            for key, value in row.items():
                if key == "iter":
                    continue
                item[key] = maybe_float(value)
            rows.append(item)
    return rows


def load_json(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def case_config_path(config_dir, case_name):
    if config_dir is None:
        return None
    filename = CASE_CONFIG_FILES.get(case_name, f"{case_name}.yaml")
    return config_dir / filename


def find_cases(result_dir, config_dir=None, selected_cases=None):
    cases = {}
    if selected_cases:
        case_names = selected_cases
    else:
        case_names = [path.name for path in sorted(result_dir.iterdir()) if path.is_dir()]

    for name in case_names:
        case_dir = result_dir / name
        csv_path = case_dir / "loss.csv"
        if not csv_path.exists():
            continue
        cfg_path = case_config_path(config_dir, name)
        cases[name] = {
            "dir": case_dir,
            "csv": csv_path,
            "rows": load_csv(csv_path),
            "metadata": load_json(case_dir / "metadata.json"),
            "config": load_yaml(cfg_path) if cfg_path is not None else {},
        }
    return cases


def series(rows, key):
    return [row[key] for row in rows if key in row and isinstance(row[key], (int, float))]


def max_abs_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return ""
    return max(abs(a[i] - b[i]) for i in range(n))


def final_delta(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return ""
    return a[n - 1] - b[n - 1]


def case_label(name):
    if name in ("seq", "seq_ddp", "sequential"):
        return "Serial"
    if name.startswith("para_"):
        name = name[len("para_"):]
    if name.startswith("m"):
        values = []
        for part in name.split("_"):
            if len(part) >= 2 and part[0] in ("m", "n", "s", "t") and part[1:].isdigit():
                values.append(part[1:])
        if len(values) == 4:
            return f"({','.join(values)})"
    return name


def choose_baseline(cases, requested):
    if requested:
        if requested not in cases:
            raise ValueError(f"baseline {requested!r} not found in {sorted(cases)}")
        return requested
    for name in cases:
        if name.startswith("seq"):
            return name
    return next(iter(cases))


def config_value(item, key, default=""):
    metadata = item.get("metadata", {})
    config = item.get("config", {})
    value = config.get(key, None)
    if value not in (None, ""):
        return value
    return metadata.get(key, default)


def csv_value(item, key, default=""):
    rows = item["rows"]
    if not rows:
        return default
    return rows[0].get(key, default)


def plot_loss_curves(result_dir, cases):
    plt.figure(figsize=(9.5, 5.2))
    for name, item in cases.items():
        rows = item["rows"]
        plt.plot(series(rows, "iter"), series(rows, "loss"), marker="o", linewidth=1.5, label=case_label(name))
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss correctness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "loss_curves.png", dpi=180)
    plt.close()


def plot_loss_delta(result_dir, cases, baseline_name):
    baseline = series(cases[baseline_name]["rows"], "loss")
    baseline_label = case_label(baseline_name)
    has_line = False

    plt.figure(figsize=(9.5, 5.2))
    for name, item in cases.items():
        if name == baseline_name:
            continue
        values = series(item["rows"], "loss")
        n = min(len(values), len(baseline))
        xs = series(item["rows"], "iter")[:n]
        ys = [values[i] - baseline[i] for i in range(n)]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"{case_label(name)} - {baseline_label}")
        has_line = True

    plt.axhline(0.0, color="black", linewidth=1.0)
    if not has_line:
        plt.text(0.5, 0.5, "no comparison cases", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("step")
    plt.ylabel("loss delta")
    plt.title(f"Loss delta vs {baseline_label}")
    plt.grid(True, alpha=0.3)
    if has_line:
        plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "loss_delta_vs_baseline.png", dpi=180)
    plt.close()


def build_summary_rows(cases, baseline_name):
    baseline_loss = series(cases[baseline_name]["rows"], "loss")
    summary_rows = []

    for name, item in cases.items():
        rows = item["rows"]
        loss_values = series(rows, "loss")
        first_row = rows[0] if rows else {}
        last_row = rows[-1] if rows else {}

        summary_rows.append({
            "case": name,
            "label": case_label(name),
            "baseline_case": baseline_name,
            "steps": len(rows),
            "loss0": first_row.get("loss", ""),
            "final_loss": last_row.get("loss", ""),
            "final_loss_delta_vs_baseline": final_delta(loss_values, baseline_loss),
            "loss_max_abs_diff_vs_baseline": max_abs_diff(loss_values, baseline_loss),
            "output_sum0": first_row.get("output_sum", ""),
            "final_output_sum": last_row.get("output_sum", ""),
            "final_peak_memory_mb": last_row.get("peak_memory_mb", ""),
            "model_type": config_value(item, "model_type"),
            "optimizer_config": config_value(item, "optimizer_config"),
            "precision": config_value(item, "precision"),
            "wp_topo": config_value(item, "wp_topo"),
            "xfmr_wp_topo": config_value(item, "xfmr_wp_topo"),
            "xfmr_sp_size": config_value(item, "xfmr_sp_size"),
            "tensor_parallel_size": config_value(item, "tensor_parallel_size"),
            "sp_tp_placement": config_value(item, "sp_tp_placement"),
            "window_assignment_mode": config_value(item, "window_assignment_mode"),
            "padding_policy": csv_value(item, "padding_policy", config_value(item, "padding_policy", "scale")),
            "padding_scale": csv_value(item, "padding_scale", config_value(item, "padding_scale")),
            "padded_shape": csv_value(item, "padded_shape"),
            "num_windows": csv_value(item, "num_windows"),
            "patch_size": config_value(item, "patch_size"),
            "window_size": config_value(item, "window_size"),
            "embedding_dim": config_value(item, "embedding_dim"),
            "num_heads": config_value(item, "num_heads"),
            "num_layers": config_value(item, "num_layers"),
        })
    return summary_rows


def write_summary(result_dir, summary_rows):
    if not summary_rows:
        return
    path = result_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    config_dir = Path(args.config_dir).resolve() if args.config_dir else None
    cases = find_cases(result_dir, config_dir, args.cases)
    if not cases:
        raise RuntimeError(f"no loss.csv files found under {result_dir}")

    baseline_name = choose_baseline(cases, args.baseline)
    summary_rows = build_summary_rows(cases, baseline_name)

    remove_stale_outputs(result_dir)
    plot_loss_curves(result_dir, cases)
    plot_loss_delta(result_dir, cases, baseline_name)
    write_summary(result_dir, summary_rows)

    print(f"baseline: {baseline_name}")
    print(f"wrote loss plots and summary to {result_dir}")


if __name__ == "__main__":
    main()
