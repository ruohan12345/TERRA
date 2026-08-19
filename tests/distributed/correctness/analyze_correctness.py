#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
    font_manager = None


DEFAULT_CASES = [
    "seq_fsdp",
    "topo_wp",
    "topo_wp_sp",
    "topo_wp_tp",
    "topo_wp_sp_tp",
]


_TIMES_FONT_REGISTERED = False


def register_times_new_roman():
    global _TIMES_FONT_REGISTERED
    if _TIMES_FONT_REGISTERED or font_manager is None:
        return

    font_dir = Path(__file__).resolve().parents[2] / "font"
    for font_path in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        candidate = font_dir / font_path
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    _TIMES_FONT_REGISTERED = True


def maybe_float(value):
    if value in (None, ""):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            item = {}
            for key, value in row.items():
                item[key] = maybe_float(value)
            if "iter" in item and isinstance(item["iter"], float):
                item["iter"] = int(item["iter"])
            elif "step" in item and isinstance(item["step"], float):
                item["iter"] = int(item["step"])
            else:
                item["iter"] = idx
            rows.append(item)
    return rows


def simple_yaml_load(path):
    data = {}
    if yaml is not None:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        if raw.startswith(" ") or raw.startswith("\t"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value in ("True", "False"):
            value = value == "True"
        else:
            try:
                value = int(value)
            except ValueError:
                pass
        data[key] = value
    return data


def activation_block_text(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    block = []
    in_block = False
    for line in lines:
        if line.startswith("activation:"):
            in_block = True
            block.append(line.rstrip())
            continue
        if in_block:
            if line and not line.startswith(" "):
                break
            block.append(line.rstrip())
    return "\n".join(block)


def find_config(config_dir, case_name):
    direct = config_dir / f"{case_name}.yaml"
    if direct.exists():
        return direct
    matches = list(config_dir.rglob(f"{case_name}.yaml"))
    return matches[0] if matches else None


def load_cases(result_dir, config_dir, selected_cases):
    cases = {}
    for case_name in selected_cases:
        case_dir = result_dir / case_name
        cfg_path = find_config(config_dir, case_name)
        failed_path = case_dir / "failed.txt"
        loss_path = case_dir / "loss.csv"
        status = "success" if loss_path.exists() else "missing"
        if failed_path.exists():
            status = "failed"
        config = simple_yaml_load(cfg_path) if cfg_path is not None and cfg_path.exists() else {}
        cases[case_name] = {
            "name": case_name,
            "dir": case_dir,
            "config_path": cfg_path,
            "config": config,
            "activation_block": activation_block_text(cfg_path) if cfg_path is not None and cfg_path.exists() else "",
            "rows": load_csv(loss_path) if loss_path.exists() else [],
            "status": status,
            "failed_text": failed_path.read_text(encoding="utf-8", errors="replace").strip() if failed_path.exists() else "",
        }
    return cases


def series(rows, key):
    return [row[key] for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]


def loss_series(item):
    return series(item["rows"], "loss")


def x_series(item):
    xs = series(item["rows"], "iter")
    if xs:
        return xs
    return list(range(len(item["rows"])))


def max_abs_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return ""
    return max(abs(a[i] - b[i]) for i in range(n))


def mean_abs_diff(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return ""
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def final_delta(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return ""
    return a[n - 1] - b[n - 1]


def config_value(item, key, default=""):
    value = item["config"].get(key, default)
    return "" if value is None else value


def case_label(item):
    return config_value(item, "case_label", item["name"])


def topology_label(item):
    return config_value(item, "topology_label", "")


def check_activation_consistency(cases):
    blocks = {}
    for name, item in cases.items():
        if item["config_path"] is None:
            continue
        blocks[name] = item["activation_block"]
    unique = {text for text in blocks.values()}
    if len(unique) > 1:
        details = "\n".join(f"--- {name} ---\n{text}" for name, text in blocks.items())
        raise RuntimeError("activation/checkpoint configs are not identical across cases:\n" + details)


def build_summary(cases, baseline_name):
    baseline_loss = loss_series(cases[baseline_name])
    rows = []
    for name, item in cases.items():
        losses = loss_series(item)
        mem = series(item["rows"], "peak_memory_mb")
        step_time = series(item["rows"], "step_time_s")
        final_loss = losses[-1] if losses else ""
        rows.append({
            "case": name,
            "label": case_label(item),
            "topology_label": topology_label(item),
            "status": item["status"],
            "steps": len(item["rows"]),
            "final_loss": final_loss,
            "final_loss_delta_vs_baseline": final_delta(losses, baseline_loss),
            "loss_max_abs_diff_vs_baseline": max_abs_diff(losses, baseline_loss),
            "loss_mean_abs_diff_vs_baseline": mean_abs_diff(losses, baseline_loss),
            "max_peak_memory_mb": max(mem) if mem else "",
            "avg_step_time_s": sum(step_time) / len(step_time) if step_time else "",
            "model_type": config_value(item, "model_type"),
            "optimizer_config": config_value(item, "optimizer_config"),
            "wp_topo": config_value(item, "wp_topo"),
            "xfmr_wp_topo": config_value(item, "xfmr_wp_topo"),
            "xfmr_sp_size": config_value(item, "xfmr_sp_size"),
            "tensor_parallel_size": config_value(item, "tensor_parallel_size"),
            "window_topology": config_value(item, "window_topology"),
            "window_assignment_mode": config_value(item, "window_assignment_mode"),
            "activation_mode": config_value(item, "activation", ""),
            "failed_text": item["failed_text"],
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_plot_style():
    if plt is None:
        return
    register_times_new_roman()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 11,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig, path_without_suffix):
    for suffix in (".png", ".pdf"):
        fig.savefig(str(path_without_suffix) + suffix, dpi=220, bbox_inches="tight")


PLOT_STYLE = {
    "seq_fsdp": {"color": "#356CA5", "linestyle": "-", "linewidth": 1.35, "zorder": 5},
    "topo_wp": {"color": "#E67E22", "linestyle": "-", "linewidth": 1.0, "zorder": 2},
    "topo_wp_sp": {"color": "#4C9F70", "linestyle": "-", "linewidth": 1.0, "zorder": 3},
    "topo_wp_tp": {"color": "#C53A3A", "linestyle": "-", "linewidth": 1.1, "zorder": 4},
    "topo_wp_sp_tp": {"color": "#8D6BB8", "linestyle": "-", "linewidth": 1.0, "zorder": 4},
}


def plot_case_label(name, item):
    # Keep the paper figure concise without changing the stored case metadata.
    if name == "seq_fsdp":
        return "Serial"
    return case_label(item)


def draw_loss_panel(ax, cases, split_step, panel, show_legend):
    late_values = []
    for name, item in cases.items():
        losses = loss_series(item)
        if not losses:
            continue
        xs = x_series(item)[:len(losses)]
        if panel == "early":
            plot_x = [x for x in xs if x <= split_step]
            plot_y = losses[:len(plot_x)]
        else:
            plot_x = [x for x in xs if x >= split_step]
            plot_y = losses[len(xs) - len(plot_x):]
        style = PLOT_STYLE.get(name, {"linewidth": 1.0})
        # Retain the original full case labels in the legend.
        ax.plot(plot_x, plot_y, label=plot_case_label(name, item), **style)
        if panel == "late":
            late_values.extend(plot_y)

    ax.set_ylabel("Loss")
    ax.grid(axis="y", alpha=0.26)
    if panel == "early":
        ax.set_xlim(0, split_step)
        ax.margins(y=0.06)
        if show_legend:
            ax.legend(
                loc="upper right",
                ncol=1,
                columnspacing=0.6,
                handlelength=1.8,
                handletextpad=0.45,
                borderpad=0.30,
                framealpha=0.92,
            )
        return

    # Training steps are zero-indexed in the metrics CSV, while the paper-facing
    # range denotes the total number of optimization steps.
    late_end = max((len(loss_series(item)) for item in cases.values()), default=split_step)
    ax.set_xlim(split_step, late_end)
    late_ticks = [split_step]
    late_ticks.extend(tick for tick in range(200, late_end, 200) if tick > split_step)
    if late_end not in late_ticks:
        late_ticks.append(late_end)
    ax.set_xticks(late_ticks)
    ax.margins(x=0.01)
    if late_values:
        lower = min(late_values)
        upper = max(late_values)
        padding = max((upper - lower) * 0.10, max(abs(upper), 1.0e-6) * 0.025)
        ax.set_ylim(max(0.0, lower - padding), upper + padding)


def plot_loss_curves(output_dir, cases, split_step):
    if plt is None:
        return
    setup_plot_style()

    # Paper-ready standalone panels for subfloat use.
    for panel, suffix in (("early", "loss_curves_early"), ("late", "loss_curves_late")):
        fig, ax = plt.subplots(figsize=(3.00, 1.95))
        draw_loss_panel(ax, cases, split_step, panel, show_legend=(panel == "early"))
        ax.set_xlabel("Training step")
        fig.subplots_adjust(left=0.21, right=0.985, bottom=0.23, top=0.985)
        savefig(fig, output_dir / suffix)
        plt.close(fig)

    # Keep a combined horizontal preview in addition to the two subfigures.
    fig, (early_ax, late_ax) = plt.subplots(
        1,
        2,
        figsize=(6.20, 1.95),
        gridspec_kw={"wspace": 0.32},
    )
    draw_loss_panel(early_ax, cases, split_step, "early", show_legend=True)
    draw_loss_panel(late_ax, cases, split_step, "late", show_legend=False)
    early_ax.set_xlabel("Training step")
    late_ax.set_xlabel("Training step")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.23, top=0.985)
    savefig(fig, output_dir / "loss_curves")
    plt.close(fig)


def plot_loss_delta(output_dir, cases, baseline_name):
    if plt is None:
        return
    baseline = loss_series(cases[baseline_name])
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    for name, item in cases.items():
        if name == baseline_name:
            continue
        losses = loss_series(item)
        n = min(len(losses), len(baseline))
        if n == 0:
            continue
        delta = [losses[i] - baseline[i] for i in range(n)]
        ax.plot(
            x_series(item)[:n],
            delta,
            label=case_label(item),
            **PLOT_STYLE.get(name, {"linewidth": 1.0}),
        )
    ax.axhline(0.0, color="black", linewidth=0.7, zorder=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss difference")
    ax.margins(x=0.01, y=0.10)
    ax.grid(axis="y", alpha=0.26)
    ax.legend(
        loc="upper right",
        ncol=2,
        columnspacing=0.7,
        handlelength=1.8,
        handletextpad=0.45,
        borderpad=0.35,
        framealpha=0.92,
    )
    fig.tight_layout(pad=0.28)
    savefig(fig, output_dir / "loss_delta_vs_baseline")
    plt.close(fig)


def write_report(output_dir, summary_rows, baseline_name):
    lines = [
        "# seq_ddp_vs_parallel_correctness report",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        "All generated cases use identical activation/checkpoint configuration.",
        "",
        "| case | status | final loss | final delta | max abs diff | mean abs diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['status']} | {row['final_loss']} | "
            f"{row['final_loss_delta_vs_baseline']} | {row['loss_max_abs_diff_vs_baseline']} | "
            f"{row['loss_mean_abs_diff_vs_baseline']} |"
        )
    lines.extend([
        "",
        "Generated figures:",
        "- `loss_curves_early.png/pdf`",
        "- `loss_curves_late.png/pdf`",
        "- `loss_curves.png/pdf` (combined horizontal preview)",
        "- `loss_delta_vs_baseline.png/pdf`",
        "",
    ])
    output_dir.joinpath("correctness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--baseline", default="seq_fsdp")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--loss_split_step", type=int, default=100)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = args.cases or DEFAULT_CASES
    cases = load_cases(result_dir, config_dir, selected_cases)
    if args.baseline not in cases or not loss_series(cases[args.baseline]):
        raise RuntimeError(f"baseline {args.baseline!r} is missing or has no loss.csv under {result_dir}")

    check_activation_consistency(cases)
    summary_rows = build_summary(cases, args.baseline)
    write_csv(output_dir / "summary.csv", summary_rows)
    plot_loss_curves(output_dir, cases, args.loss_split_step)
    plot_loss_delta(output_dir, cases, args.baseline)
    write_report(output_dir, summary_rows, args.baseline)
    print(f"[correctness] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
