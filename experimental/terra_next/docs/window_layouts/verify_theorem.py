#!/usr/bin/env python3
"""Finite, independent checks for the TERRA row/column shift-volume theorem.

Python >= 3.8; standard library only. No tensors, accelerators, network calls,
training, or performance measurements. Exhaustive finite checks supplement the
written proof; they do not replace it or establish a novelty/priority claim.

Run: python -B verify_theorem.py --output-dir .
"""
import argparse
import bisect
import datetime
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback

POLICIES = ("prefix_remainder", "paper_floor")
SCANS = ("row", "column")
REFERENCE_BUILD = "1965ceed1f12650076ffeba2ab5ea4c3de56c62333c64103748e82d57deeac87"
COUNTS = {}


def check(condition, category, detail=None):
    COUNTS[category] = COUNTS.get(category, 0) + 1
    if not condition:
        raise AssertionError(json.dumps({"category": category, "detail": detail}, sort_keys=True))


def cuts_for(n, parts, policy):
    if n < 1 or parts < 1:
        raise ValueError("Positive window count and rank count required")
    if policy == "prefix_remainder":
        q, r = divmod(n, parts)
        return [p * q + min(p, r) for p in range(parts + 1)]
    if policy == "paper_floor":
        return [p * n // parts for p in range(parts + 1)]
    raise ValueError("Unsupported balanced cut policy")


def owner_from_scan_index(index, n, parts, policy):
    """Pixel oracle: closed arithmetic assignment, independent of grid filling."""
    if policy == "paper_floor":
        return ((index + 1) * parts - 1) // n
    if policy == "prefix_remainder":
        q, r = divmod(n, parts)
        long_prefix = (q + 1) * r
        if index < long_prefix:
            return index // (q + 1)
        return r + (index - long_prefix) // q
    raise ValueError("Unsupported policy")


def owner_grid(rows, columns, parts, scan, policy):
    """Neighbour method: fill every cut interval explicitly in scan order."""
    cuts = cuts_for(rows * columns, parts, policy)
    grid = [[None for _ in range(columns)] for _ in range(rows)]
    for rank in range(parts):
        for index in range(cuts[rank], cuts[rank + 1]):
            if scan == "row":
                i, j = divmod(index, columns)
            elif scan == "column":
                j, i = divmod(index, rows)
            else:
                raise ValueError("Unsupported scan")
            grid[i][j] = rank
    return grid


def pixel_owners(rows, columns, parts, window, scan, policy):
    """Generate ownership at every padded-domain token, without neighbour areas."""
    pixels = []
    for y in range(rows * window):
        row = []
        for x in range(columns * window):
            i, j = y // window, x // window
            index = i * columns + j if scan == "row" else j * rows + i
            row.append(owner_from_scan_index(index, rows * columns, parts, policy))
        pixels.append(row)
    return pixels


def pixel_matrix(pixels, parts, shift_y, shift_x):
    """Direct oracle: visit each token once; shifted owner uses global coordinates."""
    height, width = len(pixels), len(pixels[0])
    matrix = [[0 for _ in range(parts)] for _ in range(parts)]
    for y in range(height):
        target_row = pixels[(y - shift_y) % height]
        source_row = pixels[y]
        for x in range(width):
            matrix[source_row[x]][target_row[(x - shift_x) % width]] += 1
    return matrix


def neighbour_matrix(grid, parts, window, delta, direction=1):
    """Exact four subregions per window; no use of the proposed closed form."""
    rows, columns = len(grid), len(grid[0])
    matrix = [[0 for _ in range(parts)] for _ in range(parts)]
    terms = ((0, 0, (window - delta) ** 2),
             (direction, 0, delta * (window - delta)),
             (0, direction, delta * (window - delta)),
             (direction, direction, delta ** 2))
    for i in range(rows):
        for j in range(columns):
            source = grid[i][j]
            for dy, dx, area in terms:
                target = grid[(i - dy) % rows][(j - dx) % columns]
                matrix[source][target] += area
    return matrix


def neighbour_counts(grid):
    rows, columns = len(grid), len(grid[0])
    horizontal = vertical = diagonal = 0
    for i in range(rows):
        for j in range(columns):
            value = grid[i][j]
            horizontal += value != grid[i][(j - 1) % columns]
            vertical += value != grid[(i - 1) % rows][j]
            diagonal += value != grid[(i - 1) % rows][(j - 1) % columns]
    return {"horizontal": horizontal, "vertical": vertical, "diagonal": diagonal}


def remote(matrix):
    return sum(value for i, row in enumerate(matrix) for j, value in enumerate(row) if i != j)


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def closed_form(rows, columns, parts, window, delta, scan, policy):
    """Only the two balanced policies are admissible; arbitrary cuts are rejected."""
    if not (2 <= parts <= min(rows, columns) and window >= 2 and 0 < delta < window):
        raise ValueError("Outside the balanced-cut closed-form assumptions")
    cuts = cuts_for(rows * columns, parts, policy)
    stride = columns if scan == "row" else rows
    if scan not in SCANS:
        raise ValueError("Unsupported scan")
    broken = sum(cut % stride != 0 for cut in cuts[1:-1])
    return parts * stride * window * delta + 2 * delta * (window - delta) * broken, broken


def validate_pixel_oracles(small_grid, max_window):
    cases = pixel_visits = comparisons = closed_comparisons = 0
    for rows in range(1, small_grid + 1):
        for columns in range(1, small_grid + 1):
            n = rows * columns
            # Includes P=1, every owner holding one window, and one empty owner.
            for parts in range(1, n + 2):
                for policy in POLICIES:
                    for scan in SCANS:
                        grid = owner_grid(rows, columns, parts, scan, policy)
                        for i in range(rows):
                            for j in range(columns):
                                index = i * columns + j if scan == "row" else j * rows + i
                                check(grid[i][j] == owner_from_scan_index(index, n, parts, policy),
                                      "independent_owner_mapping", (rows, columns, parts, policy, scan, i, j))
                        for window in range(1, max_window + 1):
                            pixels = pixel_owners(rows, columns, parts, window, scan, policy)
                            for delta in range(window):
                                forward = pixel_matrix(pixels, parts, delta, delta)
                                backward = pixel_matrix(pixels, parts, -delta, -delta)
                                cases += 1
                                pixel_visits += 2 * rows * columns * window * window
                                check(backward == transpose(forward), "pixel_forward_reverse_transpose")
                                for direction, actual in ((1, forward), (-1, backward)):
                                    expected = neighbour_matrix(grid, parts, window, delta, direction)
                                    check(actual == expected, "pixel_neighbour_matrix_equality",
                                          (rows, columns, parts, policy, scan, window, delta, direction))
                                    check(sum(map(sum, actual)) == n * window * window, "token_conservation")
                                    comparisons += 1
                                    if 2 <= parts <= min(rows, columns) and 0 < delta < window:
                                        formula, _ = closed_form(rows, columns, parts, window, delta, scan, policy)
                                        check(remote(actual) == formula, "pixel_closed_form_equality")
                                        closed_comparisons += 1
    return {"max_window_grid_axis": small_grid, "max_window_side": max_window,
            "policies": list(POLICIES), "scans": list(SCANS), "owner_and_shift_cases": cases,
            "forward_reverse_matrix_comparisons": comparisons,
            "direct_pixel_visits": pixel_visits, "pixel_closed_form_comparisons": closed_comparisons,
            "rank_range": "1 through R*C+1 inclusive", "shift_range": "all delta=0..w-1; both directions"}


def validate_grid_lemmas(grid_limit, max_window):
    shape_rank_cases = policy_cases = shape_theorem_checks = form_checks = 0
    for rows in range(2, grid_limit + 1):
        for columns in range(2, grid_limit + 1):
            for parts in range(2, min(rows, columns) + 1):
                shape_rank_cases += 1
                for policy in POLICIES:
                    policy_cases += 1
                    counts = {}
                    for scan in SCANS:
                        grid = owner_grid(rows, columns, parts, scan, policy)
                        counts[scan] = neighbour_counts(grid)
                        _, broken = closed_form(rows, columns, parts, 2, 1, scan, policy)
                        c = counts[scan]
                        if scan == "row":
                            expected = {"horizontal": 2 * broken, "vertical": parts * columns,
                                        "diagonal": parts * columns}
                        else:
                            expected = {"horizontal": parts * rows, "vertical": 2 * broken,
                                        "diagonal": parts * rows}
                        check(c == expected, "balanced_neighbour_count_lemma",
                              (rows, columns, parts, scan, policy, c, expected))
                        for window in range(2, max_window + 1):
                            for delta in range(1, window):
                                summed = delta * (window - delta) * (c["horizontal"] + c["vertical"]) + delta ** 2 * c["diagonal"]
                                formula, _ = closed_form(rows, columns, parts, window, delta, scan, policy)
                                check(summed == formula, "neighbour_closed_form_equality")
                                form_checks += 1
                    for window in range(2, max_window + 1, 2):
                        delta = window // 2
                        values = []
                        for scan in SCANS:
                            c = counts[scan]
                            values.append(delta ** 2 * sum(c.values()))
                        row, column = values
                        correct = column < row if columns > rows else (row < column if rows > columns else row == column)
                        check(correct, "even_half_shift_orientation_theorem",
                              (rows, columns, parts, policy, window, row, column))
                        shape_theorem_checks += 1
    return {"grid_axes_inclusive": [2, grid_limit], "shape_rank_cases": shape_rank_cases,
            "balanced_policy_cases": policy_cases, "closed_form_comparisons": form_checks,
            "half_shift_wide_square_tall_checks": shape_theorem_checks}


def fixed_examples():
    result = []
    for rows, columns, parts, window in ((43, 90, 4, 6), (8, 17, 4, 32)):
        for policy in POLICIES:
            out = {"grid": [rows, columns], "ranks": parts, "window_side": window,
                   "delta": window // 2, "cut_policy": policy,
                   "interior_scan_cuts": cuts_for(rows * columns, parts, policy)[1:-1], "results": {}}
            for scan in SCANS:
                pixels = pixel_owners(rows, columns, parts, window, scan, policy)
                actual = pixel_matrix(pixels, parts, window // 2, window // 2)
                backwards = pixel_matrix(pixels, parts, -window // 2, -window // 2)
                neighbour = neighbour_matrix(owner_grid(rows, columns, parts, scan, policy), parts, window, window // 2)
                formula, broken = closed_form(rows, columns, parts, window, window // 2, scan, policy)
                check(actual == neighbour and backwards == transpose(actual), "current_example_oracle_equality")
                check(remote(actual) == formula, "current_example_closed_form")
                if window == 6:
                    expected = 6534 if scan == "row" else (3150 if policy == "prefix_remainder" else 3132)
                else:
                    expected = 34816 if scan == "row" else 17920
                check(remote(actual) == expected, "current_example_expected_count")
                out["results"][scan] = {"broken_cuts": broken, "remote_token_rows": remote(actual),
                                          "source_to_destination_token_matrix": actual}
            result.append(out)
    return result


def assumption_boundaries():
    result = []
    def case(name, rows, columns, parts, window, dy, dx, expected):
        measured = {}
        for scan in SCANS:
            pixels = pixel_owners(rows, columns, parts, window, scan, "prefix_remainder")
            measured[scan] = remote(pixel_matrix(pixels, parts, dy, dx))
        check(measured == expected, "assumption_boundary_expected_count", (name, measured, expected))
        result.append({"name": name, "grid": [rows, columns], "ranks": parts, "window_side": window,
                       "shift_y_x": [dy, dx], "cut_policy": "prefix_remainder", "remote_token_rows": measured})
    case("P=1 has no remote transport; closed form is not applicable", 3, 5, 1, 2, 1, 1, {"row": 0, "column": 0})
    case("odd window can reverse wide-grid preference", 5, 6, 5, 3, 1, 1, {"row": 90, "column": 91})
    case("P=R*C: unique-window owners can tie on a wide grid", 2, 3, 6, 2, 1, 1, {"row": 18, "column": 18})
    case("tall grid reverses the half-shift comparison", 8, 4, 4, 2, 1, 1, {"row": 32, "column": 64})
    case("square grid ties", 4, 4, 4, 2, 1, 1, {"row": 32, "column": 32})
    case("horizontal-only shift can favour row ownership on a wide grid", 4, 8, 4, 2, 0, 1, {"row": 0, "column": 32})
    for arguments in ((3, 5, 1, 2, 1, "row", "prefix_remainder"),
                      (2, 3, 6, 2, 1, "row", "prefix_remainder"),
                      (4, 4, 2, 1, 0, "row", "prefix_remainder"),
                      (4, 4, 2, 2, 1, "row", "arbitrary_cuts")):
        rejected = False
        try:
            closed_form(*arguments)
        except ValueError:
            rejected = True
        check(rejected, "closed_form_invalid_assumptions_rejected", arguments)
    # Negative regression: min interval length >= C alone does NOT prove A_diag=P*C.
    rows = columns = 4
    parts, window, cuts = 3, 2, [0, 5, 9, 16]
    pixels = [[bisect.bisect_right(cuts[1:-1], (y // window) * columns + x // window)
               for x in range(columns * window)] for y in range(rows * window)]
    actual = remote(pixel_matrix(pixels, parts, 1, 1))
    grid = [[bisect.bisect_right(cuts[1:-1], i * columns + j) for j in range(columns)] for i in range(rows)]
    c = neighbour_counts(grid)
    erroneous = 2 * (parts * columns + 2)
    check(actual == 27 and c == {"horizontal": 4, "vertical": 12, "diagonal": 11}
          and erroneous == 28 and actual != erroneous, "arbitrary_cut_weakening_negative_regression")
    result.append({"name": "arbitrary cuts with min length >= max(R,C) do not satisfy the balanced lemma",
                   "grid": [4, 4], "ranks": 3, "window_side": 2, "shift_y_x": [1, 1],
                   "custom_segment_lengths": [5, 4, 7], "neighbour_counts": c,
                   "actual_remote_token_rows": actual, "invalid_weakened_formula_prediction": erroneous,
                   "theorem_applicable": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--small-grid", type=int, default=5)
    parser.add_argument("--grid-limit", type=int, default=30)
    parser.add_argument("--max-window", type=int, default=6)
    parser.add_argument("--source-layout", type=Path)
    args = parser.parse_args()
    if args.small_grid < 2 or args.grid_limit < 2 or args.max_window < 2:
        parser.error("grid/window coverage limits must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    timer = time.perf_counter()
    receipt = {"format": "terra_balanced_window_shift_theorem_verification_v1", "status": "RUNNING",
               "scope": "Finite mathematical verification of logical token counts; no hardware or training performance",
               "formalization_date_local": "2026-11-17", "started_utc": started.isoformat(),
               "python": sys.version, "platform": platform.platform(), "pure_standard_library": True,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "policies": list(POLICIES), "failures": [],
               "limits": {"small_grid": args.small_grid, "medium_grid": args.grid_limit, "max_window": args.max_window},
               "mapping_reference": {"source_build_hash": REFERENCE_BUILD,
                   "path": "terra/core/FMP/layout.py", "description": "v8 prefix-remainder mapping; paper_floor is a distinct analysed policy",
                   "source_identity_is_provenance_not_new_runtime_validation": True},
               "claim_limits": ["Finite verification supplements the written proof; it is not a substitute for it.",
                   "No entry/exit, convolution halo, gradient collectives, local packing, latency, TFLOPS, or wire-byte estimate.",
                   "No novel/first/priority claim is inferred from these checks.",
                   "Existing full-resolution NPU BF16 strict FAIL is unchanged and not explained by this verifier."]}
    if args.source_layout:
        receipt["mapping_reference"]["local_source_path"] = str(args.source_layout.resolve())
        receipt["mapping_reference"]["local_source_sha256"] = hashlib.sha256(args.source_layout.read_bytes()).hexdigest()
    lines = ["Mathematical verification only; no training or hardware timing.", "Policies: " + ", ".join(POLICIES)]
    try:
        receipt["small_pixel_oracle_checks"] = validate_pixel_oracles(args.small_grid, args.max_window)
        lines.append("PASS: independent pixel oracle, neighbour matrices, and admissible closed forms")
        receipt["medium_neighbour_lemma_checks"] = validate_grid_lemmas(args.grid_limit, args.max_window)
        lines.append("PASS: balanced-cut neighbour lemma and even half-shift wide/square/tall theorem")
        receipt["current_case_examples"] = fixed_examples()
        lines.append("PASS: current W6/W32 examples with distinct v8-prefix and paper-floor policies")
        receipt["assumption_boundary_examples"] = assumption_boundaries()
        lines.append("PASS: degenerate, odd-window, rank-degree, shift-direction, and arbitrary-cut counterexamples")
        receipt["status"] = "PASS"
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failures"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        lines.append("FAIL: " + repr(exc))
    receipt["checks_executed"] = COUNTS
    receipt["failure_count"] = len(receipt["failures"])
    receipt["verification_script_wall_seconds_not_training_performance"] = time.perf_counter() - timer
    receipt["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lines.append(json.dumps({"status": receipt["status"], "checks_executed": COUNTS, "failures": receipt["failure_count"]}, sort_keys=True))
    output = args.output_dir / "verification.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "verification.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])
    print("receipt:", output)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
