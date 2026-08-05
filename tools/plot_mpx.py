"""Render the MPX thesis figures from the CSVs written by `parse_mpx_logs.py`.

Reads committed data, never raw logs, so every figure regenerates from the
repository alone. Emits vector PDF for LaTeX plus a PNG preview.

Two figures so far:

  reach      knowledge vs trials, one line per seed -- the reach claim, and the
             seed variance that goes with it
  anatomy    one seed as three stacked panels sharing the trials axis --
             knowledge, mean specificity of reliable rules, population size

`anatomy` is stacked panels rather than one plot with several y-scales on
purpose: the three measures have unrelated units, and overlaying them on twin
axes would let the eye read crossings that carry no meaning. It is the figure
that shows *why* the apparent plateau is not a failure -- specificity is still
falling toward the ideal and the population is still condensing while knowledge
looks flat.

Palette is the validated categorical order (worst adjacent CVD dE 35.9 on white);
aqua sits marginally under 3:1 on white, so every series is direct-labelled.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SERIES_COLORS = ["#2a78d6", "#1baf7a", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

LINE_WIDTH = 1.4
IDEAL_MARKER_SIZE = 26


def address_bits(size):
    """MPX-k has a address bits with k = a + 2**a; a correct rule specifies a+1."""
    bits = 1
    while bits + 2**bits < size:
        bits += 1
    if bits + 2**bits != size:
        raise ValueError(f"{size} is not a valid multiplexer size")
    return bits


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows, *columns):
    for row in rows:
        for column in columns:
            row[column] = float(row[column]) if row[column] else 0.0
    return rows


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
    })


def millions(value, _position):
    return f"{value / 1e6:g}M"


def spread(points, min_gap, x_tolerance, ceiling):
    """Nudge end-of-line labels apart vertically, but only where they actually collide.

    Labels collide only when their anchors are close in *both* axes, so lines that
    finish at the same knowledge but thousands of trials apart keep their true y.
    Within a cluster the stack is centred on the group, so it stays under `ceiling`
    instead of marching off the top of the axes.
    """
    adjusted = [point["y"] for point in points]
    order = sorted(range(len(points)), key=lambda index: points[index]["x"])

    cluster = [order[0]]
    clusters = [cluster]
    for index in order[1:]:
        if points[index]["x"] - points[cluster[-1]]["x"] <= x_tolerance:
            cluster.append(index)
        else:
            cluster = [index]
            clusters.append(cluster)

    for cluster in clusters:
        if len(cluster) == 1:
            continue
        cluster.sort(key=lambda index: points[index]["y"])
        centre = sum(points[index]["y"] for index in cluster) / len(cluster)
        start = centre - min_gap * (len(cluster) - 1) / 2
        for position, index in enumerate(cluster):
            adjusted[index] = min(start + position * min_gap, ceiling)
    return adjusted


def color_map(rows):
    """Stable seed -> hue over the whole dataset, so filtering never repaints."""
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) > len(SERIES_COLORS):
        raise SystemExit(
            f"{len(seeds)} seeds exceeds the {len(SERIES_COLORS)}-slot palette; "
            "use small multiples rather than generating hues"
        )
    return {seed: SERIES_COLORS[index] for index, seed in enumerate(seeds)}


def group_by_seed(rows, size, variant):
    grouped = defaultdict(list)
    for row in rows:
        if int(row["size"]) == size and row["variant"] == variant:
            grouped[int(row["seed"])].append(row)
    for points in grouped.values():
        points.sort(key=lambda row: row["trials"])
    return dict(sorted(grouped.items()))


def save(figure, out_dir, stem, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = out_dir / f"{stem}.{suffix}"
        figure.savefig(path)
        print(f"wrote {path}")
    plt.close(figure)


def plot_reach(trajectory, verdicts, size, variant, out_dir, formats):
    series = group_by_seed(trajectory, size, variant)
    if not series:
        raise SystemExit(f"no trajectory rows for size={size} variant={variant}")
    colors = color_map(trajectory)

    solved = {
        int(row["seed"]): row["trials"]
        for row in verdicts
        if int(row["size"]) == size and row["variant"] == variant and row["verdict"] == "SUCCESS"
    }

    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    endpoints = []
    truncated = False
    for seed, points in series.items():
        trials = [row["trials"] for row in points]
        knowledge = [row["knowledge"] for row in points]
        axes.plot(trials, knowledge, color=colors[seed], linewidth=LINE_WIDTH,
                  label=f"seed {seed}", solid_capstyle="round")
        if seed in solved:
            axes.plot([solved[seed]], [1.0], marker="o", markersize=4.5,
                      color=colors[seed], markeredgecolor="white", markeredgewidth=0.8)
        else:
            truncated = True
            axes.plot([trials[-1]], [knowledge[-1]], marker="o", markersize=4.5,
                      markerfacecolor="white", markeredgecolor=colors[seed], markeredgewidth=1.2)
        endpoints.append({"seed": seed, "x": trials[-1], "y": knowledge[-1]})

    x_span = max(point["x"] for point in endpoints) or 1
    label_y = spread(endpoints, min_gap=0.06, x_tolerance=0.08 * x_span, ceiling=1.06)
    for label, y in zip(endpoints, label_y):
        axes.annotate(f"seed {label['seed']}", xy=(label["x"], y),
                      xytext=(8, 0), textcoords="offset points",
                      color=INK_SECONDARY, fontsize=7.5, va="center")

    axes.axhline(1.0, color=INK_MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=0)
    axes.set_xlabel("explore trials")
    axes.set_ylabel("knowledge")
    axes.set_ylim(0, 1.08)
    axes.set_xlim(left=0)
    axes.xaxis.set_major_formatter(FuncFormatter(millions))
    axes.set_title(f"MPX-{size} knowledge acquisition ({variant} variant, GA on)", loc="left", pad=10)
    axes.legend(loc="center right", ncol=1)
    if truncated:
        figure.text(0.0, -0.04,
                    "Filled marker: knowledge reached 1.0.  Hollow marker: run stopped at its "
                    "wall-clock cap, not a failure to converge.",
                    color=INK_MUTED, fontsize=7)

    save(figure, out_dir, f"mpx{size}_reach_{variant}", formats)


def plot_anatomy(trajectory, size, variant, seed, out_dir, formats):
    series = group_by_seed(trajectory, size, variant)
    if seed not in series:
        raise SystemExit(f"no trajectory for seed {seed} at size={size} variant={variant}")
    points = series[seed]
    color = color_map(trajectory)[seed]
    ideal = address_bits(size) + 1

    trials = [row["trials"] for row in points]
    panels = [
        ("knowledge", [row["knowledge"] for row in points], None),
        ("mean specificity\nof reliable rules", [row["spec"] for row in points], ideal),
        ("population size", [row["pop"] for row in points], None),
    ]

    figure, axes_list = plt.subplots(3, 1, figsize=(6.4, 6.2), sharex=True)
    for axes, (label, values, reference) in zip(axes_list, panels):
        axes.plot(trials, values, color=color, linewidth=LINE_WIDTH, solid_capstyle="round")
        axes.set_ylabel(label)
        if reference is not None:
            axes.axhline(reference, color=INK_MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=0)
            axes.annotate(f"ideal {reference}", xy=(0.995, reference), xycoords=("axes fraction", "data"),
                          xytext=(0, 4), textcoords="offset points",
                          color=INK_SECONDARY, fontsize=7.5, ha="right")

    axes_list[0].set_ylim(0, 1.08)
    axes_list[0].set_title(
        f"MPX-{size} seed {seed}: knowledge, rule specificity and population size",
        loc="left", pad=10,
    )
    axes_list[-1].set_xlabel("explore trials")
    axes_list[-1].set_xlim(left=0)
    axes_list[-1].xaxis.set_major_formatter(FuncFormatter(millions))

    save(figure, out_dir, f"mpx{size}_anatomy_s{seed}_{variant}", formats)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectory-csv", type=Path, default=Path("reports/mpx_trajectory.csv"))
    parser.add_argument("--verdict-csv", type=Path, default=Path("reports/mpx_verdicts.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--size", type=int, default=70)
    parser.add_argument("--variant", default="pyalcs")
    parser.add_argument("--anatomy-seed", type=int, default=42)
    parser.add_argument("--formats", default="pdf,png", help="comma-separated: pdf, png, pgf, svg")
    parser.add_argument("--figures", default="reach,anatomy")
    args = parser.parse_args()

    trajectory = numeric(load(args.trajectory_csv), "trials", "wall_s", "knowledge", "reliable", "spec", "pop")
    verdicts = numeric(load(args.verdict_csv), "trials", "knowledge", "reliable", "spec", "wall_s")
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    figures = {item.strip() for item in args.figures.split(",") if item.strip()}

    apply_style()
    if "reach" in figures:
        plot_reach(trajectory, verdicts, args.size, args.variant, args.out_dir, formats)
    if "anatomy" in figures:
        plot_anatomy(trajectory, args.size, args.variant, args.anatomy_seed, args.out_dir, formats)


if __name__ == "__main__":
    main()
