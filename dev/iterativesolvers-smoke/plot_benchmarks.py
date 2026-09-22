"""Plot saved CG measurements; run with Python + matplotlib."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
rows = []
with_layout = "--layout" in sys.argv
sources = ["benchmark-results.csv", "benchmark-precision-results.csv"]
if with_layout:
    sources.append("benchmark-layout-results.csv")
for name in sources:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row.setdefault("eltype", "Float64")
            rows.append(row)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
fig.subplots_adjust(left=0.075, right=0.97, top=0.75, bottom=0.27, wspace=0.16)
fig.suptitle("Single-GPU CG solve time versus matrix dimension", fontsize=19, y=0.97, weight="bold")
fig.text(0.5, 0.91, "NVIDIA A30X  ·  Dense N × N matrices  ·  Median of 5 warmed solves", ha="center", color="#4b5563")

styles = {"cuNumeric": ("#1664c0", "o", "cuNumeric"),
          "DaggerPatched": ("#d77b0b", "s", "Dagger (matvec workaround)"),
          "CuArray": ("#159276", "^", "CUDA / CuArray")}
if with_layout:
    fig.subplots_adjust(top=0.70)
    styles["cuNumericTransposed"] = ("#8b3fb0", "D", "cuNumeric (transposed storage)")
for ax, dtype, title, ticks, oom in zip(
    axes, ("Float64", "Float32"),
    ("FP64 · rtol = 10⁻⁸ · 18–19 iterations", "FP32 · rtol = 10⁻⁵ · 11–12 iterations"),
    ([256, 1024, 4096, 8192, 16384, 32768, 49152], [8192, 16384, 32768, 49152, 65536]),
    (49152, 65536),
):
    for backend, (color, marker, label) in styles.items():
        series = sorted((r for r in rows if r["eltype"] == dtype and r["backend"] == backend), key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in series]
        ys = [float(r["median_ms"]) for r in series]
        lower = [y - float(r["min_ms"]) for y, r in zip(ys, series)]
        upper = [float(r["max_ms"]) - y for y, r in zip(ys, series)]
        ax.errorbar(ns, ys, yerr=[lower, upper], color=color, marker=marker,
                    markersize=6, linewidth=2, elinewidth=1.3, capsize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1.7, 2300)
    ax.set_xlim(ticks[0] / 1.2, ticks[-1] * 1.18)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.minorticks_off()
    ax.tick_params(axis="x", rotation=45)
    ax.grid(which="major", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, pad=15, weight="bold")
    ax.set_xlabel("Matrix dimension N (log scale)", labelpad=10)
    ax.text(0.04, 0.95, f"Dagger allocation failed at N = {oom:,}",
            transform=ax.transAxes, va="top", fontsize=9, color="#9c5706")

axes[0].set_ylabel("CG solve time (ms, log scale)", labelpad=10)
axes[1].annotate("cuNumeric: 1,410 ms", xy=(65536, 1409.988826),
                 xytext=(21000, 850), fontsize=10, color=styles["cuNumeric"][0],
                 arrowprops={"arrowstyle": "->", "color": styles["cuNumeric"][0]})
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=2 if with_layout else 3, frameon=False)
fig.text(0.075, 0.105, "Whiskers show sample min–max (not confidence intervals). Setup and transfers excluded; solve completion synchronized.", fontsize=10, color="#4b5563")
fig.text(0.075, 0.075, "Compare backends within each precision: tolerances and iteration counts differ. All successful solves passed residual checks.", fontsize=10, color="#4b5563")
fig.text(0.075, 0.045, "GPU memory allowance increased for larger cases; full settings and raw samples are recorded alongside this figure.", fontsize=10, color="#4b5563")
stem = "cg-scaling-layout" if with_layout else "cg-scaling"
for suffix in ("png", "svg", "pdf"):
    fig.savefig(HERE / f"{stem}.{suffix}", dpi=180, facecolor="white")
plt.close(fig)
