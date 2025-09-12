import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter


def _koch(o, scale):
    if o == 0:
        ang = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0])
        return scale * np.exp(1j * ang)
    Z = _koch(o - 1, scale)
    out = [Z[0]]
    rot = np.exp(-1j * np.pi / 3)
    for i in range(len(Z) - 1):
        z0, z1 = Z[i], Z[i + 1]
        dz = z1 - z0
        p1 = z0 + dz / 3
        p2 = p1 + (dz / 3) * rot
        p3 = z0 + 2 * dz / 3
        out.extend([p1, p2, p3, z1])
    return np.array(out, dtype=complex)


def koch_snowflake(order, scale=1.0):
    pts = _koch(order, scale)
    return pts.real, pts.imag


def resample_polyline(x, y, step):
    z = x + 1j * y
    out = [z[0]]
    for a, b in zip(z[:-1], z[1:]):
        L = np.abs(b - a)
        n = max(1, int(np.ceil(L / step)))
        seg = np.linspace(a, b, n + 1)[1:]
        out.extend(seg)
    out = np.array(out)
    return out.real, out.imag


def normalize_to_unit_square(x, y, pad=0.05):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w, h = xmax - xmin, ymax - ymin
    s = max(w, h)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    x2 = (x - cx) / s + 0.5
    y2 = (y - cy) / s + 0.5
    x3 = pad + (1 - 2 * pad) * x2
    y3 = pad + (1 - 2 * pad) * y2
    return x3, y3


def box_count(points_xy, eps, bounds=(0, 1, 0, 1)):
    xmin, xmax, ymin, ymax = bounds
    x, y = points_xy[:, 0], points_xy[:, 1]
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    x, y = x[mask], y[mask]
    i = np.floor((x - xmin) / eps).astype(int)
    j = np.floor((y - ymin) / eps).astype(int)
    nx = int(np.ceil((xmax - xmin) / eps))
    ny = int(np.ceil((ymax - ymin) / eps))
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)
    return np.unique(np.stack([i, j], axis=1), axis=0).shape[0]


def grid_boxes_intersected(points_xy, eps, bounds=(0, 1, 0, 1)):
    xmin, xmax, ymin, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / eps))
    ny = int(np.ceil((ymax - ymin) / eps))
    x, y = points_xy[:, 0], points_xy[:, 1]
    i = np.floor((x - xmin) / eps).astype(int)
    j = np.floor((y - ymin) / eps).astype(int)
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)
    unique = np.unique(np.stack([i, j], axis=1), axis=0)
    rects = [
        Rectangle((xmin + ii * eps, ymin + jj * eps), eps, eps) for ii, jj in unique
    ]
    return rects


def counts_for(points_xy, eps_vals):
    return np.array([box_count(points_xy, eps) for eps in eps_vals], dtype=int)


def slope_and_intercept(eps, N, fit_slice=None):
    x = np.log(1 / eps)
    y = np.log(N)
    if fit_slice is not None:
        x = x[fit_slice]
        y = y[fit_slice]
    b, a = np.polyfit(x, y, 1)
    return b, a


eps_vals = 1 / np.array(
    [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192], dtype=float
)
fit_slice = slice(2, -2)
eps_min = float(eps_vals.min())

n_circle = int(np.ceil(2000 * (0.0104 / eps_min)))
theta = np.linspace(0, 2 * np.pi, max(n_circle, 5000), endpoint=False)
circle_xy = np.column_stack([0.5 + 0.40 * np.cos(theta), 0.5 + 0.40 * np.sin(theta)])

grid_step = eps_min / 3.0
g = np.arange(0.2, 0.8 + 1e-12, grid_step)
xx, yy = np.meshgrid(g, g, indexing="xy")
square_xy = np.column_stack([xx.ravel(), yy.ravel()])

xs, ys = koch_snowflake(order=5, scale=1.0)
xs, ys = normalize_to_unit_square(xs, ys, pad=0.08)
xs, ys = resample_polyline(xs, ys, step=0.002)
snowflake_xy = np.column_stack([xs, ys])

N_circle = counts_for(circle_xy, eps_vals)
N_square = counts_for(square_xy, eps_vals)
N_snow = counts_for(snowflake_xy, eps_vals)

D_circle, _a = slope_and_intercept(eps_vals, N_circle, fit_slice)
D_square, _b = slope_and_intercept(eps_vals, N_square, fit_slice)
D_snow, _c = slope_and_intercept(eps_vals, N_snow, fit_slice)

os.makedirs("images", exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[8, 7], hspace=0.45)

titles = ["Smooth curve (circle)", "Filled square", "Koch snowflake"]
subtitles = [
    "A smooth 1-D boundary",
    "A solid 2-D area",
    "Woah! A wild fractal appears!"
]
datasets = [
    (circle_xy, N_circle, D_circle),
    (square_xy, N_square, D_square),
    (snowflake_xy, N_snow, D_snow),
]

eps_show = eps_vals[5]
outline_fx = [pe.withStroke(linewidth=3, foreground="white")]

for j, (pts, Nvals, Dhat) in enumerate(datasets):
    ax = fig.add_subplot(gs[0, j])
    ax.set_title(titles[j], fontsize=13, pad=8)
    for v in np.arange(0, 1 + 1e-9, eps_show):
        ax.plot([v, v], [0, 1], lw=0.6, color="0.85", zorder=0)
        ax.plot([0, 1], [v, v], lw=0.6, color="0.85", zorder=0)
    rects = grid_boxes_intersected(pts, eps_show)
    pc = PatchCollection(
        rects, facecolor="tab:orange", alpha=0.3, edgecolor="none", zorder=1
    )
    ax.add_collection(pc)
    ax.plot(pts[:, 0], pts[:, 1], lw=2.25, color="royalblue", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.5,
        -0.10,
        subtitles[j],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="0.35",
    )
    ax.text(
        0.02,
        0.97,
        f"box size ε = {eps_show:.3f}\nboxes used N(ε) = {box_count(pts, eps_show)}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        path_effects=outline_fx,
    )
    ax.margins(0.08)

ax = fig.add_subplot(gs[1, :])
x_all = np.log(1 / eps_vals)


def plot_series(N, label):
    y_all = np.log(N)
    x = x_all[fit_slice]
    y = y_all[fit_slice]
    b, a = np.polyfit(x, y, 1)
    ax.plot(1 / eps_vals, N, "o", label=f"{label}: data")
    xx = np.linspace((1 / eps_vals).min() * 0.9, (1 / eps_vals).max() * 1.1, 200)
    ax.plot(xx, np.exp(a + b * np.log(xx)), "-", label=f"{label}: slope ≈ {b:.2f}")
    return b


Dc = plot_series(N_circle, "Circle")
Ds = plot_series(N_square, "Square")
Df = plot_series(N_snow, "Snowflake")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("How many boxes fit across?  (bigger → smaller boxes)")
ax.set_ylabel("How many boxes are needed?  N(ε)")
ax.set_title(
    "Box-counting: the line's slope ≈ the 'dimension' of the shape", fontsize=13
)

ax.text(
    0.03,
    0.95,
    "On a log-log plot,\nstraight lines mean 'power-law' growth.\nThe slope of each line ≈ dimension:",
    transform=ax.transAxes,
    va="top",
    ha="left",
)
ax.annotate(
    "",
    xy=(0.43, 0.54),
    xytext=(0.32, 0.74),
    xycoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.5),
)

ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
ax.grid(True, which="both", ls="--", alpha=0.45)


def _format_eps_from_invx(invx):
    if invx == 0:
        return ""
    eps = 1.0 / invx
    if eps >= 0.1:
        return f"{eps:.2f}"
    elif eps >= 0.01:
        return f"{eps:.3f}"
    else:
        return f"{eps:.1e}"


ax.set_xticks(1 / eps_vals, minor=False)

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 3))
ax.xaxis.set_major_formatter(formatter)
ax.tick_params(axis="x", labelsize=9)

ax_top = ax.secondary_xaxis("top", functions=(lambda u: u, lambda u: u))
ax_top.set_xscale("log")
ax_top.set_xlabel("Actual box size ε (smaller to the right)", fontsize=11)
ax_top.set_xticks(1 / eps_vals, minor=False)
ax_top.set_xticklabels([_format_eps_from_invx(v) for v in (1 / eps_vals)])
ax_top.tick_params(axis="x", labelsize=6)

fig.suptitle(
    "Fig. 2 — Box-counting dimension (Minkowski-Bouligand)", fontsize=18, y=0.98
)
fig.tight_layout()

out_path = "images/fig2_box_counting.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Estimated D_box (circle)   ≈ {D_circle:.3f}  (theory: 1)")
print(f"Estimated D_box (square)   ≈ {D_square:.3f}  (theory: 2)")
print(
    f"Estimated D_box (snowflake)≈ {D_snow:.3f}  (theory: log 4 / log 3 ≈ {np.log(4)/np.log(3):.3f})"
)
print(f"\nSaved figure → {out_path}")
