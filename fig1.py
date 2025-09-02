import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def _koch(o, scale):
    """
    Return the points of the Koch snowflake of order `o` with side length 1.
    The result is a complex128 NumPy array of shape (4 * 3**(o - 1),).
    """
    if o == 0:
        # base case: equilateral triangle
        ang = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3, 0.0])
        return scale * np.exp(1j * ang)

    # recursion
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
    """
    Return the points of the Koch snowflake of order `order` with side length `scale`.

    Parameters
    ----------
    order : int
        The order of the Koch snowflake.
    scale : float, optional
        The side length of the Koch snowflake. Default is 1.0.

    Returns
    -------
    x : ndarray
        The x-coordinates of the Koch snowflake.
    y : ndarray
        The y-coordinates of the Koch snowflake.
    """
    pts = _koch(order, scale)
    return pts.real, pts.imag


fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[8, 7], hspace=0.45)

orders = [0, 1, 2]
titles = ["Start: Equilateral triangle", "Step 1: Add bumps", "Step 2: More bumps"]

for idx, order in enumerate(orders):
    ax = fig.add_subplot(gs[0, idx])
    x, y = koch_snowflake(order, scale=1)
    ax.plot(x, y, lw=2, color="royalblue")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titles[idx], fontsize=13, pad=8)
    ax.margins(0.08)

    if order == 1:
        outline = [pe.withStroke(linewidth=3, foreground="white")]
        ax.annotate(
            "Each segment → 4 pieces",
            xy=(0.5, 0.07),
            xycoords="axes fraction",
            xytext=(0.5, -0.18),
            textcoords="axes fraction",
            ha="center",
            fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=2),
            clip_on=False,
            path_effects=outline,
        )
        ax.text(
            0.5,
            -0.28,
            r"$N = 4,\ r = 1/3$",
            ha="center",
            va="top",
            fontsize=12,
            color="crimson",
            transform=ax.transAxes,
            clip_on=False,
            path_effects=outline,
        )

ax4 = fig.add_subplot(gs[1, :])
scales = np.array([1, 1 / 3, 1 / 9, 1 / 27])
N = 4 ** np.arange(0, 4)
ax4.plot(scales, N, "o-", color="crimson")
for r, n in zip(scales, N):
    ax4.annotate(
        f"{int(n)} pieces",
        xy=(r, n),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        clip_on=False,
    )

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("How much each piece is shrunk (scaling factor)")
ax4.set_ylabel("Number of self-similar pieces")
ax4.set_title("Fractal dimension: Number of pieces vs. scaling", fontsize=13)

outline = [pe.withStroke(linewidth=3, foreground="white")]
ax4.text(
    0.18,
    15,
    r"$D=\frac{\log 4}{\log 3}\approx 1.26$",
    fontsize=14,
    color="crimson",
    path_effects=outline,
)
ax4.text(
    0.33,
    15,
    "(Dimension is not a whole number!)",
    fontsize=12,
    color="crimson",
    path_effects=outline,
)

ax4.grid(True, which="both", ls="--", alpha=0.5)
ax4.margins(x=0.08, y=0.15)
fig.suptitle("Fig. 1 — Koch snowflake: A classic fractal", fontsize=18, y=0.98)

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig1_koch.png", dpi=300, bbox_inches="tight")
plt.show()
