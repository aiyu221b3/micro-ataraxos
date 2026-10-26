

import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------
# Colour palette
# -----------------------

COLORS = {
    "purple": "#8568B3",
    "violet": "#9B81C6",
    "cyan": "#55AEB8",
    "cyan_light": "#76BEC6",
    "pink": "#C77FA2",
    "pink_light": "#D99AB6",
    "ink": "#35313B",
    "muted": "#67616F",
    "grid": "#DDD6E9",
    "spine": "#756D7A",
    "baseline": "#958D9D",
}

# -----------------------
# Font setup
# -----------------------

FONT_FAMILY = "DejaVu Sans"


def setup_plot_style(output_root="."):
    """Apply the paper plotting style globally."""

    global FONT_FAMILY

    output_root = Path(output_root)
    font_dir = output_root / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "InterVariable.ttf"

    try:
        if not font_path.exists():
            url = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
            with urllib.request.urlopen(url, timeout=20) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))

            member = next(
                name for name in archive.namelist()
                if Path(name).name == "InterVariable.ttf"
            )

            font_path.write_bytes(archive.read(member))

        font_manager.fontManager.addfont(str(font_path))
        FONT_FAMILY = FontProperties(fname=str(font_path)).get_name()

    except Exception:
        FONT_FAMILY = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["spine"],
        "axes.linewidth": 0.75,
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.grid": False,
        "grid.color": COLORS["grid"],
        "grid.linestyle": ":",
        "grid.linewidth": 0.75,
        "grid.alpha": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


# -----------------------
# Axis styling
# -----------------------

def budget_tick(value, position):
    value = int(round(value))
    return f"{value/1000:g}k" if value >= 1000 else f"{value:,}"


def style_axis(ax, budgets=None, show_xlabel=True):
    """Apply the Decoy axis styling."""

    if budgets is not None:
        ax.set_xscale("log", base=10)
        ax.set_xticks(budgets)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(budget_tick))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ax.set_axisbelow(True)

    ax.grid(axis="y", linestyle=":", color=COLORS["grid"], linewidth=0.75)
    ax.grid(axis="x", linestyle=":", color=COLORS["grid"], linewidth=0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])

    ax.tick_params(axis="both", length=3, width=0.7, direction="out")

    if show_xlabel:
        ax.set_xlabel("Training budget (environment steps)")

    ax.margins(x=0.08)


def add_panel_label(ax, label):
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


# -----------------------
# Legend helpers
# -----------------------

def line_handle(color, label):
    return Line2D(
        [0], [0],
        color=color,
        marker="o",
        markerfacecolor=color,
        markeredgecolor="none",
        linewidth=1.4,
        markersize=4.5,
        label=label,
    )


def ci_handle():
    return Patch(
        facecolor=COLORS["ink"],
        edgecolor="none",
        alpha=0.16,
        label="Pointwise 95% bootstrap CI",
    )


# -----------------------
# Save helper
# -----------------------

def save_figure(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    return path
