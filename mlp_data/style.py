"""One consistent, colourblind-safe look for every chart in the course."""
from __future__ import annotations

# Okabe-Ito palette: distinguishable under the common forms of colour blindness.
BLUE, ORANGE, GREEN, RED, SKY, PURPLE, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#6B7280")


def apply() -> None:
    """Set matplotlib defaults. Safe to call more than once."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update({
        "figure.figsize": (8, 3.8), "figure.dpi": 110, "savefig.dpi": 110,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY]),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.axisbelow": True,
    })
