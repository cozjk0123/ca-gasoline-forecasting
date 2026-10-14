"""Shared chart style so every figure in the notebooks, report and slides looks the same."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

# Categorical slots in fixed order (validated palette); colour follows the entity.
SERIES = {
    "ca_gas": "#2a78d6",     # California gasoline
    "us_gas": "#eb6834",     # U.S. gasoline
    "brent": "#1baf7a",      # Brent crude
}
MODEL_COLORS = {
    "naive": "#8a8984",
    "seasonal_naive": "#b5b4ae",
    "ets": "#e87ba4",
    "arima_fourier": "#6250d6",
    "arimax": "#2a78d6",
    "chronos_bolt": "#eb6834",
    "moirai2": "#1baf7a",
    "llm_anon": "#eda100",
    "llm_context": "#e34948",
    "combo": "#008300",
}
MODEL_LABELS = {
    "naive": "Naive (random walk)",
    "seasonal_naive": "Seasonal naive",
    "ets": "ETS",
    "arima_fourier": "ARIMA + Fourier",
    "arimax": "ARIMAX (Brent + events)",
    "chronos_bolt": "Chronos-Bolt (zero-shot)",
    "moirai2": "Moirai 2.0 (zero-shot)",
    "llm_anon": "Claude, anonymized",
    "llm_context": "Claude, full context",
    "combo": "Combination",
}
TEST_COLOR = "#f1f0ec"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def shade_test(ax, start, end):
    ax.axvspan(start, end, color=TEST_COLOR, zorder=0, lw=0)
    ax.text(start, 1.0, " test period", transform=ax.get_xaxis_transform(), va="top", ha="left",
            fontsize=8, color=INK2)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
