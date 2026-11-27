from pathlib import Path
import argparse
import csv
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"bert": "BERT XXL", "electra": "ELECTRA XXL", "geppetto": "GePpeTto"}
VARIANTS = {"original": "Originale", "lowercase": "Minuscolo"}
SEEDS = {"bert": (42, 123, 456), "electra": (42, 123, 456), "geppetto": (42, 123, 456)}
METRICS = {"upos_accuracy": "Accuracy UPOS", "upos_macro_f1_present": "Macro-F1 UPOS - classi presenti"}
COLORS = {"original": "#246b8e", "lowercase": "#be763a"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9c2c8", "text.color": "#24343d",
    "axes.labelcolor": "#24343d", "xtick.color": "#53626a",
    "ytick.color": "#24343d", "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def read_results(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    runs = {}
    for row in rows:
        key = row["model"], row["variant"], int(row["seed"])
        if key in runs:
            raise ValueError(f"Esecuzione duplicata: {key}")
        scores = {metric: float(row[metric]) for metric in METRICS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError(f"Metriche non valide per {key}: attesi valori tra 0 e 1")
        runs[key] = scores
    expected = {(model, variant, seed) for model in MODELS for variant in VARIANTS for seed in SEEDS[model]}
    if set(runs) != expected:
        raise ValueError(
            f"Servono tutti i {len(expected)} run: {len(expected - runs.keys())} mancanti, "
            f"{len(runs.keys() - expected)} non previsti. Completare il piano e il riepilogo nel notebook."
        )
    return runs


def configure_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(MODELS)), MODELS.values())
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5eaed", linewidth=0.8)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scores(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, (metric, title) in zip(axes, METRICS.items()):
        limits = []
        for variant_index, (variant, label) in enumerate(VARIANTS.items()):
            values = [[100 * runs[model, variant, seed][metric] for seed in SEEDS[model]] for model in MODELS]
            means = [statistics.mean(group) for group in values]
            deviations = [statistics.stdev(group) if len(group) > 1 else None for group in values]
            positions = [index + (variant_index - 0.5) * 0.36 for index in range(len(MODELS))]
            ax.bar(positions, means, width=0.34, color=COLORS[variant], label=label)
            for position, mean, deviation in zip(positions, means, deviations):
                if deviation is not None:
                    ax.errorbar(position, mean, yerr=deviation, fmt="none", capsize=4,
                                ecolor="#34434b", elinewidth=1.0)
                    limits.append((mean - deviation, mean + deviation))
                else:
                    limits.append((mean, mean))
        low = min(pair[0] for pair in limits)
        high = max(pair[1] for pair in limits)
        span = max(high - low, 0.1)
        ax.set_ylim(low - 0.2 * span, high + 0.25 * span)
        configure_axis(ax, title, "Punteggio (%)")
        # Mark the truncated baseline explicitly when magnifying small differences.
        if ax.get_ylim()[0] > 0:
            for shift in (-0.012, 0.012):
                ax.plot([-0.012, 0.012], [shift - 0.008, shift + 0.008],
                        transform=ax.transAxes, color="#53626a", clip_on=False, linewidth=1.1)
            ax.set_xlabel("Asse verticale troncato", fontsize=8, color="#53626a", labelpad=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.24), ncol=2)
    fig.suptitle("Media e deviazione standard campionaria su 3 seed per ogni modello", fontsize=10)
    save(fig, output, "upos_comparison")


def plot_ablation(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    colors = ("#246b8e", "#81919d", "#be763a")
    for ax, (metric, title) in zip(axes, METRICS.items()):
        # Pair the same seed before averaging: std is measured on the differences.
        changes = [[100 * (runs[model, "lowercase", seed][metric] - runs[model, "original", seed][metric])
                    for seed in SEEDS[model]] for model in MODELS]
        means = [statistics.mean(group) for group in changes]
        deviations = [statistics.stdev(group) if len(group) > 1 else None for group in changes]
        ax.bar(range(len(MODELS)), means, width=0.56, color=colors)
        limits = []
        for position, (mean, deviation) in enumerate(zip(means, deviations)):
            if deviation is not None:
                ax.errorbar(position, mean, yerr=deviation, fmt="none", capsize=4,
                            ecolor="#34434b", elinewidth=1.0)
                limits.append((mean - deviation, mean + deviation))
            else:
                limits.append((mean, mean))
        low = min(0, min(pair[0] for pair in limits))
        high = max(0, max(pair[1] for pair in limits))
        span = max(high - low, 0.1)
        ax.set_ylim(low - 0.18 * span, high + 0.18 * span)
        ax.axhline(0, color="#81919d", linewidth=0.9)
        configure_axis(ax, title, "Differenza (punti percentuali)")
    fig.suptitle("Ablation: minuscolo meno originale\n"
                 "Media e deviazione standard degli scarti appaiati su 3 seed per ogni modello",
                 fontsize=10)
    save(fig, output, "upos_casing_ablation")


def main():
    parser = argparse.ArgumentParser(description="Grafici UPOS dai risultati salvati, senza training o inferenza.")
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/upos_trainer_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/report/figures")
    args = parser.parse_args()
    try:
        runs = read_results(args.runs / "results.csv")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Risultati non disponibili o incompleti: {error}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    plot_scores(runs, args.out)
    plot_ablation(runs, args.out)
    print(f"Salvate due figure PDF e PNG in {args.out}")


if __name__ == "__main__":
    main()
