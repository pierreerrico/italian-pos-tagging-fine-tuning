from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"bert": "BERT XXL", "electra": "ELECTRA XXL", "geppetto": "GePpeTto"}
SEEDS = (42, 123, 456)
UPOS_METRICS = ("upos_accuracy", "upos_macro_f1_all", "upos_macro_f1_present")
FEATS_METRICS = ("feats_exact_accuracy", "feats_attribute_accuracy", "feats_macro_f1_all",
                 "feats_macro_f1_present", "joint_accuracy")
SCORE_PANELS = {"upos_accuracy": "Accuracy UPOS", "upos_macro_f1_present": "Macro-F1 UPOS - classi presenti"}
MODEL_COLORS = {"bert": "#246b8e", "electra": "#81919d", "geppetto": "#be763a"}
TASK_COLORS = {"upos": "#246b8e", "upos_feats": "#be763a"}
INK, GRID = "#24343d", "#e5eaed"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9c2c8", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#53626a", "ytick.color": INK, "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(upos_root, feats_root):
    baseline = read_json(upos_root / "experiment.json")
    joint = read_json(feats_root / "experiment.json")
    if joint.get("schema") != "upos_feats_pretrained_mean_pooling_v1" or joint.get("initialization") != "shared_pretrained_upos_init_v1":
        raise ValueError("Servono i run FEATS controllati dal pretrained, non la continuazione da UPOS")
    if set(baseline["variants"]) != {"original", "lowercase"} or joint["variants"] != ["original"]:
        raise ValueError("Attese entrambe le varianti UPOS e soltanto l'originale per FEATS")
    for settings in (baseline, joint):
        if settings["selection_metric"] != "upos_macro_f1_all" or tuple(settings["seeds"]) != SEEDS:
            raise ValueError("I due obiettivi devono condividere selezione UPOS e tre seed")
    for key in ("dataset", "data_manifest", "models", "pooling"):
        if baseline[key] != joint[key]:
            raise ValueError(f"Protocollo UPOS/FEATS non appaiato: {key}")
    return baseline, joint


def read_results(directory, variants, metrics):
    runs = {}
    for row in read_csv(directory / "results.csv"):
        key = row["model"], row["variant"], int(row["seed"])
        if key in runs:
            raise ValueError(f"Esecuzione duplicata: {key}")
        scores = {metric: float(row[metric]) for metric in metrics}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError(f"Metriche fuori [0, 1]: {key}")
        run = directory / key[0] / key[1] / f"seed_{key[2]}"
        complete = read_json(run / "complete.json")
        for name, field in (("experiment.json", "experiment_sha256"), ("test_results.json", "test_results_sha256")):
            if digest(run / name) != complete[field]:
                raise ValueError(f"Checksum non valido: {run / name}")
        saved = read_json(run / "test_results.json")
        if any(not math.isclose(value, float(saved[metric]), rel_tol=1e-11, abs_tol=1e-12)
               for metric, value in scores.items()):
            raise ValueError(f"Riepilogo diverso dalle metriche del run: {key}")
        info = read_json(run / "training_info.json")
        if info["seed"] != key[2] or info["system"] != f"{key[0]}_{key[1]}" or info["selection_metric"] != "upos_macro_f1_all":
            raise ValueError(f"Identità o selezione incompatibile: {key}")
        runs[key] = scores
    expected = {(model, variant, seed) for model in MODELS for variant in variants for seed in SEEDS}
    if set(runs) != expected:
        raise ValueError(f"Servono {len(expected)} run completi in {directory}: "
                         f"{len(expected - runs.keys())} mancanti, {len(runs.keys() - expected)} non previsti")
    return runs


def decimal(value, digits=2, signed=False):
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",")


def axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(MODELS)), MODELS.values(), fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def bars(ax, groups, title, delta=False):
    means = [statistics.mean(group) for group in groups]
    deviations = [statistics.stdev(group) for group in groups]
    positions = np.arange(len(MODELS))
    ax.bar(positions, means, width=0.56, color=list(MODEL_COLORS.values()),
           yerr=deviations, capsize=4, error_kw={"ecolor": "#34434b", "elinewidth": 1.1}, zorder=2)
    lows = [mean - sd for mean, sd in zip(means, deviations)]
    highs = [mean + sd for mean, sd in zip(means, deviations)]
    low, high = min(lows), max(highs)
    if delta:
        low, high = min(0.0, low), max(0.0, high)
    span = max(high - low, 0.08)
    ax.set_ylim(low - 0.18 * span, high + 0.32 * span)
    for position, mean, bottom, top in zip(positions, means, lows, highs):
        label_y = bottom - 0.045 * span if delta and mean < 0 else top + 0.045 * span
        ax.text(position, label_y, decimal(mean, signed=delta), ha="center",
                va="top" if delta and mean < 0 else "bottom", fontsize=10, fontweight="bold")
    axis(ax, title, "Differenza (punti percentuali)" if delta else "Punteggio (%)")
    if delta:
        ax.axhline(0, color="#81919d", linewidth=1.0, zorder=1)
    elif ax.get_ylim()[0] > 0:
        for shift in (-0.012, 0.012):
            ax.plot([-0.012, 0.012], [shift - 0.008, shift + 0.008], transform=ax.transAxes,
                    color="#53626a", clip_on=False, linewidth=1.1)
        ax.set_xlabel("Asse verticale troncato", fontsize=9, color="#53626a", labelpad=10)


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scores(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for ax, (metric, title) in zip(axes, SCORE_PANELS.items()):
        groups = [[100 * runs[model, "original", seed][metric] for seed in SEEDS] for model in MODELS]
        bars(ax, groups, title)
    fig.suptitle("UPOS - testo originale", fontsize=12)
    fig.supxlabel("Colonne: media; barre di errore: deviazione standard campionaria su tre seed", fontsize=9)
    save(fig, output, "upos_comparison")


def plot_delta(baseline, compared, output, name, title, compared_variant):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for ax, (metric, label) in zip(axes, SCORE_PANELS.items()):
        groups = [[100 * (compared[model, compared_variant, seed][metric] - baseline[model, "original", seed][metric])
                   for seed in SEEDS] for model in MODELS]
        bars(ax, groups, label, delta=True)
    fig.suptitle(title, fontsize=12)
    fig.supxlabel("Media e deviazione standard campionaria degli scarti appaiati su tre seed", fontsize=9)
    save(fig, output, name)


def plot_feats(runs, output):
    panels = {"feats_exact_accuracy": "Accuracy esatta FEATS", "feats_macro_f1_present": "Macro-F1 FEATS - presenti",
              "joint_accuracy": "Accuracy congiunta"}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), layout="constrained")
    for ax, (metric, title) in zip(axes, panels.items()):
        groups = [[100 * runs[model, "original", seed][metric] for seed in SEEDS] for model in MODELS]
        bars(ax, groups, title)
    fig.suptitle("UPOS + FEATS - testo originale", fontsize=12)
    fig.supxlabel("Media ± SD campionaria su tre seed. Macro-F1: coppie presenti, senza <NONE>.", fontsize=9)
    save(fig, output, "feats_comparison")


def attribute_values(root, runs, settings):
    known = list(settings["morphology"]["head_labels"])
    reference, tables = None, {}
    for model in MODELS:
        for seed in SEEDS:
            path = root / model / "original" / f"seed_{seed}" / "per_feature_metrics.csv"
            rows = read_csv(path)
            support = {row["feature"]: int(row["support"]) for row in rows}
            if len(support) != len(rows) or any("<NONE>" in pair for pair in support):
                raise ValueError(f"Inventario delle coppie non valido: {path}")
            if reference is None:
                reference = support
            elif support != reference:
                raise ValueError(f"Supporti gold diversi: {path}")
            present = [float(row["f1"]) for row in rows if int(row["support"]) > 0]
            if not math.isclose(statistics.mean(present), runs[model, "original", seed]["feats_macro_f1_present"], abs_tol=1e-12):
                raise ValueError(f"Macro-F1 diversa dalla tabella per coppia: {path}")
            per_attribute = defaultdict(list)
            for row in rows:
                if int(row["support"]) > 0:
                    per_attribute[row["feature"].split("=", 1)[0]].append(float(row["f1"]))
            tables[model, seed] = {key: 100 * statistics.mean(values) for key, values in per_attribute.items()}
    attributes = known + sorted({pair.split("=", 1)[0] for pair in reference} - set(known))
    supports = [sum(value for pair, value in reference.items() if pair.split("=", 1)[0] == key) for key in attributes]
    means = np.array([[statistics.mean(tables[model, seed][key] for seed in SEEDS) if support else math.nan
                       for model in MODELS] for key, support in zip(attributes, supports)])
    deviations = np.array([[statistics.stdev(tables[model, seed][key] for seed in SEEDS) if support else math.nan
                            for model in MODELS] for key, support in zip(attributes, supports)])
    return known, attributes, supports, means, deviations


def plot_attributes(root, runs, settings, output):
    known, attributes, supports, means, deviations = attribute_values(root, runs, settings)
    fig, ax = plt.subplots(figsize=(7.6, 7.3), layout="constrained")
    palette = LinearSegmentedColormap.from_list("italian_pos", ["#f5f8fa", "#a6c3d0", "#246b8e"])
    palette.set_bad("#e9edf0")
    im = ax.imshow(np.ma.masked_invalid(means), vmin=0, vmax=100, cmap=palette, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS.values(), fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(top=False, bottom=False, left=False, pad=9)
    ax.set_yticks(range(len(attributes)), [key if key in known else f"{key} - fuori training" for key in attributes])
    for row in range(len(attributes)):
        for col in range(len(MODELS)):
            label = (f"{decimal(means[row, col])}\n± {decimal(deviations[row, col])}"
                     if math.isfinite(means[row, col]) else "n.d.")
            ax.text(col, row, label, ha="center", va="center", fontsize=9.6,
                    color="white" if means[row, col] >= 80 else INK)
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(attributes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)
    if len(attributes) > len(known):
        ax.axhline(len(known) - 0.5, color=INK, linewidth=1.5)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(len(attributes)), [f"{support:,}".replace(",", ".") for support in supports], fontsize=10)
    right.tick_params(right=False, pad=8)
    right.set_ylabel("Supporto gold (parole)", fontsize=10)
    for spine in (*ax.spines.values(), *right.spines.values()):
        spine.set_visible(False)
    colorbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.025, ticks=[0, 25, 50, 75, 100])
    colorbar.set_label("Macro-F1 delle coppie presenti per attributo (%)", fontsize=10)
    fig.suptitle("FEATS per attributo - media ± SD su tre seed", fontsize=12)
    fig.supxlabel("Ogni coppia attributo-valore presente pesa uguale, escludendo <NONE>.\nn.d.: nessuna coppia gold presente. Typo=Yes: una occorrenza fuori inventario.", fontsize=9)
    save(fig, output, "feats_attribute_f1")


def history(root, model, seed):
    run = root / model / "original" / f"seed_{seed}"
    rows = read_csv(run / "training_history.csv")
    evaluations = [row for row in rows if row.get("eval_upos_macro_f1_all")]
    if len(evaluations) != 10 or [float(row["epoch"]) for row in evaluations] != list(range(1, 11)):
        raise ValueError(f"Attese dieci valutazioni per epoca: {run}")
    info = read_json(run / "training_info.json")
    selected_step = int(info["best_checkpoint"].replace("\\", "/").rsplit("/", 1)[-1].removeprefix("checkpoint-"))
    selected = next((row for row in evaluations if int(row["step"]) == selected_step), None)
    maximum = max(float(row["eval_upos_macro_f1_all"]) for row in evaluations)
    if selected is None or not math.isclose(float(selected["eval_upos_macro_f1_all"]), maximum, abs_tol=1e-12):
        raise ValueError(f"Checkpoint non selezionato dalla macro-F1 UPOS: {run}")
    return rows, selected


def curve(ax, histories, metric, color, style="-", percent=False):
    series = [[(float(row["epoch"]), float(row[metric])) for row in rows if row.get(metric)] for rows, _ in histories]
    x = np.array([point[0] for point in series[0]])
    if any(len(values) != len(x) or not np.allclose(x, [point[0] for point in values]) for values in series):
        raise ValueError(f"Curve dei seed non allineate: {metric}")
    values = np.array([[point[1] for point in sequence] for sequence in series]) * (100 if percent else 1)
    if not np.isfinite(values).all():
        raise ValueError(f"Curva con valori non finiti: {metric}")
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=1.7)
    lower = mean - sd if percent else np.maximum(mean - sd, 1e-6)
    ax.fill_between(x, lower, mean + sd, color=color, alpha=0.12, linewidth=0)


def plot_dynamics(upos_root, feats_root, output):
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.3), layout="constrained", sharex=True)
    for col, (model, name) in enumerate(MODELS.items()):
        for task, root in (("upos", upos_root), ("upos_feats", feats_root)):
            histories = [history(root, model, seed) for seed in SEEDS]
            color = TASK_COLORS[task]
            curve(axes[0, col], histories, "loss", color)
            curve(axes[0, col], histories, "eval_loss", color, "--")
            curve(axes[1, col], histories, "eval_upos_macro_f1_all", color, percent=True)
        axes[0, col].set_title(name, loc="left", fontweight="bold", pad=12)
        axes[0, col].set_yscale("log")
        axes[1, col].set_xlabel("Epoca")
        for ax in axes[:, col]:
            ax.set_axisbelow(True)
            ax.grid(color=GRID, linewidth=0.7)
            ax.set_xticks([1, 3, 5, 7, 10])
            ax.set_xlim(0.7, 10.3)
        axes[1, col].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
        axes[1, col].yaxis.set_major_locator(MaxNLocator(5))
    axes[0, 0].set_ylabel("Loss (scala logaritmica)")
    axes[1, 0].set_ylabel("Macro-F1 UPOS totale - dev (%)")
    handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=1.7, label=label)
               for color, style, label in ((TASK_COLORS["upos"], "-", "UPOS - training"),
                   (TASK_COLORS["upos"], "--", "UPOS - dev"),
                   (TASK_COLORS["upos_feats"], "-", "UPOS + FEATS - training"),
                   (TASK_COLORS["upos_feats"], "--", "UPOS + FEATS - dev"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Dinamiche di apprendimento - testo originale", fontsize=12, y=1.115)
    fig.supxlabel("Linee: media; fasce: ±SD su tre seed.\nLoss con obiettivi diversi; selezione comune sulla macro-F1 UPOS del development.", fontsize=9)
    save(fig, output, "training_dynamics")


def main():
    parser = argparse.ArgumentParser(description="Figure UPOS e FEATS dai 27 run salvati, senza training o inferenza.")
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/upos_trainer_v1")
    parser.add_argument("--feats-runs", type=Path, default=ROOT / "outputs/upos_feats_trainer_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/report/figures")
    args = parser.parse_args()
    try:
        _, joint_settings = validate_protocol(args.runs, args.feats_runs)
        baseline = read_results(args.runs, ("original", "lowercase"), UPOS_METRICS)
        joint = read_results(args.feats_runs, ("original",), (*UPOS_METRICS, *FEATS_METRICS))
        args.out.mkdir(parents=True, exist_ok=True)
        plot_scores(baseline, args.out)
        plot_delta(baseline, baseline, args.out, "upos_casing_ablation",
                   "Ablation delle maiuscole - minuscolo meno originale", "lowercase")
        plot_feats(joint, args.out)
        plot_delta(baseline, joint, args.out, "upos_feats_ablation",
                   "Ablation FEATS - UPOS + FEATS meno solo UPOS", "original")
        plot_attributes(args.feats_runs, joint, joint_settings, args.out)
        plot_dynamics(args.runs, args.feats_runs, args.out)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"Risultati non disponibili, incompleti o incoerenti: {error}\n")
    print(f"Verificati 18 run UPOS e 9 UPOS + FEATS. Salvate sei figure PDF e PNG in {args.out}")


if __name__ == "__main__":
    main()
