"""Tables and figures for complete, paired evaluations."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.judge import load_ratings


def scored(output, trait):
    answers, ratings = load_ratings(output, trait)
    frame = pd.DataFrame(answers).merge(pd.DataFrame(ratings), on="id", validate="one_to_one")
    frame[[trait, "coherence"]] = frame[[trait, "coherence"]].astype(float)
    valid = frame[trait].notna() & frame.coherence.notna()
    frame["coherent_target"] = np.where(valid, 100 * ((frame[trait] >= 50) & (frame.coherence >= 80)), np.nan)
    return frame


def _wide(question_means, metric):
    return question_means.pivot(index="question_id", columns="condition", values=metric)


def _effect(table, metric, weights):
    paired = table.reindex(columns=list(weights)).dropna()
    values = sum(paired[name] * weight for name, weight in weights.items()).to_numpy()
    if len(values):
        rng = np.random.default_rng(42)
        means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        mean = float(values.mean())
    else:
        mean = low = high = np.nan
    return {
        "metric": metric, "mean_difference": mean, "ci95_low": low,
        "ci95_high": high, "paired_questions": len(values),
    }


def _figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _summary(frame, trait):
    return frame.groupby("condition").agg(
        answers=("id", "size"),
        questions=("question_id", "nunique"),
        target_mean=(trait, "mean"),
        target_valid=(trait, "count"),
        coherence_mean=("coherence", "mean"),
        coherence_valid=("coherence", "count"),
        target_nonzero=(trait, lambda score: int((score > 0).sum())),
        target_ge50=(trait, lambda score: int((score >= 50).sum())),
        coherent_target_rate=("coherent_target", "mean"),
        jointly_valid=("coherent_target", "count"),
        mean_tokens=("generated_tokens", "mean"),
        truncated=("finish_reason", lambda reason: int(reason.eq("length").sum())),
    ).reset_index()


def _heatmaps(frame, trait, output):
    subset = frame[frame.layer.notna()]
    layers, alphas = sorted(subset.layer.unique()), sorted(subset.alpha.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, axis, palette in zip((trait, "coherence"), axes, ("Reds", "Greens")):
        grid = subset.groupby(["layer", "alpha"])[metric].mean().unstack()
        grid = grid.reindex(index=layers, columns=alphas)
        image = axis.imshow(grid.values, aspect="auto", vmin=0, vmax=100, cmap=palette)
        axis.set_xticks(range(len(alphas)), [f"{value:g}" for value in alphas])
        axis.set_yticks(range(len(layers)), [str(int(value)) for value in layers])
        axis.set(xlabel="alpha", ylabel="Layer (zero based)", title=metric)
        for i, row in enumerate(grid.values):
            for j, value in enumerate(row):
                axis.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis)
    return _figure(fig, output / "heatmaps.png")


def _common_tables(frame, trait, output):
    output.mkdir(exist_ok=True)
    metrics = (trait, "coherence", "coherent_target")
    summary = _summary(frame, trait)
    summary.to_csv(output / "summary.csv", index=False)
    frame.to_csv(output / "answers_and_scores.csv", index=False)

    question_means = frame.groupby(["question_id", "condition"])[list(metrics)].mean().reset_index()
    baseline = question_means[question_means.condition.eq("base:baseline")].set_index("question_id")
    question_means["target_change_from_base"] = question_means[trait] - question_means.question_id.map(baseline[trait])
    question_means.to_csv(output / "question_effects.csv", index=False)
    wide = {metric: _wide(question_means, metric) for metric in metrics}
    differences = pd.DataFrame([
        {"condition": condition, "reference": "base:baseline",
         **_effect(table, metric, {condition: 1, "base:baseline": -1})}
        for condition in summary.condition if condition != "base:baseline"
        for metric, table in wide.items()
    ])
    differences.to_csv(output / "differences_from_base.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(summary.coherence_mean, summary.target_mean)
    for i, row in summary.iterrows():
        axis.annotate(row.condition, (row.coherence_mean, row.target_mean),
                      xytext=(3, 3 + (i % 3) * 5), textcoords="offset points", fontsize=7)
    axis.set(xlabel="Coherence", ylabel=trait, title="Trait and coherence")
    return summary, differences, wide, [_figure(fig, output / "tradeoff.png")]


def dev_report(frame, trait, output):
    out = Path(output) / "report"
    summary, differences, _, figures = _common_tables(frame, trait, out)
    if frame.layer.notna().any():
        figures.append(_heatmaps(frame, trait, out))
    return summary, differences, figures


def _test_comparisons(choices, wide):
    def add(choice, model, vector, name, weights):
        for metric, scores in wide.items():
            comparisons.append({
                "choice": choice, "model": model, "vector": vector, "comparison": name,
                **_effect(scores, metric, weights),
            })

    comparisons = []
    add("shared", "normal", "none", "lora_without_steering", {"normal:baseline": 1, "base:baseline": -1})
    for choice in choices:
        name = choice["name"]
        steered = f"steered_{name}"
        base_old = f"base:{name}_old"
        for variant in ("old", "new_raw", "new_matched"):
            suffix = f"{name}_{variant}"
            for model in ("normal", steered):
                current, baseline = f"{model}:{suffix}", f"{model}:baseline"
                add(name, model, variant, "within_model_steering", {current: 1, baseline: -1})
                add(name, model, variant, "restoration_gap", {current: 1, base_old: -1})
                add(name, model, variant, "change_in_steering_effect",
                    {current: 1, baseline: -1, base_old: -1, "base:baseline": 1})
            normal_old, steered_old = f"normal:{suffix}", f"{steered}:{suffix}"
            add(name, steered, variant, "steered_vs_normal_training", {steered_old: 1, normal_old: -1})
            add(name, steered, variant, "extra_attenuation_steered_vs_normal",
                {steered_old: 1, f"{steered}:baseline": -1, normal_old: -1, "normal:baseline": 1})
        add(name, steered, "baseline", "steered_vs_normal_training",
            {f"{steered}:baseline": 1, "normal:baseline": -1})
    return pd.DataFrame(comparisons)


def test_report(frame, root):
    root = Path(root)
    trait = read_json(root / "run.json")["trait"]
    choices = read_json(root / "choices.json")["choices"]
    output = root / "test" / "all" / "report"
    summary, _, wide, figures = _common_tables(frame, trait, output)
    effects = _test_comparisons(choices, wide)
    effects.to_csv(output / "paired_effects.csv", index=False)
    for choice in choices:
        name = choice["name"]
        suffixes = [f"{name}_{variant}" for variant in ("old", "new_raw", "new_matched")]
        conditions = ["base:baseline", f"base:{name}_old"]
        for model in ("normal", f"steered_{name}"):
            conditions.extend(f"{model}:{suffix}" for suffix in ["baseline", *suffixes])
        selected = summary.set_index("condition").loc[conditions]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for axis, column, label in zip(axes, ("target_mean", "coherence_mean"), (trait, "coherence")):
            axis.bar(range(len(selected)), selected[column])
            axis.set_xticks(range(len(selected)), selected.index, rotation=65, ha="right", fontsize=8)
            axis.set(title=f"{name} — {label}", ylim=(0, 100))
        figures.append(_figure(fig, output / f"{name}_conditions.png"))
    return summary, effects, figures


def extraction_report(root):
    folder = Path(root) / "vectors" / "base"
    frame = pd.read_csv(folder / "geometry.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    for axis, metric in zip(axes, ("vector_norm", "relative_norm", "split_half_cosine")):
        axis.plot(frame.layer, frame[metric], marker=".")
        axis.set(xlabel="Layer (zero based)", title=metric)
    return frame, _figure(fig, folder / "geometry.png")


def compare_vectors(root, label):
    from torch.nn.functional import cosine_similarity

    from src.workflow import load_vector

    folder = Path(root) / "vectors" / label
    old, new = load_vector(root, "base"), load_vector(root, label)
    if old["pair_ids"] != new["pair_ids"]:
        raise ValueError("Extraction pair IDs differ.")
    frame = pd.read_csv(folder / "geometry.csv")
    frame["old_norm"] = old["vectors"].norm(dim=-1).numpy()
    frame["norm_ratio"] = frame.vector_norm / frame.old_norm
    frame["cosine_to_old"] = cosine_similarity(old["vectors"], new["vectors"], dim=-1).numpy()
    frame.to_csv(folder / "geometry.csv", index=False)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3))
    left.plot(frame.layer, frame.cosine_to_old)
    left.set(xlabel="Layer (zero based)", title="Cosine to original", ylim=(-1.05, 1.05))
    right.plot(frame.layer, frame.norm_ratio)
    right.axhline(1, color="grey", linestyle="--")
    right.set(xlabel="Layer (zero based)", title="New norm / original norm")
    return frame, _figure(fig, folder / "comparison.png")


def validation_loss(root, labels):
    folder = Path(root) / "adapters"
    fig, axis = plt.subplots(figsize=(8, 4))
    for label in labels:
        log = pd.read_csv(folder / label / "loss.csv").dropna(subset=["eval_loss"])
        axis.plot(log.epoch, log.eval_loss, marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Validation response loss")
    axis.legend()
    return _figure(fig, folder / "validation_loss.png")


def examples(frame, trait, condition):
    wide = frame.pivot_table(index="question_id", columns="condition", values=trait, aggfunc="mean")
    columns = ["question_id", "question", "condition", "answer", trait, "coherence"]
    if condition not in wide or "base:baseline" not in wide:
        return frame.iloc[0:0][columns]
    delta = (wide[condition] - wide["base:baseline"]).dropna().sort_values()
    if delta.empty:
        return frame.iloc[0:0][columns]
    ids = [delta.abs().idxmax(), delta.index[len(delta) // 2]]
    return frame.loc[
        frame.question_id.isin(ids) & frame.condition.isin(["base:baseline", condition])
        & frame.repeat.eq(0), columns
    ]
