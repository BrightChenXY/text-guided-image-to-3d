from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from PIL import Image

from pipelines.image_editor import edit_image_with_prompt
from training.dataset import DEFAULT_PROMPT_SUFFIX, Pix2PixJsonlDataset
from training.trellis_eval import TrellisProxyWeights, evaluate_edited_image_with_trellis


PLOT_COLOR_BASELINE = "#6b7280"
PLOT_COLOR_LORA = "#f57c00"
METRIC_SPECS = (
    ("front_similarity", "trellis/front_similarity"),
    ("mean_coverage_score", "trellis/coverage_score"),
    ("mean_centering_score", "trellis/centering_score"),
    ("mean_connectivity_score", "trellis/connectivity_score"),
    ("mean_border_margin_score", "trellis/border_margin_score"),
    ("overall_score", "trellis/mean_score"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline InstructPix2Pix against a LoRA-enhanced model using TRELLIS proxy scoring.",
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        required=True,
        help="Validation metadata.jsonl used for the comparison.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        required=True,
        help="Path to the LoRA weights directory, for example .../best_checkpoint/lora.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/outputs/trellis_compare"),
        help="Directory used to store metrics, charts, and preview images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="How many validation rows to compare.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Square resolution used when preparing validation images.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=("pad", "crop"),
        default="pad",
        help="Image resize strategy for validation examples.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Image editing inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Prompt guidance scale for both baseline and LoRA evaluation.",
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="Image guidance scale for both baseline and LoRA evaluation.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA cross-attention scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic editing.",
    )
    parser.add_argument(
        "--trellis-seed",
        type=int,
        default=0,
        help="Seed forwarded to TRELLIS during proxy evaluation.",
    )
    parser.add_argument(
        "--trellis-render-size",
        type=int,
        default=256,
        help="TRELLIS render resolution used for proxy scoring.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default=DEFAULT_PROMPT_SUFFIX,
        help="Prompt suffix used to build the evaluation prompt.",
    )
    return parser.parse_args()


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _extract_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    metrics = result.get("metrics", {})
    return {
        "front_similarity": _round(metrics.get("front_similarity")),
        "mean_coverage_score": _round(metrics.get("mean_coverage_score")),
        "mean_centering_score": _round(metrics.get("mean_centering_score")),
        "mean_connectivity_score": _round(metrics.get("mean_connectivity_score")),
        "mean_border_margin_score": _round(metrics.get("mean_border_margin_score")),
        "overall_score": _round(result.get("score")),
        "success": 1.0 if result.get("success") else 0.0,
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return float(mean(values))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_preview_strip(
    original_image: Image.Image,
    target_image: Image.Image,
    baseline_image: Image.Image,
    lora_image: Image.Image,
) -> Image.Image:
    width, height = original_image.size
    canvas = Image.new("RGB", (width * 4, height), color=(255, 255, 255))
    canvas.paste(original_image.resize((width, height)), (0, 0))
    canvas.paste(target_image.resize((width, height)), (width, 0))
    canvas.paste(baseline_image.resize((width, height)), (width * 2, 0))
    canvas.paste(lora_image.resize((width, height)), (width * 3, 0))
    return canvas


def _plot_summary_bars(summary: dict[str, Any], output_path: Path) -> None:
    labels = ["Baseline", "LoRA"]
    values = [
        float(summary["baseline"]["mean_metrics"]["overall_score"]),
        float(summary["lora"]["mean_metrics"]["overall_score"]),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        labels,
        values,
        color=[PLOT_COLOR_BASELINE, PLOT_COLOR_LORA],
        width=0.55,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TRELLIS Mean Score")
    ax.set_title("Baseline vs LoRA")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_metric_groups(summary: dict[str, Any], output_path: Path) -> None:
    metric_labels = [
        "Front Sim.",
        "Coverage",
        "Centering",
        "Connectivity",
        "Border Margin",
        "Mean Score",
    ]
    baseline_values = [
        float(summary["baseline"]["mean_metrics"]["front_similarity"]),
        float(summary["baseline"]["mean_metrics"]["mean_coverage_score"]),
        float(summary["baseline"]["mean_metrics"]["mean_centering_score"]),
        float(summary["baseline"]["mean_metrics"]["mean_connectivity_score"]),
        float(summary["baseline"]["mean_metrics"]["mean_border_margin_score"]),
        float(summary["baseline"]["mean_metrics"]["overall_score"]),
    ]
    lora_values = [
        float(summary["lora"]["mean_metrics"]["front_similarity"]),
        float(summary["lora"]["mean_metrics"]["mean_coverage_score"]),
        float(summary["lora"]["mean_metrics"]["mean_centering_score"]),
        float(summary["lora"]["mean_metrics"]["mean_connectivity_score"]),
        float(summary["lora"]["mean_metrics"]["mean_border_margin_score"]),
        float(summary["lora"]["mean_metrics"]["overall_score"]),
    ]

    positions = list(range(len(metric_labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        [position - width / 2 for position in positions],
        baseline_values,
        width=width,
        color=PLOT_COLOR_BASELINE,
        label="Baseline",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        lora_values,
        width=width,
        color=PLOT_COLOR_LORA,
        label="LoRA",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("TRELLIS Proxy Metric Comparison")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_delta_bars(summary: dict[str, Any], output_path: Path) -> None:
    keys = [metric_key for metric_key, _ in METRIC_SPECS]
    labels = [
        "Front Sim.",
        "Coverage",
        "Centering",
        "Connectivity",
        "Border Margin",
        "Mean Score",
    ]
    deltas = [
        float(summary["lora"]["mean_metrics"][key]) - float(summary["baseline"]["mean_metrics"][key])
        for key in keys
    ]
    colors = [PLOT_COLOR_LORA if delta >= 0 else "#c62828" for delta in deltas]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, deltas, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_ylabel("LoRA - Baseline")
    ax.set_title("TRELLIS Proxy Delta")
    for bar, value in zip(bars, deltas):
        va = "bottom" if value >= 0 else "top"
        offset = 0.01 if value >= 0 else -0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.val_metadata = args.val_metadata.expanduser().resolve()
    args.lora_path = args.lora_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.val_metadata.exists():
        raise FileNotFoundError(f"Validation metadata not found: {args.val_metadata}")
    if not args.lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    dataset = Pix2PixJsonlDataset(
        metadata_path=args.val_metadata,
        tokenizer=None,
        resolution=args.resolution,
        prompt_suffix=args.prompt_suffix,
        resize_mode=args.resize_mode,
        max_records=args.max_samples,
        skip_missing_images=True,
    )

    score_weights = TrellisProxyWeights()
    per_sample_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    lora_rows: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        example = dataset.get_visual_example(index)
        original_image = example["original_image"]
        target_image = example["edited_image"]
        prompt = example["prompt"]
        record = example.get("record")

        baseline_image = edit_image_with_prompt(
            original_image,
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            seed=args.seed,
            lora_path="",
            lora_scale=1.0,
        )
        lora_image = edit_image_with_prompt(
            original_image,
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            seed=args.seed,
            lora_path=args.lora_path,
            lora_scale=args.lora_scale,
        )

        baseline_result = evaluate_edited_image_with_trellis(
            baseline_image,
            work_dir=args.output_dir / "baseline" / f"sample_{index:04d}",
            seed=args.trellis_seed,
            render_size=args.trellis_render_size,
            score_weights=score_weights,
        )
        lora_result = evaluate_edited_image_with_trellis(
            lora_image,
            work_dir=args.output_dir / "lora" / f"sample_{index:04d}",
            seed=args.trellis_seed,
            render_size=args.trellis_render_size,
            score_weights=score_weights,
        )

        preview = _build_preview_strip(
            original_image=original_image,
            target_image=target_image,
            baseline_image=baseline_image,
            lora_image=lora_image,
        )
        preview.save(previews_dir / f"sample_{index:04d}.png")

        baseline_metrics = _extract_metrics(baseline_result)
        lora_metrics = _extract_metrics(lora_result)
        baseline_rows.append(baseline_metrics)
        lora_rows.append(lora_metrics)

        row = {
            "sample_index": index,
            "metadata_id": getattr(record, "metadata_id", None),
            "original_dataset_index": getattr(record, "original_dataset_index", None),
            "prompt": prompt,
            "baseline_front_similarity": baseline_metrics["front_similarity"],
            "baseline_coverage_score": baseline_metrics["mean_coverage_score"],
            "baseline_centering_score": baseline_metrics["mean_centering_score"],
            "baseline_connectivity_score": baseline_metrics["mean_connectivity_score"],
            "baseline_border_margin_score": baseline_metrics["mean_border_margin_score"],
            "baseline_mean_score": baseline_metrics["overall_score"],
            "baseline_success": baseline_metrics["success"],
            "lora_front_similarity": lora_metrics["front_similarity"],
            "lora_coverage_score": lora_metrics["mean_coverage_score"],
            "lora_centering_score": lora_metrics["mean_centering_score"],
            "lora_connectivity_score": lora_metrics["mean_connectivity_score"],
            "lora_border_margin_score": lora_metrics["mean_border_margin_score"],
            "lora_mean_score": lora_metrics["overall_score"],
            "lora_success": lora_metrics["success"],
            "delta_mean_score": _round(
                float(lora_metrics["overall_score"] or 0.0)
                - float(baseline_metrics["overall_score"] or 0.0)
            ),
        }
        per_sample_rows.append(row)
        print(
            f"[{index + 1}/{len(dataset)}] baseline={baseline_metrics['overall_score']:.4f} | "
            f"lora={lora_metrics['overall_score']:.4f} | prompt={prompt[:80]}",
            flush=True,
        )

    summary = {
        "sample_count": len(per_sample_rows),
        "lora_path": str(args.lora_path),
        "trellis_render_size": args.trellis_render_size,
        "trellis_seed": args.trellis_seed,
        "score_weights": score_weights.to_dict(),
        "baseline": {
            "mean_metrics": {
                key: _round(_mean_metric(baseline_rows, key))
                for key, _ in METRIC_SPECS
            },
            "success_rate": _round(_mean_metric(baseline_rows, "success")),
        },
        "lora": {
            "mean_metrics": {
                key: _round(_mean_metric(lora_rows, key))
                for key, _ in METRIC_SPECS
            },
            "success_rate": _round(_mean_metric(lora_rows, "success")),
        },
    }
    summary["delta"] = {
        key: _round(
            float(summary["lora"]["mean_metrics"][key] or 0.0)
            - float(summary["baseline"]["mean_metrics"][key] or 0.0)
        )
        for key, _ in METRIC_SPECS
    }
    summary["better_model"] = (
        "lora"
        if float(summary["lora"]["mean_metrics"]["overall_score"] or 0.0)
        >= float(summary["baseline"]["mean_metrics"]["overall_score"] or 0.0)
        else "baseline"
    )

    _save_csv(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
    _save_json(args.output_dir / "summary.json", summary)
    _plot_summary_bars(summary, args.output_dir / "mean_score_comparison.png")
    _plot_metric_groups(summary, args.output_dir / "proxy_metrics_comparison.png")
    _plot_delta_bars(summary, args.output_dir / "proxy_metric_delta.png")

    print(f"Saved comparison outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
