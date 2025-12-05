from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.analysis.stats import compute_model_stats
from src.scoring.aggregator import flatten_grouped_scores
from src.utils.io import read_jsonl, ensure_dir, write_jsonl
from src.utils.logging_utils import init_logger
from src.utils.env import load_env

try:  # 可选依赖：仅在安装 matplotlib 时才绘制直方图
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:  # 可选依赖：仅在安装 pandas 时写 Excel
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def run_analyze(config_path: str | Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s", config_path)
    config = PipelineConfig.from_yaml(config_path)
    raw_rows = read_jsonl(config.output_files.judge_results)
    judge_rows = flatten_grouped_scores(raw_rows)
    if not judge_rows:
        logger.warning("No judge_results found at %s", config.output_files.judge_results)
        print("No judge_results found. Please run the pipeline/judge stage first.")
        return

    # 基于全部打分记录按 model_type/model_name 统计
    stats = compute_model_stats(judge_rows)

    # 1) 打印到 stdout
    print("Model performance summary:")
    for key, s in stats.items():
        print(
            f"{key}: count={s['count']}, mean={s['mean']:.3f}, "
            f"std={s['std']:.3f}, min={s['min']:.3f}, max={s['max']:.3f}"
        )

    # 2) 保存 JSON / JSONL / Excel 报告到 reports/
    reports_dir = ensure_dir(Path(config_path).resolve().parent / "../reports")

    # 2.1 JSONL：每行一个模型统计（保持向后兼容）
    stats_jsonl_path = reports_dir / "score_stats.jsonl"
    stat_rows = [
        {
            "model": model,
            **values,
        }
        for model, values in stats.items()
    ]
    write_jsonl(stats_jsonl_path, stat_rows)

    # 2.2 JSON：整体一个对象，便于快速查看与下游消费
    stats_json_path = reports_dir / "score_stats.json"
    with stats_json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 2.3 Excel：一行一个模型，列为指标（若安装了 pandas）
    if pd is not None and stat_rows:
        df = pd.DataFrame(stat_rows)  # type: ignore[arg-type]
        xlsx_path = reports_dir / "score_stats.xlsx"
        df.to_excel(xlsx_path, index=False)

    # 3) 可选：绘制直方图
    if plt is not None:
        # 3.1 所有模型整体分布
        all_scores = [float(r.get("aggregate_score", 0.0)) for r in judge_rows]
        plots_dir = ensure_dir(reports_dir / "score_plots")

        plt.figure(figsize=(6, 4))
        plt.hist(all_scores, bins=20, alpha=0.7)
        plt.title("Aggregate score distribution")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "aggregate_score_hist.png")
        plt.close()
        logger.info("Saved aggregate score histogram to %s", plots_dir / "aggregate_score_hist.png")

        # 3.2 按模型拆分的直方图（每个模型一张图）
        # key 形如 "model_type/model_name"
        per_model_scores: dict[str, list[float]] = {}
        for row in judge_rows:
            key = f'{row.get("model_type","")}/{row.get("model_name","")}'
            per_model_scores.setdefault(key, []).append(float(row.get("aggregate_score", 0.0)))

        for key, scores in per_model_scores.items():
            if not scores:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(scores, bins=20, alpha=0.7)
            plt.title(f"Aggregate score distribution - {key}")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.tight_layout()
            safe_key = key.replace("/", "_")
            out_path = plots_dir / f"aggregate_score_hist_{safe_key}.png"
            plt.savefig(out_path)
            plt.close()
            logger.info("Saved per-model histogram to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze judge scores and print basic statistics")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_analyze(args.config)


