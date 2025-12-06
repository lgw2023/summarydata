from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import PipelineConfig
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging_utils import init_logger
from src.utils.env import load_env

try:  # 可选依赖：仅在安装 matplotlib 时才绘制图表
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _flatten_candidates(generated_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 generated_responses.jsonl 中的一行多候选展开为扁平列表，方便做统计与绘图。
    """
    flat: List[Dict[str, Any]] = []
    for row in generated_rows:
        sample_id = row.get("sample_id")
        for cand in row.get("candidates", []) or []:
            flat.append(
                {
                    "sample_id": sample_id,
                    "model_type": cand.get("model_type", ""),
                    "model_name": cand.get("model_name", ""),
                    "response": cand.get("response", "") or "",
                }
            )
    return flat


def run_analyze_generated(config_path: str | Path, raw_data_path: str | Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Loading config from %s (raw_data=%s)", config_path, raw_data_path)
    config = PipelineConfig.from_yaml(config_path, raw_data_path=raw_data_path)

    generated_rows = read_jsonl(config.output_files.generated_responses)
    if not generated_rows:
        logger.warning("No generated_responses found at %s", config.output_files.generated_responses)
        print("No generated_responses found. Please run the generation stage first.")
        return

    flat = _flatten_candidates(generated_rows)
    if not flat:
        print("generated_responses.jsonl contains no candidates.")
        return

    # 1) 统计每个模型的候选数量与平均长度
    buckets: Dict[str, List[int]] = {}
    for cand in flat:
        key = f'{cand.get("model_type","")}/{cand.get("model_name","")}'
        length = len(str(cand.get("response", "")))
        buckets.setdefault(key, []).append(length)

    print("Generated response length summary (by model_type/model_name):")
    for key, lengths in buckets.items():
        if not lengths:
            continue
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(
            f"{key}: count={len(lengths)}, "
            f"avg_len={avg_len:.1f}, min_len={min_len}, max_len={max_len}"
        )

    # 2) 每个样本的候选数量分布
    per_sample_counts: Dict[str, int] = {}
    for row in generated_rows:
        sid = str(row.get("sample_id"))
        per_sample_counts[sid] = len(row.get("candidates", []) or [])
    counts = list(per_sample_counts.values())
    print(
        "\nCandidates per sample: "
        f"min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.2f}"
    )

    # 3) 可选：绘制长度分布与候选数量分布
    if plt is not None:
        reports_dir = ensure_dir(Path(config_path).resolve().parent / "../reports")
        plots_dir = ensure_dir(reports_dir / "generated_plots")

        # 3.1 所有候选的长度直方图
        all_lengths = [len(str(c["response"])) for c in flat]
        plt.figure(figsize=(6, 4))
        plt.hist(all_lengths, bins=30, alpha=0.7)
        plt.title("Response length distribution (all models)")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = plots_dir / "response_length_all.png"
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved overall response length histogram to %s", out_path)

        # 3.2 按模型拆分的长度直方图
        for key, lengths in buckets.items():
            if not lengths:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(lengths, bins=30, alpha=0.7)
            plt.title(f"Response length - {key}")
            plt.xlabel("Length (characters)")
            plt.ylabel("Count")
            plt.tight_layout()
            safe_key = key.replace("/", "_")
            out_path = plots_dir / f"response_length_{safe_key}.png"
            plt.savefig(out_path)
            plt.close()
            logger.info("Saved per-model response length histogram to %s", out_path)

        # 3.3 每个样本候选数量分布
        plt.figure(figsize=(6, 4))
        plt.hist(counts, bins=range(1, max(counts) + 2), align="left", rwidth=0.8)
        plt.title("Candidates per sample")
        plt.xlabel("Number of candidates")
        plt.ylabel("Count of samples")
        plt.tight_layout()
        out_path = plots_dir / "candidates_per_sample.png"
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved candidates-per-sample histogram to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze generated_responses.jsonl for basic sanity checks and visualizations"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (用于生成分析配置，不再固定原始数据文件)",
    )
    parser.add_argument(
        "--raw-data",
        required=True,
        help="本次实验的输入数据文件路径（例如 CSV/Excel），用于确定 data/<输入文件名>/ 下的 generated_responses.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    init_logger()
    args = parse_args()
    run_analyze_generated(args.config, raw_data_path=args.raw_data)


