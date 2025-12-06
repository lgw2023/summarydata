from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


@dataclass
class OutputFiles:
    samples: Path
    generated_responses: Path
    judge_results: Path
    ranked_pairs: Path


@dataclass
class RankingConfig:
    top_k: int = 1
    bottom_k: int = 1
    min_score_diff: float = 0.0


@dataclass
class PipelineConfig:
    """
    统一的 Pipeline 配置对象。

    重要约定（为支持“按输入文件名隔离输出目录”）：
    - raw_data 表示**本次实验的输入数据文件**（CSV / Excel 等）；
    - 若 YAML 中未显式指定 processed_dir / intermediate_dir / output_files.*，
      则会根据 raw_data 的文件名自动推导出形如：

          <PROJECT_ROOT>/data/<raw_data_stem>/processed/
          <PROJECT_ROOT>/data/<raw_data_stem>/intermediate/

      以及：

          samples.jsonl
          generated_responses.jsonl
          judge_results.jsonl
          ranked_pairs.jsonl
    """

    raw_data: Path
    processed_dir: Path
    intermediate_dir: Path
    generators: list[dict[str, Any]]
    judges: list[dict[str, Any]]
    ranking: RankingConfig
    output_files: OutputFiles

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        raw_data_path: str | Path | None = None,
    ) -> "PipelineConfig":
        """
        从 YAML 载入 PipelineConfig。

        参数：
        - path: 配置文件路径（通常为 configs/*.yaml）
        - raw_data_path: **可选**，用于在命令行中显式指定输入数据文件；
          若提供，则优先于 YAML 中的 paths.raw_data，便于做到：
          - “脚本运行时必须显式传入输入文件”，而不是依赖写死在 YAML 里的默认值；
          - 根据输入文件名自动切分输出目录。
        """
        path = Path(path).resolve()
        base_dir = path.parent
        # 基于当前约定：configs/ 位于项目根目录下，故 project_root = configs/ 的上一级。
        # 若未来目录结构调整，这里可以改为显式注入。
        project_root = base_dir.parent

        config = load_yaml(path)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid config file: {path}, expected a mapping at top level.")

        paths_cfg = config.get("paths") or {}
        pipeline_cfg = config.get("pipeline") or {}

        if not isinstance(paths_cfg, dict):
            raise ValueError("Config missing required 'paths' section (must be a mapping).")
        if not isinstance(pipeline_cfg, dict):
            raise ValueError("Config missing required 'pipeline' section (must be a mapping).")

        # ===== 1. 解析 raw_data（优先使用命令行传入的 raw_data_path） =====
        raw_data_value: str | Path | None
        if raw_data_path is not None:
            raw_data_value = raw_data_path
        else:
            raw_data_value = paths_cfg.get("raw_data")

        if not raw_data_value:
            raise ValueError(
                "Raw data path is not specified. "
                "Please either:\n"
                "- pass raw_data_path to PipelineConfig.from_yaml(...), or\n"
                "- provide 'paths.raw_data' in your YAML config."
            )

        raw_data = Path(raw_data_value)
        if not raw_data.is_absolute():
            # 命令行显式传入的 raw_data_path：
            # - 更符合用户直觉的是「相对于项目根目录」而不是「相对于 YAML 所在目录」，
            #   因为大多数用户都会在项目根目录下执行脚本。
            # - 这样可以直接使用诸如 `--raw-data data_diff_sample.csv`
            #   或 `--raw-data data/my_exp.csv` 的相对路径。
            #
            # YAML 内部的 paths.raw_data（即 raw_data_path 为 None 的情况）则继续保持
            # 之前的行为：相对于配置文件所在目录进行解析，以兼容旧配置。
            if raw_data_path is not None:
                raw_data = project_root / str(raw_data_value)
            else:
                raw_data = _resolve_path(base_dir, str(raw_data_value))

        # 基础路径存在性校验
        if not raw_data.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data}")

        # 根据 raw_data 文件名自动推导本次实验的“命名空间”
        # 例如：summary_train_v3.csv -> summary_train_v3
        run_name = raw_data.stem

        # ===== 2. 解析 processed / intermediate 目录 =====
        # 若 YAML 中未显式给出，则根据 raw_data 自动推导：
        #   <PROJECT_ROOT>/data/<run_name>/processed
        #   <PROJECT_ROOT>/data/<run_name>/intermediate
        default_processed_dir = project_root / "data" / run_name / "processed"
        default_intermediate_dir = project_root / "data" / run_name / "intermediate"

        processed_dir_cfg = paths_cfg.get("processed_dir")
        intermediate_dir_cfg = paths_cfg.get("intermediate_dir")

        processed_dir = _resolve_path(
            base_dir,
            processed_dir_cfg or str(default_processed_dir),
        )
        intermediate_dir = _resolve_path(
            base_dir,
            intermediate_dir_cfg or str(default_intermediate_dir),
        )

        ranking_cfg = pipeline_cfg.get("ranking", {}) or {}
        output_cfg = pipeline_cfg.get("output_files", {}) or {}

        # 排序配置校验
        top_k = int(ranking_cfg.get("top_k", 1))
        bottom_k = int(ranking_cfg.get("bottom_k", 1))
        min_score_diff = float(ranking_cfg.get("min_score_diff", 0.0))
        if top_k <= 0 or bottom_k <= 0:
            raise ValueError("ranking.top_k and ranking.bottom_k must be positive integers.")
        if min_score_diff < 0.0:
            raise ValueError("ranking.min_score_diff must be >= 0.")

        generators_cfg = pipeline_cfg.get("generators")
        judges_cfg = pipeline_cfg.get("judges")
        if not isinstance(generators_cfg, list) or not generators_cfg:
            raise ValueError("pipeline.generators must be a non-empty list.")
        if not isinstance(judges_cfg, list) or not judges_cfg:
            raise ValueError("pipeline.judges must be a non-empty list.")

        # ===== 3. 解析输出文件路径 =====
        # 仍然允许在 YAML 中自定义 output_files.*，但默认情况下：
        # - 与 raw_data 强绑定，落在 data/<run_name>/processed/ 下；
        # - 文件名固定为 samples.jsonl / generated_responses.jsonl / judge_results.jsonl / ranked_pairs.jsonl。
        default_samples_path = processed_dir / "samples.jsonl"
        default_generated_responses_path = processed_dir / "generated_responses.jsonl"
        default_judge_results_path = processed_dir / "judge_results.jsonl"
        default_ranked_pairs_path = processed_dir / "ranked_pairs.jsonl"

        return cls(
            raw_data=raw_data,
            processed_dir=processed_dir,
            intermediate_dir=intermediate_dir,
            generators=generators_cfg,
            judges=judges_cfg,
            ranking=RankingConfig(
                top_k=top_k,
                bottom_k=bottom_k,
                min_score_diff=min_score_diff,
            ),
            output_files=OutputFiles(
                samples=_resolve_path(
                    base_dir,
                    output_cfg.get("samples", str(default_samples_path)),
                ),
                generated_responses=_resolve_path(
                    base_dir,
                    output_cfg.get(
                        "generated_responses",
                        str(default_generated_responses_path),
                    ),
                ),
                judge_results=_resolve_path(
                    base_dir,
                    output_cfg.get(
                        "judge_results",
                        str(default_judge_results_path),
                    ),
                ),
                ranked_pairs=_resolve_path(
                    base_dir,
                    output_cfg.get(
                        "ranked_pairs",
                        str(default_ranked_pairs_path),
                    ),
                ),
            ),
        )
