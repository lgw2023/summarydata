from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import logging
import os
import time
from urllib.parse import urlparse

from src.data_loader.excel_loader import Sample
from src.generators.base import Candidate
from src.scoring.parser import safe_parse_json, compute_overall_score
from score_prompt import GROUND_PROMPT_TPL, STRUCT_PROMPT_TPL

try:  # 可选依赖，仅在使用 LLMJudge 时才需要
    import httpx  # type: ignore
    from httpx import RequestError, HTTPStatusError  # type: ignore
except Exception:  # pragma: no cover
    httpx = None
    RequestError = Exception  # type: ignore
    HTTPStatusError = Exception  # type: ignore


@dataclass
class JudgeScore:
    """
    单次 judge 对一个候选的打分结果。

    为了便于后续聚合与排序，这里携带 candidate 的基础元信息。
    """

    sample_id: str
    candidate_id: str
    model_type: str
    model_name: str
    judge: str
    ground_score: float
    structure_score: float
    ground_max_score: float = 5.0
    structure_max_score: float = 5.0
    ground_raw: str | None = None
    structure_raw: str | None = None
    # 新增：记录用于打分的完整 prompt，便于调试与复现
    ground_prompt: str | None = None
    structure_prompt: str | None = None
    notes: str | None = None


class BaseJudge:
    name: str = "judge"

    def judge(self, sample: Sample, candidate: Candidate) -> JudgeScore:  # pragma: no cover - interface
        raise NotImplementedError


class LLMJudge(BaseJudge):
    """
    使用 LLM 作为裁判的 Judge。

    假设使用 OpenAI 兼容接口（如 DeepSeek / DeepInfra 等）：
    - 通过 HTTP POST 调用 /v1/chat/completions
    - 使用 score_prompt.py 中的 GROUND_PROMPT_TPL 与 STRUCT_PROMPT_TPL
    """

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 60.0,
        max_retries: int = 29,
        backoff_base: float = 1.0,
        ground_prompt_tpl: str | None = None,
        struct_prompt_tpl: str | None = None,
    ) -> None:
        if httpx is None:  # pragma: no cover
            raise RuntimeError("httpx is required for LLMJudge but is not installed.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._client = httpx.Client(timeout=timeout)
        self._logger = logging.getLogger(__name__)
        # 允许从外部传入自定义提示词模板（例如从 prompts/*.txt 读取）
        self._ground_prompt_tpl = ground_prompt_tpl or GROUND_PROMPT_TPL
        self._struct_prompt_tpl = struct_prompt_tpl or STRUCT_PROMPT_TPL

    @staticmethod
    def _build_input_data(sample: Sample) -> str:
        """
        构建与 score_prompt 中描述一致的 input_data：
        - 简化为「上一轮对话（若有：last_query + last_answer_phone）+ 当前 query」。
        """
        parts = []
        last_query = getattr(sample, "last_query", None)
        last_answer = sample.last_answer_phone
        if (last_query and last_query.strip()) or (last_answer and last_answer.strip()):
            # 上一轮用户提问
            if last_query and last_query.strip():
                parts.append(f"user: {last_query.strip()}")
            else:
                parts.append("user: （上一轮问题略）")
            # 上一轮助手回复
            if last_answer and last_answer.strip():
                parts.append(f"assistant: {last_answer.strip()}")
        parts.append(f"user: {sample.query}")
        return "\n".join(parts)

    @staticmethod
    def _build_modules_block(sample: Sample) -> str:
        """
        根据 data/suggest/rag/service 字段构建 modules_block 文本。
        """
        blocks = []
        blocks.append("[个人数据]")
        blocks.append((sample.data or "").strip() or "（无）")
        blocks.append("")

        blocks.append("[专家建议]")
        blocks.append((sample.suggest or "").strip() or "（无）")
        blocks.append("")

        blocks.append("[知识库知识]")
        blocks.append((sample.rag or "").strip() or "（无）")
        blocks.append("")

        # 课程库（用于 COURSE_LIB_MISSING 等规则的核对）
        blocks.append("[课程库]")
        if sample.service:
            blocks.append(sample.service.strip())
        else:
            blocks.append("（无）")

        return "\n".join(blocks)

    @staticmethod
    def _build_history_input(sample: Sample) -> str:
        """
        构建与 score_prompt 中描述一致的 history_input：
        - 简化为「上一轮对话（若有：last_query + last_answer_phone）」。
        """
        parts = []
        last_query = getattr(sample, "last_query", None)
        last_answer = getattr(sample, "last_answer_phone", None)
        if last_answer and last_answer.strip():
            # 上一轮用户提问
            if last_query and last_query.strip():
                parts.append(f"user: {last_query.strip()}")
            else:
                parts.append("user: （上一轮问题略）")
            # 上一轮助手回复
            parts.append(f"assistant: {last_answer.strip()}")
        else:
            parts.append("（无）")
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """
        调用 OpenAI 兼容接口，并返回模型的文本输出。
        带有限次数的重试与指数退避。
        """
        # 规范化 OpenAI 兼容接口的 URL 拼接逻辑：
        # - DashScope：通常 base_url 形如 https://dashscope.aliyuncs.com/compatible-mode/v1
        # - DeepInfra：通常 base_url 形如 https://api.deepinfra.com/v1/openai
        # - DeepSeek：文档示例多为 https://api.deepseek.com/chat/completions
        #
        # 统一约定：
        # - 若 base_url 已经包含 /chat/completions，则直接使用；
        # - 若域名包含 deepseek.com 且路径中尚未出现 /v1，则追加 /chat/completions；
        # - 若路径中已经出现过 /v1（无论后面是否还有 /openai 等），则只追加 /chat/completions；
        # - 否则默认追加 /v1/chat/completions。
        base = self._base_url.rstrip("/")
        parsed = urlparse(base)
        path = parsed.path or ""
        host = parsed.netloc or ""
        if "/chat/completions" in path:
            url = base
        elif "deepseek.com" in host and "/v1" not in path:
            url = f"{base}/chat/completions"
        elif "/v1" in path:
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            start_ts = time.time()
            try:
                # 每一次 Judge 的 LLM 调用开始时输出反馈
                self._logger.info(
                    "LLM judge call start | model=%s | attempt=%d/%d",
                    self._model_name,
                    attempt + 1,
                    self._max_retries + 1,
                )
                resp = self._client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                duration = time.time() - start_ts
                # 每一次 Judge 调用成功后的反馈日志
                self._logger.info(
                    "LLM judge call success | model=%s | attempt=%d/%d | elapsed=%.2fs",
                    self._model_name,
                    attempt + 1,
                    self._max_retries + 1,
                    duration,
                )
                return content
            except HTTPStatusError as exc:  # type: ignore[assignment]
                last_exc = exc
                status = exc.response.status_code
                # 5xx 认为可重试，4xx 直接失败
                if 500 <= status < 600 and attempt < self._max_retries:
                    delay = self._backoff_base * (2**attempt)
                    # 对可重试的 HTTP 异常，打印失败反馈
                    self._logger.warning(
                        "LLMJudge HTTP %s, retrying in %.1fs (attempt %d/%d)",
                        status,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise
            except RequestError as exc:  # 网络异常，可重试
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._backoff_base * (2**attempt)
                    # 对网络异常同样打印失败反馈
                    self._logger.warning(
                        "LLMJudge network error %r, retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise

        # 按理说不会到这里；保底抛出最后一次异常
        assert last_exc is not None
        raise last_exc

    def _score_once(self, prompt: str) -> tuple[float, float, str, str | None]:
        """
        执行一次 LLM 打分，并解析为 (score, max_score, raw_output, error_note)。
        若解析失败，则返回 (0.0, 5.0) 并在 note 中记录原因。
        """
        raw_output = ""
        note: str | None = None
        try:
            raw_output = self._call_llm(prompt)
            parsed = safe_parse_json(raw_output)
            score, max_score = compute_overall_score(parsed)
            return score, max_score, raw_output, None
        except Exception as exc:  # pragma: no cover - 主要对应网络/解析异常
            note = f"LLMJudge error: {exc!r}"
            # 解析失败时给一个保守的 0 分，max_score 使用默认 5
            return 0.0, 5.0, raw_output, note

    def judge(self, sample: Sample, candidate: Candidate) -> JudgeScore:
        input_data = self._build_input_data(sample)
        modules_block = self._build_modules_block(sample)
        history_input = self._build_history_input(sample)

        # 注意：score_prompt.py 中的模板本身包含大量 JSON 示例大括号 `{}`，
        # 若直接使用 str.format 会把这些内容误认为占位符，导致 KeyError。
        # 这里改为最简单且稳定的占位符替换逻辑，只替换我们约定的三个占位符：
        # {input_data} / {modules_block} / {answer}，而不会解析其它大括号。
        ground_prompt = (
            self._ground_prompt_tpl.replace("{modules_block}", modules_block)
            .replace("{history_input}", history_input)
            .replace("{input_data}", input_data)
            .replace("{answer}", candidate.response)
        )
        struct_prompt = (
            self._struct_prompt_tpl.replace("{modules_block}", modules_block)
            .replace("{history_input}", history_input)
            .replace("{input_data}", input_data)
            .replace("{answer}", candidate.response)
        )

        ground_score, ground_max, ground_raw, note1 = self._score_once(ground_prompt)
        struct_score, struct_max, struct_raw, note2 = self._score_once(struct_prompt)

        note = "; ".join(n for n in (note1, note2) if n) or None

        return JudgeScore(
            sample_id=sample.sample_id,
            candidate_id=candidate.candidate_id,
            model_type=candidate.model_type,
            model_name=candidate.model_name,
            judge=self.name,
            ground_score=ground_score,
            structure_score=struct_score,
            ground_max_score=ground_max,
            structure_max_score=struct_max,
            ground_raw=ground_raw or None,
            structure_raw=struct_raw or None,
            ground_prompt=ground_prompt,
            structure_prompt=struct_prompt,
            notes=note,
        )


def build_judges(configs: List[dict]) -> List[BaseJudge]:
    """
    根据配置构建 judge 列表。

    支持：
    - {"name": "llm", "base_url": "...", "api_key": "...", "model_name": "...", "timeout": 60, "max_retries": 2}
    """
    judges: List[BaseJudge] = []
    for cfg in configs:
        name = cfg.get("name")
        if name == "llm":
            # 支持从 .env / 环境变量读取敏感信息，避免直接写在配置文件中
            from src.judges.prompt_loader import load_prompts_from_config

            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            model_name = cfg.get("model_name")

            base_url_env = cfg.get("base_url_env")
            api_key_env = cfg.get("api_key_env")
            model_name_env = cfg.get("model_name_env")

            if not base_url and base_url_env:
                base_url = os.getenv(base_url_env)
            if not api_key and api_key_env:
                api_key = os.getenv(api_key_env)
            if not model_name and model_name_env:
                model_name = os.getenv(model_name_env)

            if not base_url or not api_key:
                raise ValueError(
                    "LLM judge requires 'base_url' and 'api_key' (or corresponding *_env names) in config."
                )
            if not model_name:
                raise ValueError(
                    "LLM judge requires 'model_name' or 'model_name_env' in config (and corresponding env value)."
                )

            ground_tpl, struct_tpl = load_prompts_from_config(cfg)

            judges.append(
                LLMJudge(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    timeout=float(cfg.get("timeout", 60.0)),
                    max_retries=int(cfg.get("max_retries", 29)),
                    backoff_base=float(cfg.get("backoff_base", 1.0)),
                    ground_prompt_tpl=ground_tpl,
                    struct_prompt_tpl=struct_tpl,
                )
            )
        else:
            raise ValueError(f"Unknown judge: {name}")
    return judges


