from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import logging
import os
import time
import json
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_loader.excel_loader import Sample
from src.data_loader.context_builder import build_context_text
from src.utils.id_utils import make_candidate_id

try:  # 可选依赖：仅在使用真实 LLM 生成时才需要
    import httpx  # type: ignore
    from httpx import RequestError, HTTPStatusError  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore
    RequestError = Exception  # type: ignore
    HTTPStatusError = Exception  # type: ignore

from response_prompt_v2 import SYSTEMT_PROMPT_PHONE_GENERAL


# 每个「模型」在内部对不同样本进行并行生成时的默认最大并发数。
# - 可通过环境变量 GENERATOR_SAMPLE_WORKERS_PER_MODEL 覆盖；
# - 默认为 1，即保持原先的“同一模型内按样本串行”的行为，避免对线上服务产生突发压力。
try:
    DEFAULT_PER_MODEL_SAMPLE_WORKERS = max(
        4, int(os.getenv("GENERATOR_SAMPLE_WORKERS_PER_MODEL", "1"))
    )
except ValueError:  # 环境变量配置非法时回退为 1
    DEFAULT_PER_MODEL_SAMPLE_WORKERS = 2


# 统一占位回复文案：当底层 LLM 调用在重试后仍然失败时使用。
FALLBACK_RESPONSE_DEFAULT = (
    "【系统提示】本条回复在调用模型时出现异常，已使用占位文本，请在标注或训练时忽略本候选。"
)


@dataclass
class Candidate:
    """
    统一的候选回复结构，基本对齐 TASK.md 2.3 中的定义。
    """

    sample_id: str
    candidate_id: str
    model_type: str  # experimental | open_source | closed_source | reference
    model_name: str
    response: str
    gen_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "response": self.response,
            "gen_config": self.gen_config,
        }


class BaseGenerator:
    """
    所有生成器的抽象基类。

    - name: 配置中使用的标识符（如 "echo"、"experimental" 等）
    - model_type: 用于区分 experimental / open_source / closed_source / reference
    - model_name: 具体模型名称（如内部实验模型名、LLM 名称等）
    """

    name: str = "base"
    model_type: str = "experimental"
    model_name: str = "base"

    def __init__(self, gen_config: Dict[str, Any] | None = None) -> None:
        self._gen_config = gen_config or {}

    def _build_candidate(self, sample: Sample, response: str, index: int | None = None) -> Candidate:
        candidate_id = make_candidate_id(sample.sample_id, self.model_name, index=index)
        return Candidate(
            sample_id=sample.sample_id,
            candidate_id=candidate_id,
            model_type=self.model_type,
            model_name=self.model_name,
            response=response,
            gen_config=self._gen_config,
        )

    def generate(self, sample: Sample) -> Candidate:  # pragma: no cover - interface
        raise NotImplementedError


class _OpenAIChatGenerator(BaseGenerator):
    """
    基于 OpenAI 兼容接口的通用生成器基类。

    约定配置字段（通过 PipelineConfig 传入 gen_config）：
    - base_url / api_key：直接配置；
    - base_url_env / api_key_env：从环境变量读取（推荐，配合 .env）；
    - model_name：模型名称（必填，若未显式给出则使用各子类默认值）；
    - temperature/top_p/max_tokens：采样参数，可选；
    - timeout/max_retries/backoff_base：HTTP 超时与重试策略。
    """

    def __init__(self, model_name: str, gen_config: Dict[str, Any] | None = None) -> None:
        super().__init__(gen_config=gen_config)
        self.model_name = model_name

        cfg = gen_config or {}
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")

        base_url_env = cfg.get("base_url_env")
        api_key_env = cfg.get("api_key_env")

        if not base_url and base_url_env:
            base_url = os.getenv(str(base_url_env))
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env))

        self._base_url: str | None = str(base_url).rstrip("/") if base_url else None
        self._api_key: str | None = str(api_key) if api_key else None

        self._timeout: float = float(cfg.get("timeout", 60.0))
        self._max_retries: int = int(cfg.get("max_retries", 29))
        self._backoff_base: float = float(cfg.get("backoff_base", 1.0))
        self._logger = logging.getLogger(__name__)

        # 仅在 httpx 可用且 endpoint 配置完整时才真正启用远程调用
        self._client: httpx.Client | None
        if httpx is not None and self._base_url and self._api_key:  # type: ignore[truthy-function]
            self._client = httpx.Client(timeout=self._timeout)  # type: ignore[call-arg]
        else:
            self._client = None

        # 系统提示词：默认使用“小艺”健康管家人设
        self._system_prompt: str = str(cfg.get("system_prompt") or SYSTEMT_PROMPT_PHONE_GENERAL)

        # 采样参数（也写入 gen_config 中，便于后续追踪）
        self._temperature: float = float(cfg.get("temperature", 0.7))
        self._top_p: float = float(cfg.get("top_p", 0.9))
        # self._max_tokens: int = int(cfg.get("max_tokens", 4096))

    # ===== OpenAI 兼容 HTTP 调用 =====
    def _build_messages(self, sample: Sample) -> List[Dict[str, str]]:
        """
        根据 response_prompt 中约定的结构构建 user 消息：
        - 使用 data_loader.context_builder.build_context_text 拼出上下文；
        - system 消息采用“小艺”健康管家系统提示词。
        """
        context = build_context_text(sample)
        user_content = (
            f"{context}\n\n"
            "请结合以上 [个人数据]、[专家建议]、[知识库知识]、[课程库] 与 [对话历史]，"
            "以“小艺”健康管家的身份，用中文为用户提供结构清晰、可执行的运动健康建议。"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用 OpenAI 兼容接口并返回模型输出文本。

        设计约定：
        - 若 endpoint 未配置（缺少 base_url / api_key / httpx），立即抛出异常，
          视为配置错误，由上层决定是否继续整个任务；
        - 若 HTTP / 网络请求或返回内容解析过程中出现异常，则在完成重试后
          **不再向外抛出异常，而是返回一个占位字符串**，以避免单条样本
          或单个模型的失败中断整个大任务。
        """
        if self._client is None or not self._base_url or not self._api_key:
            raise RuntimeError("LLM endpoint is not configured for this generator.")

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
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            # "temperature": self._temperature,
            # "top_p": self._top_p,
            # "max_tokens": self._max_tokens,
        }
        if "qwen" in self.model_name.lower() and "aliyun" in url.lower():
            payload["enable_thinking"] = False

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            start_ts = time.time()
            try:
                # 每一次底层 LLM HTTP 调用开始时输出反馈，便于在 pipeline 中观察进度
                # self._logger.info(
                #     "LLM generate call start | model=%s | attempt=%d/%d",
                #     self.model_name,
                #     attempt + 1,
                #     self._max_retries + 1,
                # )
                # 若为指定的调试模型（如 qwen3-30b-a3b / qwen3-32b），在第一次尝试时打印一段可直接复制的
                # curl 命令（包含 URL / 头部 / 完整 JSON Body），便于在终端手动重放请求进行排查。
                # if attempt == 0 and self.model_name.lower() in {"qwen3-30b-a3b", "qwen3-32b"}:
                #     try:
                #         body_str = json.dumps(payload, ensure_ascii=False)
                #         curl_cmd = (
                #             "curl -X POST \\\n"
                #             f"  '{url}' \\\n"
                #             "  -H 'Content-Type: application/json' \\\n"
                #             f"  -H 'Authorization: Bearer {self._api_key}' \\\n"
                #             "  -d @- << 'EOF'\n"
                #             f"{body_str}\n"
                #             "EOF"
                #         )
                #         self._logger.error("LLM debug curl command (copy & run in shell):\n%s", curl_cmd)
                #     except Exception as log_exc:  # pragma: no cover - 防御性日志
                #         self._logger.warning(
                #             "Failed to log debug request for model=%s: %r",
                #             self.model_name,
                #             log_exc,
                #         )

                resp = self._client.post(url, headers=headers, json=payload)  # type: ignore[union-attr]
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # 防御性校验：确保返回内容为非空字符串
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty or invalid content returned from LLM.")
                duration = time.time() - start_ts
                # 每一次 LLM 调用成功后的反馈日志
                self._logger.info(
                    "LLM generate call success | model=%s | attempt=%d/%d | elapsed=%.2fs | content_len=%d",
                    self.model_name,
                    attempt + 1,
                    self._max_retries + 1,
                    duration,
                    len(content),
                )
                return content
            except HTTPStatusError as exc:  # type: ignore[assignment]
                last_exc = exc
                status = exc.response.status_code
                # 5xx 认为可重试，4xx 直接失败并回退到占位回复
                if 500 <= status < 600 and attempt < self._max_retries:
                    # 统一在重试前至少等待 3 秒，必要时可通过 backoff_base 继续拉长等待时间
                    delay = max(3.0, self._backoff_base * (2**attempt))
                    # 对可重试的 HTTP 异常，同样输出一次本轮调用的失败反馈
                    self._logger.warning(
                        "Generator HTTP %s on model=%s url=%s, retrying in %.1fs (attempt %d/%d)",
                        status,
                        self.model_name,
                        url,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue
                # 4xx 或最后一次 5xx：跳出重试循环，使用占位回复
                break
            except RequestError as exc:  # 网络异常，可重试
                last_exc = exc
                if attempt < self._max_retries:
                    # 统一在重试前至少等待 3 秒，必要时可通过 backoff_base 继续拉长等待时间
                    delay = max(3.0, self._backoff_base * (2**attempt))
                    # 对网络异常同样输出失败反馈
                    self._logger.warning(
                        "Generator network error on model=%s url=%s error=%r, retrying in %.1fs (attempt %d/%d)",
                        self.model_name,
                        url,
                        exc,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    continue
                # 已经是最后一次重试：跳出循环，使用占位回复
                break
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                # 返回 JSON 结构异常或 content 解析失败，直接视为不可重试错误
                last_exc = exc
                # 同样给出一次失败反馈
                self._logger.error(
                    "Generator parse error on model=%s (attempt %d/%d): %r",
                    self.model_name,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                break

        # 若走到此处，说明在重试后仍然失败，返回占位回复而不是抛出异常
        self._logger.error(
            "LLM call failed after %d attempts for model %s, using fallback response. Last error: %r",
            self._max_retries + 1,
            self.model_name,
            last_exc,
        )
        # 支持通过 gen_config 覆盖占位文案（可选）
        fallback = (
            self._gen_config.get("fallback_response") if hasattr(self, "_gen_config") else None  # type: ignore[attr-defined]
        )
        if not isinstance(fallback, str) or not fallback.strip():
            fallback = FALLBACK_RESPONSE_DEFAULT
        return fallback

    def _generate_via_llm(self, sample: Sample) -> str:
        """
        高层封装：构建 messages 并调用 LLM。

        说明：
        - 若底层 HTTP / 解析逻辑失败，则 `_call_llm` 会直接返回占位回复字符串，
          因此此处不再向外抛出异常，保证逐条样本生成的健壮性。
        """
        messages = self._build_messages(sample)
        return self._call_llm(messages)


class ExperimentalGenerator(_OpenAIChatGenerator):
    """
    内部实验模型生成器：必须正确配置 OpenAI 兼容接口，否则直接报错。
    """

    name = "experimental"
    model_type = "experimental"

    def __init__(self, model_name: str = "exp-model", gen_config: Dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, gen_config=gen_config)

    def generate(self, sample: Sample) -> Candidate:
        if self._client is None or not self._base_url or not self._api_key:
            raise RuntimeError(
                f"ExperimentalGenerator for model '{self.model_name}' is not properly configured "
                "(missing httpx client / base_url / api_key)."
            )
        content = self._generate_via_llm(sample)
        return self._build_candidate(sample, response=content)


class OpenSourceGenerator(_OpenAIChatGenerator):
    """
    开源模型生成器：依赖 OpenAI 兼容接口（如 DeepInfra 等），不再提供本地占位实现。
    """

    name = "open_source"
    model_type = "open_source"

    def __init__(self, model_name: str = "open-source-llm", gen_config: Dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, gen_config=gen_config)

    def generate(self, sample: Sample) -> Candidate:
        if self._client is None or not self._base_url or not self._api_key:
            raise RuntimeError(
                f"OpenSourceGenerator for model '{self.model_name}' is not properly configured "
                "(missing httpx client / base_url / api_key)."
            )
        content = self._generate_via_llm(sample)
        return self._build_candidate(sample, response=content)


class ClosedSourceGenerator(_OpenAIChatGenerator):
    """
    闭源模型生成器：依赖 OpenAI 兼容接口（如 DeepSeek / Claude 等），不再提供本地占位实现。
    """

    name = "closed_source"
    model_type = "closed_source"

    def __init__(self, model_name: str = "closed-source-llm", gen_config: Dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, gen_config=gen_config)

    def generate(self, sample: Sample) -> Candidate:
        if self._client is None or not self._base_url or not self._api_key:
            raise RuntimeError(
                f"ClosedSourceGenerator for model '{self.model_name}' is not properly configured "
                "(missing httpx client / base_url / api_key)."
            )
        content = self._generate_via_llm(sample)
        return self._build_candidate(sample, response=content)


class ReferenceGenerator(BaseGenerator):
    """
    将 a_answer / b_answer 等参考答案作为 reference 类型候选。
    为简单起见，目前仅取第一条有效参考答案。
    """

    name = "reference"
    model_type = "reference"

    def __init__(self, model_name: str = "reference", gen_config: Dict[str, Any] | None = None) -> None:
        """
        参考答案生成器。

        - model_name：用于区分不同参考答案来源（如 ref_a / ref_b），会写入 candidate_id 与 model_name 字段；
        - gen_config.answer_key：可选，限定只从 reference_answers 中的某个 key 取值（如 "a_answer" 或 "b_answer"）。
          若未指定，则回退为按 ("a_answer", "b_answer") 顺序取第一条非空。
        """
        super().__init__(gen_config=gen_config)
        self.model_name = model_name

        cfg = gen_config or {}
        key = cfg.get("answer_key")
        self._answer_key: str | None = str(key) if key else None

    def generate(self, sample: Sample) -> Candidate:
        reference_text = None
        if sample.reference_answers:
            if self._answer_key:
                # 显式指定只使用某一列参考答案（如 a_answer 或 b_answer）
                reference_text = sample.reference_answers.get(self._answer_key)
            else:
                # 未指定时，兼容旧逻辑：按固定顺序取第一条非空
                for key in ("a_answer", "b_answer"):
                    if key in sample.reference_answers and sample.reference_answers[key]:
                        reference_text = sample.reference_answers[key]
                        break

        if not reference_text:
            reference_text = "（无参考答案）"

        label = self._answer_key or self.model_name
        content = reference_text
        return self._build_candidate(sample, response=content)


def build_generators(configs: List[dict]) -> List[BaseGenerator]:
    """
    根据配置构建生成器列表。

    配置示例：
    - {"name": "experimental", "model_name": "exp_v1", "temperature": 0.7}
    - {"name": "open_source", "model_name": "Qwen-7B"}
    - {"name": "closed_source", "model_name": "gpt-4.1-mini"}
    - {"name": "reference"}
    """
    generators: List[BaseGenerator] = []
    for cfg in configs:
        name = cfg.get("name")
        extra_cfg: Dict[str, Any] = {k: v for k, v in cfg.items() if k != "name"}

        if name == "experimental":
            generators.append(
                ExperimentalGenerator(
                    model_name=extra_cfg.get("model_name", "exp-model"),
                    gen_config=extra_cfg,
                )
            )
        elif name == "open_source":
            generators.append(
                OpenSourceGenerator(
                    model_name=extra_cfg.get("model_name", "open-source-llm"),
                    gen_config=extra_cfg,
                )
            )
        elif name == "closed_source":
            generators.append(
                ClosedSourceGenerator(
                    model_name=extra_cfg.get("model_name", "closed-source-llm"),
                    gen_config=extra_cfg,
                )
            )
        elif name == "reference":
            generators.append(
                ReferenceGenerator(
                    model_name=extra_cfg.get("model_name", "reference"),
                    gen_config=extra_cfg,
                )
            )
        else:
            raise ValueError(f"Unknown generator: {name}")

    return generators


def _run_generator_over_samples(generator: BaseGenerator, samples: Sequence[Sample]) -> List[Candidate]:
    """
    辅助函数：对同一模型批量生成，用于在线程池中并发调度。

    行为说明：
    - 默认情况下（DEFAULT_PER_MODEL_SAMPLE_WORKERS == 1），保持旧逻辑：
      在单线程中依次对每个样本调用一次 generate；
    - 当将环境变量 GENERATOR_SAMPLE_WORKERS_PER_MODEL 设置为 > 1 时，
      会在“同一模型内部”对不同样本做并行生成。
    """
    if not samples:
        return []

    # 计算该模型内部的样本级并发数（至少为 1，且不超过样本总数）
    per_model_workers = max(1, min(DEFAULT_PER_MODEL_SAMPLE_WORKERS, len(samples)))

    # 串行路径：保持与历史实现完全一致的行为
    if per_model_workers == 1:
        results: List[Candidate] = []
        for sample in samples:
            results.append(generator.generate(sample))
        return results

    # 并行路径：在同一模型内部对不同样本进行并发生成
    logger = logging.getLogger(__name__)
    results: List[Candidate] = []
    with ThreadPoolExecutor(max_workers=per_model_workers) as executor:
        future_to_sample = {
            executor.submit(generator.generate, sample): sample for sample in samples
        }
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                cand = future.result()
                results.append(cand)
            except Exception as exc:  # pragma: no cover - 防御性日志
                logger.exception(
                    "Generator %s (%s) failed on sample %s during per-model parallel generation: %r",
                    getattr(generator, "model_name", getattr(generator, "name", "unknown")),
                    getattr(generator, "model_type", "unknown"),
                    getattr(sample, "sample_id", "unknown"),
                    exc,
                )

    return results


def generate_candidates(
    samples: Sequence[Sample],
    generators: Sequence[BaseGenerator],
    max_workers: int | None = None,
) -> List[Candidate]:
    """
    在「同一模型内按样本批量生成」的前提下，对「不同模型」进行异步并发执行。

    - 对于每个 generator，会在其各自的线程中依次处理所有样本；
    - 不同 generator 之间通过 ThreadPoolExecutor 并发执行，充分利用 I/O 等待时间；
    - 生成结果仍然是扁平的 Candidate 列表，后续可按 sample_id 分组。
    """
    if not generators or not samples:
        return []

    logger = logging.getLogger(__name__)
    # 限制最大并发度，默认不超过模型数量
    worker_count = max(1, min(len(generators), max_workers or len(generators)))

    all_candidates: List[Candidate] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_gen = {
            executor.submit(_run_generator_over_samples, gen, samples): gen for gen in generators
        }
        for future in as_completed(future_to_gen):
            gen = future_to_gen[future]
            try:
                gen_candidates = future.result()
                all_candidates.extend(gen_candidates)
            except Exception as exc:  # pragma: no cover - 防御性日志
                logger.exception(
                    "Generator %s (%s) failed during batch generation: %r",
                    getattr(gen, "model_name", getattr(gen, "name", "unknown")),
                    getattr(gen, "model_type", "unknown"),
                    exc,
                )

    return all_candidates
