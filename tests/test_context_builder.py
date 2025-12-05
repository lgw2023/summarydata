from __future__ import annotations

from src.data_loader.excel_loader import Sample
from src.data_loader.context_builder import build_context_text


def test_build_context_text_contains_sections():
    sample = Sample(
        sample_id="1",
        query="今天天气如何锻炼？",
        data="身高 170cm，体重 60kg",
        suggest="建议一周三次有氧运动",
        rag="一些知识库内容",
        service="- <course_1>\n课程描述",
        last_answer_phone="上一次的回答",
        reference_answers=None,
    )
    ctx = build_context_text(sample)
    # 核心区块存在
    assert "[个人数据]" in ctx
    assert "[专家建议]" in ctx
    assert "[知识库知识]" in ctx
    assert "[课程库]" in ctx
    assert "[对话历史]" in ctx
    assert "[用户提问]" in ctx
    # query 应该出现在末尾
    assert "今天天气如何锻炼" in ctx


