from __future__ import annotations

from src.scoring.parser import safe_parse_json, compute_overall_score


def test_safe_parse_json_basic():
    text = '{"checks": [{"score": 4}, {"score": 2}], "confidence": 0.8}'
    parsed = safe_parse_json(text)
    assert parsed["confidence"] == 0.8
    assert len(parsed["checks"]) == 2


def test_safe_parse_json_with_wrapping_text():
    text = "some log... {\"checks\": [{\"score\": 3}], \"confidence\": 1} ...tail"
    parsed = safe_parse_json(text)
    assert parsed["checks"][0]["score"] == 3


def test_compute_overall_score():
    parsed = {"checks": [{"score": 3}, {"score": 5}]}
    score, max_score = compute_overall_score(parsed)
    assert score == 4.0
    assert max_score == 5.0


