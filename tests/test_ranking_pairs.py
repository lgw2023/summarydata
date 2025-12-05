from __future__ import annotations

from src.ranking.ranker import rank_candidates, build_pairs


def test_rank_and_pairs_with_min_score_diff():
    rows = [
        {"sample_id": "1", "candidate_id": "c1", "aggregate_score": 4.5},
        {"sample_id": "1", "candidate_id": "c2", "aggregate_score": 3.0},
        {"sample_id": "2", "candidate_id": "c3", "aggregate_score": 4.1},
        {"sample_id": "2", "candidate_id": "c4", "aggregate_score": 4.05},
    ]
    ranked = rank_candidates(rows)

    # 对于 sample 1，分差 1.5，应保留
    # 对于 sample 2，分差 0.05，小于 0.1 阈值，应被过滤
    pairs = build_pairs(ranked, top_k=1, bottom_k=1, min_score_diff=0.1)
    assert any(p.sample_id == "1" for p in pairs)
    assert all(p.sample_id != "2" for p in pairs)


