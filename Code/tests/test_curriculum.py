"""Tests for curriculum filtering utilities."""

from mathlm.data import CurriculumConfig, GSM8KExample, apply_curriculum


def _examples():
    return [
        GSM8KExample(uid="a", question="short", answer="1"),
        GSM8KExample(uid="b", question="much longer question text", answer="2"),
        GSM8KExample(uid="c", question="medium", answer="with more detail"),
    ]


def test_easy_curriculum_prefers_short_questions():
    cfg = CurriculumConfig(split="easy", max_problems=2)
    filtered = apply_curriculum(_examples(), cfg)
    assert [ex.uid for ex in filtered] == ["a", "c"]


def test_medium_curriculum_uses_question_and_answer_lengths():
    cfg = CurriculumConfig(split="medium", max_problems=1)
    filtered = apply_curriculum(_examples(), cfg)
    assert filtered[0].uid == "a"


def test_hard_curriculum_prioritizes_long_questions():
    cfg = CurriculumConfig(split="hard", max_problems=1)
    filtered = apply_curriculum(_examples(), cfg)
    assert filtered[0].uid == "b"


def test_full_curriculum_keeps_order():
    cfg = CurriculumConfig(split="full", max_problems=None)
    filtered = apply_curriculum(_examples(), cfg)
    assert [ex.uid for ex in filtered] == ["a", "b", "c"]
