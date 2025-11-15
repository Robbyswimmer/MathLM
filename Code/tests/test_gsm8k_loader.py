"""Tests for GSM8k data loading utilities."""

from pathlib import Path

from mathlm.data import GSM8KExample, load_raw_split, save_examples


def test_load_raw_split_reads_jsonl(tmp_path: Path):
    raw_file = tmp_path / "gsm8k_train.jsonl"
    raw_file.write_text(
        "\n".join(
            [
                '{"id": "x", "question": "Q1?", "answer": "A1"}',
                '{"question": "Q2?", "answer": "A2"}',
            ]
        )
    )
    examples = load_raw_split(raw_file)
    assert len(examples) == 2
    assert examples[0].uid == "x"
    assert examples[1].uid.startswith("gsm8k_train")


def test_save_examples_round_trip(tmp_path: Path):
    examples = [
        GSM8KExample(uid="id1", question="What?", answer="Ans"),
        GSM8KExample(uid="id2", question="Another?", answer="Resp"),
    ]
    out = tmp_path / "processed" / "subset.jsonl"
    save_examples(examples, out)
    loaded = load_raw_split(out)
    assert [ex.uid for ex in loaded] == ["id1", "id2"]
