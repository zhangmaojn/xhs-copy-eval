from pathlib import Path

from xhs_eval.pipeline import run_pipeline

ROOT = Path(__file__).parents[1]


def test_offline_pipeline(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    result = run_pipeline(ROOT / "configs/demo.yaml")
    assert result["sample_count"] == 18
    assert 0 < result["judge_pass_rate"] < 1
    assert Path(result["report"]).is_file()
