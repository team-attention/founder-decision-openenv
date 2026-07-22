from yc_founder_decision_env.baselines import evaluate


def test_baselines_use_same_held_out_split_and_revision() -> None:
    random_report = evaluate("random", 20260722)
    rule_report = evaluate("retrieval_rule", 20260722)
    assert random_report["split"] == rule_report["split"] == "held_out"
    assert random_report["split_size"] == rule_report["split_size"] == 8
    assert random_report["dataset_revision"] == rule_report["dataset_revision"] == "0.1.0"
    assert all(t["case_id"].startswith("ycfd-") for t in random_report["trajectories"])
