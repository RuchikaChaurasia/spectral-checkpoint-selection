"""Tests for the evaluation metrics, the checkpoint list and the aggregation. Run:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.eval.aggregate import METHODS, main as aggregate_main, selection_table
from src.eval.metrics import accuracy, bootstrap_f1, bootstrap_indices, macro_f1, per_class_recall
from src.eval.real_world import mode_of, needed_checkpoints


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 65, 600)
        p = np.where(rng.random(600) < 0.6, y, rng.integers(0, 65, 600))
        ref = f1_score(y, p, labels=list(range(65)), average="macro", zero_division=0)
        assert macro_f1(y, p) == pytest.approx(ref, abs=1e-12)


def test_macro_f1_counts_missing_classes_as_zero():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 0, 1, 1])
    assert macro_f1(y, p) == pytest.approx(2 / 65)   # 2 perfect classes out of 65
    assert accuracy(y, p) == 1.0


def test_per_class_recall():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    r = per_class_recall(y, p)
    assert r[0] == 0.5 and r[1] == 1.0 and np.isnan(r[2])


def test_bootstrap_uses_the_same_resamples():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 65, 300); p = y.copy()
    idx = bootstrap_indices(300, 50, seed=0)
    assert np.allclose(bootstrap_f1(y, p, idx), bootstrap_f1(y, p, idx))
    assert (bootstrap_f1(y, p, idx) <= 1.0).all()


def _frozen(runs_ft, epoch_vl=12, epoch_w=30):
    sel = {r: {"val_loss_epoch": epoch_vl,
               "rule_w": {s: {str(k): {"epoch": epoch_w, "fired": False} for k in (1, 2, 3)} | {"min_mean_alpha": 3.4}
                          for s in ("all", "last_stage")}} for r in runs_ft}
    return {"primary": {"scope": "all", "patience": 2}, "selections": sel,
            "linear_probe": {r.replace("ft_", "lp_"): {"val_loss_epoch": 20} for r in runs_ft}}


def test_needed_checkpoints_lists_every_selected_epoch_once():
    f = _frozen(["ft_b10_s42"])
    assert needed_checkpoints(f) == [("ft_b10_s42", 12), ("ft_b10_s42", 30), ("lp_b10_s42", 20)]
    assert len(needed_checkpoints(f, all_epochs=True)) == 31 + 1
    assert mode_of("ft_b10_s42") == "finetune" and mode_of("lp_b10_s42") == "linear_probe"


def test_selection_table_reads_only_the_frozen_file():
    t = selection_table(_frozen(["ft_b10_s42", "ft_b10_s123"]))
    assert set(t.method) == set(METHODS)
    assert t[(t.method == METHODS[2])].epoch.eq(30).all() and t[t.method == METHODS[1]].epoch.eq(12).all()


def test_aggregate_end_to_end(tmp_path):
    """Synthetic predictions for 2 budgets x 2 seeds; the 'better' method must come out higher."""
    rng = np.random.default_rng(2)
    runs = [f"ft_b{b}_s{s}" for b in (10, 100) for s in (42, 123)]
    f = _frozen(runs)
    (tmp_path / "sel.json").write_text(json.dumps(f))
    n = 400
    y = rng.integers(0, 65, n)
    tgt = pd.DataFrame({"row_id": np.arange(n), "label": np.arange(n) % 65, "class": [f"c{i % 65}" for i in range(n)]})
    tgt.to_csv(tmp_path / "target.csv", index=False)
    y = tgt.label.to_numpy()
    acc = {("ft", 12): 0.9, ("ft", 30): 0.8, ("lp", 20): 0.7}
    for run, e in [(r, e) for r in runs for e in (12, 30)] + [(r.replace("ft_", "lp_"), 20) for r in runs]:
        a = acc[(run[:2], e)]
        p = np.where(rng.random(n) < a, y, rng.integers(0, 65, n))
        for split in ("real_world", "product_test"):
            d = tmp_path / "pred" / split; d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"row_id": np.arange(n), "label": y, "pred": p, "confidence": 0.9}).to_csv(
                d / f"{run}_e{e:02d}_{split}.csv", index=False)
    aggregate_main(["--frozen", str(tmp_path / "sel.json"), "--predictions", str(tmp_path / "pred"),
                    "--target", str(tmp_path / "target.csv"), "--out", str(tmp_path / "out"), "--reps", "50"])
    s = pd.read_csv(tmp_path / "out/tables/main_comparison.csv").set_index(["budget", "method"])
    assert s.loc[(10, METHODS[1]), "rw_f1_mean"] > s.loc[(10, METHODS[2]), "rw_f1_mean"] > s.loc[(10, METHODS[0]), "rw_f1_mean"]
    d = pd.read_csv(tmp_path / "out/tables/paired_differences.csv")
    rw = d[d.comparison.str.startswith("Rule W")]
    assert (rw.diff_rw_macro_f1 < 0).all() and (rw.ci_hi < 0).all()   # Rule W (epoch 30) is worse in this synthetic case
    assert (tmp_path / "out/figures/main_comparison.png").exists()
    assert (tmp_path / "out/tables/ablation_patience.csv").exists()
