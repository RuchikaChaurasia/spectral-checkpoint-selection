"""Unit tests for the selection rules and the freeze step. Run from the repo root:  python -m pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

from src.spectral.rule_w import lowest_val_loss_epoch, rule_w

FLAT = {e: 2.5 for e in range(0, 31)}
DATA = Path(__file__).parent / "data"


def test_never_fires_selects_final_epoch():
    assert rule_w(FLAT, 2) == (30, False)


def test_fires_at_epochs_1_and_2_selects_epoch_1():
    assert rule_w({**FLAT, 1: 1.9, 2: 1.9}, 2) == (1, True)


def test_streak_starting_at_epoch_2_also_selects_epoch_1():
    assert rule_w({**FLAT, 2: 1.9, 3: 1.9}, 2) == (1, True)


def test_single_dip_only_patience_1_fires():
    a = {**FLAT, 10: 1.9}
    assert rule_w(a, 1) == (9, True)
    assert rule_w(a, 2) == (30, False)


def test_long_dip_all_patiences_agree():
    a = {**FLAT, 10: 1.9, 11: 1.9, 12: 1.9}
    assert [rule_w(a, k) for k in (1, 2, 3)] == [(9, True)] * 3


def test_dip_at_the_very_end():
    a = {**FLAT, 29: 1.9, 30: 1.9}
    assert rule_w(a, 2) == (28, True)
    assert rule_w(a, 3) == (30, False)


def test_failed_fit_breaks_a_streak():
    assert rule_w({**FLAT, 5: 1.9, 6: float("nan"), 7: 1.9}, 2) == (30, False)
    b = {**FLAT, 5: 1.9, 7: 1.9}
    del b[6]
    assert rule_w(b, 2) == (30, False)


def test_epoch_zero_is_ignored():
    assert rule_w({**FLAT, 0: 1.0}, 1) == (30, False)


def test_threshold_is_strict():
    assert rule_w({**FLAT, 4: 2.0, 5: 2.0}, 2) == (30, False)


def test_noisy_alpha_by_patience():
    a = {**FLAT, 4: 1.9, 8: 1.9, 9: 1.9}
    assert [rule_w(a, k) for k in (1, 2, 3)] == [(3, True), (7, True), (30, False)]


def test_val_loss_epoch_matches_run_info():
    # run_info.json of ft_b100_s42 says lowest_val_loss_epoch = 12
    assert lowest_val_loss_epoch(pd.read_csv(DATA / "ft_b100_s42_metrics.csv")) == 12


def test_val_loss_ignores_epoch_zero_and_takes_earliest_tie():
    m = pd.DataFrame({"epoch": [0, 1, 2, 3], "val_loss": [0.1, 0.5, 0.4, 0.4]})
    assert lowest_val_loss_epoch(m) == 2


def test_freeze_end_to_end(tmp_path):
    """Synthetic alpha logs for 12 runs: one run dips below 2 at epochs 10-12."""
    import json

    from src.spectral.freeze import main

    runs = [f"ft_b{b}_s{s}" for b in (10, 25, 50, 100) for s in (42, 123, 999)]
    rows = []
    for r in runs:
        b, s = r.split("_")[1][1:], r.split("_")[2][1:]
        for scope in ("all", "last_stage"):
            for e in range(0, 31):
                a = 1.9 if (r == "ft_b25_s123" and scope == "all" and e in (10, 11, 12)) else 3.0
                rows.append({"run": r, "budget": int(b), "seed": int(s), "epoch": e, "scope": scope, "mean_alpha": a})
        for prefix in ("ft", "lp"):
            d = tmp_path / "ckpt" / r.replace("ft_", f"{prefix}_")
            d.mkdir(parents=True)
            pd.read_csv(DATA / "ft_b100_s42_metrics.csv").to_csv(d / "metrics.csv", index=False)
    alpha = tmp_path / "mean_alpha.csv"
    pd.DataFrame(rows).to_csv(alpha, index=False)
    out = tmp_path / "frozen.json"
    main(["--alpha", str(alpha), "--runs-dir", str(tmp_path / "ckpt"), "--out", str(out)])
    sel = json.loads(out.read_text())["selections"]
    assert sel["ft_b25_s123"]["rule_w"]["all"]["2"] == {"epoch": 9, "fired": True}
    assert sel["ft_b25_s123"]["rule_w"]["last_stage"]["2"] == {"epoch": 30, "fired": False}
    assert sel["ft_b10_s42"]["rule_w"]["all"]["2"] == {"epoch": 30, "fired": False}
    assert sel["ft_b10_s42"]["val_loss_epoch"] == 12
    doc = json.loads(out.read_text())
    assert set(doc["linear_probe"]) == {r.replace("ft_", "lp_") for r in runs}
    assert doc["linear_probe"]["lp_b10_s42"]["val_loss_epoch"] == 12
    with pytest.raises(SystemExit):  # a second run must not overwrite the frozen file
        main(["--alpha", str(alpha), "--runs-dir", str(tmp_path / "ckpt"), "--out", str(out)])
