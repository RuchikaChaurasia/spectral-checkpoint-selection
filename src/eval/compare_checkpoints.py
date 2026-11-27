from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def prediction_path(root, run, epoch):
    return root / "real_world" / f"{run}_e{epoch:02d}_real_world.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="results/selections/frozen_selections.json")
    parser.add_argument("--predictions", default="results/predictions")
    parser.add_argument("--target", default="configs/splits/target_real_world.csv")
    parser.add_argument(
        "--out",
        default="results/tables/rule_w_vs_val_loss.csv"
    )
    parser.add_argument(
        "--summary-out",
        default="results/tables/rule_w_vs_val_loss_summary.csv"
    )
    args = parser.parse_args()

    frozen = json.loads(Path(args.frozen).read_text())
    pred_root = Path(args.predictions)

    target = pd.read_csv(args.target)
    class_names = dict(zip(target["label"], target["class"]))

    rows = []

    # Only fine-tuning runs are compared.
    for run, info in frozen["selections"].items():
        val_epoch = int(info["val_loss_epoch"])
        rule_epoch = int(info["rule_w"]["all"]["2"]["epoch"])

        val_path = prediction_path(pred_root, run, val_epoch)
        rule_path = prediction_path(pred_root, run, rule_epoch)

        if not val_path.exists():
            raise FileNotFoundError(val_path)
        if not rule_path.exists():
            raise FileNotFoundError(rule_path)

        val = pd.read_csv(val_path)
        rule = pd.read_csv(rule_path)

        val = val[["row_id", "label", "pred", "confidence"]].rename(
            columns={
                "label": "true_label",
                "pred": "val_loss_pred",
                "confidence": "val_loss_confidence",
            }
        )

        rule = rule[["row_id", "label", "pred", "confidence"]].rename(
            columns={
                "label": "rule_true_label",
                "pred": "rule_w_pred",
                "confidence": "rule_w_confidence",
            }
        )

        merged = val.merge(rule, on="row_id", how="inner")

        if len(merged) != len(val) or len(merged) != len(rule):
            raise ValueError(f"Row mismatch for {run}")

        if not (merged["true_label"] == merged["rule_true_label"]).all():
            raise ValueError(f"True-label mismatch for {run}")

        val_correct = merged["val_loss_pred"] == merged["true_label"]
        rule_correct = merged["rule_w_pred"] == merged["true_label"]

        merged["category"] = np.select(
            [
                (~val_correct) & rule_correct,
                val_correct & (~rule_correct),
                val_correct & rule_correct,
            ],
            [
                "Rule W fixes",
                "Rule W breaks",
                "Both correct",
            ],
            default="Both wrong",
        )

        budget = int(run.split("_")[1][1:])
        seed = int(run.split("_")[2][1:])

        merged.insert(0, "run", run)
        merged.insert(1, "budget", budget)
        merged.insert(2, "seed", seed)
        merged["val_loss_epoch"] = val_epoch
        merged["rule_w_epoch"] = rule_epoch
        merged["true_class"] = merged["true_label"].map(class_names)
        merged["val_loss_class"] = merged["val_loss_pred"].map(class_names)
        merged["rule_w_class"] = merged["rule_w_pred"].map(class_names)
        merged["prediction_changed"] = (
            merged["val_loss_pred"] != merged["rule_w_pred"]
        )
        merged["confidence_difference"] = (
            merged["rule_w_confidence"] - merged["val_loss_confidence"]
        )

        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    summary = (
        result.groupby(["budget", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(["budget", "category"])
    )

    summary.to_csv(args.summary_out, index=False)

    print(f"Wrote: {output}")
    print(f"Wrote: {args.summary_out}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()