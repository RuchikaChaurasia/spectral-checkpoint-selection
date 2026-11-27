from pathlib import Path
import pandas as pd


INPUT = Path("results/tables/rule_w_vs_val_loss.csv")
OUTPUT = Path("results/tables/error_examples_100.csv")


df = pd.read_csv(INPUT)
df = df[df["budget"] == 100].copy()


def choose_examples(category, count, sort_columns):
    subset = df[df["category"] == category].copy()
    subset = subset.sort_values(sort_columns, ascending=False)

    # Avoid selecting the same image more than once.
    subset = subset.drop_duplicates("row_id")

    return subset.head(count)


fixes = choose_examples(
    "Rule W fixes",
    2,
    ["rule_w_confidence", "val_loss_confidence"],
)

breaks = choose_examples(
    "Rule W breaks",
    2,
    ["val_loss_confidence", "rule_w_confidence"],
)

both_correct = choose_examples(
    "Both correct",
    1,
    ["rule_w_confidence", "val_loss_confidence"],
)

both_wrong = choose_examples(
    "Both wrong",
    1,
    ["val_loss_confidence", "rule_w_confidence"],
)

examples = pd.concat(
    [fixes, breaks, both_correct, both_wrong],
    ignore_index=True,
)

columns = [
    "run",
    "row_id",
    "true_class",
    "val_loss_class",
    "rule_w_class",
    "val_loss_confidence",
    "rule_w_confidence",
    "category",
]

examples = examples[columns]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
examples.to_csv(OUTPUT, index=False)

print(f"Wrote: {OUTPUT}")
print()
print(examples.to_string(index=False))