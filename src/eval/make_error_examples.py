from pathlib import Path
import textwrap

import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


INPUT = Path("results/tables/error_examples_100.csv")
OUT_DIR = Path("results/figures/error_examples_100")
CONTACT_SHEET = Path("results/figures/error_examples_100.png")

REVISION = "d430d44779e929b335be368ab3456484e5de36bb"


examples = pd.read_csv(INPUT)

dataset = load_dataset(
    "flwrlabs/office-home",
    revision=REVISION,
    split="train",
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)

font = ImageFont.load_default()
cards = []

for i, row in examples.iterrows():
    image = dataset[int(row["row_id"])]["image"].convert("RGB")
    image.thumbnail((360, 270))

    card = Image.new("RGB", (400, 390), "white")
    card.paste(image, ((400 - image.width) // 2, 10))

    draw = ImageDraw.Draw(card)

    text = (
        f"Category: {row['category']}\n"
        f"True: {row['true_class']}\n"
        f"Val-loss: {row['val_loss_class']}\n"
        f"Rule W: {row['rule_w_class']}\n"
        f"Run: {row['run']}, row_id: {row['row_id']}"
    )

    y = 295
    for line in text.splitlines():
        draw.text((10, y), line, fill="black", font=font)
        y += 16

    filename = OUT_DIR / f"example_{i + 1}_{row['category'].replace(' ', '_')}.png"
    card.save(filename)
    cards.append(card)

sheet = Image.new("RGB", (800, 1170), "#dddddd")

for i, card in enumerate(cards):
    x = (i % 2) * 400
    y = (i // 2) * 390
    sheet.paste(card, (x, y))

sheet.save(CONTACT_SHEET)

print(f"Wrote individual images to: {OUT_DIR}")
print(f"Wrote contact sheet to: {CONTACT_SHEET}")