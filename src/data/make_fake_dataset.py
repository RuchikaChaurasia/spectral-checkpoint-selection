"""Make a tiny fake Office-Home-shaped dataset for offline testing (owner: Disha).

It has the same columns as the real one (image, domain, label), the same 65 class
names and the same 4 domain names, plus planted problems the audit must catch:
near-duplicate pairs in Product, one Product image copied into Real World, and
one corrupt image.

    python -m src.data.make_fake_dataset /tmp/fake_officehome
    OFFICEHOME_LOCAL_PATH=/tmp/fake_officehome python -m src.data.audit --config <test config>
"""
from __future__ import annotations

import argparse
import io

import numpy as np
from PIL import Image, ImageDraw

CLASSES = [
    "Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket", "Calculator", "Calendar",
    "Candles", "Chair", "Clipboards", "Computer", "Couch", "Curtains", "Desk_Lamp", "Drill", "Eraser",
    "Exit_Sign", "Fan", "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork", "Glasses", "Hammer",
    "Helmet", "Kettle", "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker", "Monitor", "Mop", "Mouse",
    "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil", "Postit_Notes", "Printer", "Push_Pin",
    "Radio", "Refrigerator", "Ruler", "Scissors", "Screwdriver", "Shelf", "Sink", "Sneakers", "Soda",
    "Speaker", "Spoon", "TV", "Table", "Telephone", "ToothBrush", "Toys", "Trash_Can", "Webcam",
]


def _img(rng, c, clutter):
    w, h = int(rng.integers(160, 320)), int(rng.integers(160, 320))
    bg = tuple(int(v) for v in rng.integers(0, 255, 3)) if clutter else (255, 255, 255)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    for _ in range(3 + int(clutter) * 6):   # random distractor blobs make every image different
        x, y = rng.integers(0, w), rng.integers(0, h)
        s = int(rng.integers(10, 60))
        d.ellipse([x, y, x + s, y + s], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    hue = (c * 37) % 255
    color = (hue, (hue * 3) % 255, (hue * 7) % 255)
    k = 3 + c % 5
    cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.3, 0.7)
    r, rot = min(w, h) * rng.uniform(0.15, 0.3), rng.uniform(0, 2 * np.pi)
    pts = [(cx + r * np.cos(2 * np.pi * i / k + rot), cy + r * np.sin(2 * np.pi * i / k + rot)) for i in range(k)]
    d.polygon(pts, fill=color)
    return im


def _bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main(argv=None):
    import datasets

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, nargs=4, default=[12, 6, 2, 2],
                    metavar=("PRODUCT", "REALWORLD", "ART", "CLIPART"))
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    images, domains, labels = [], [], []
    for dom, k, clutter in zip(["Product", "Real World", "Art", "Clipart"], args.per_class,
                               [False, True, True, False]):
        for c in range(len(CLASSES)):
            for _ in range(k):
                images.append({"bytes": _bytes(_img(rng, c, clutter)), "path": None})
                domains.append(dom)
                labels.append(c)
    product = [i for i, d in enumerate(domains) if d == "Product"]
    # planted problems: 5 near-duplicate pairs in Product, one Product image in Real World, one corrupt file
    for j in range(5):
        src = product[j * 20]
        im = Image.open(io.BytesIO(images[src]["bytes"])).convert("RGB")
        images.append({"bytes": _bytes(im.resize((im.width - 4, im.height - 4))), "path": None})
        domains.append("Product")
        labels.append(labels[src])
    rw = domains.index("Real World")
    images[rw] = dict(images[product[7]])
    labels[rw] = labels[product[7]]
    images[product[-1]] = {"bytes": b"not a jpeg", "path": None}

    feats = datasets.Features({
        "image": datasets.Image(),
        "domain": datasets.Value("string"),
        "label": datasets.ClassLabel(names=CLASSES),
    })
    ds = datasets.Dataset.from_dict({"image": images, "domain": domains, "label": labels}, features=feats)
    ds.save_to_disk(args.out)
    print(f"Fake dataset with {len(ds)} rows -> {args.out}")


if __name__ == "__main__":
    main()
