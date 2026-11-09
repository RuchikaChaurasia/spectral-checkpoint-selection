# Dataset Card: Office-Home (Group 9)

*Owner: Disha Rawat. Numbers come from `results/data_audit/` and `configs/splits/split_summary.json`,
written by `src/data/audit.py`, `src/data/make_splits.py` and `notebooks/dataset_card_review.ipynb`.*

## 1. Source and version
| Item | Value |
|---|---|
| Dataset | Office-Home (Venkateswara et al., CVPR 2017) |
| Original source | https://www.hemanthdv.org/officeHomeDataset.html |
| Copy used | Hugging Face `flwrlabs/office-home` (Flower Labs) |
| Pinned revision | `d430d44779e929b335be368ab3456484e5de36bb` (last changed 29 Aug 2024) |
| Format | Parquet; one split named `train` holding all 15,588 images; columns `image`, `domain`, `label` |
| Download size | about 1.16 GB (about 1.3 GB loaded) |
| Accessed | 7 November 2026 (pinned revision d430d44); the proposal cites first access on 24 October 2026 |
| Audit run | 2026-11-07T16:21:35+00:00 |

**Access:**
```python
from datasets import load_dataset
ds = load_dataset("flwrlabs/office-home", split="train",
                  revision="d430d44779e929b335be368ab3456484e5de36bb")
```

## 2. License and intended use
- License on the hub: **other: fair-use notice** pointing to the original Office-Home website. It is not an open-source license.
- Use: research and teaching in this course only. The repository holds row IDs, labels, metric summaries and a few report figures, not the dataset.
- Citation: Venkateswara et al. (2017); the hub card also asks that Flower (Beutel et al., 2020) be cited when the Flower tooling is used.

## 3. Task, inputs and targets
- **Task:** 65-way single-label object classification.
- **Input:** one image, converted to RGB at load time, resized and cropped to 224 × 224, ImageNet-normalised.
- **Target:** integer label 0–64 (`Alarm_Clock` = 0 … `Webcam` = 64), stored as a `ClassLabel`.
- **Annotation format:** one class label per image; no boxes, no file names.
- **Image ID:** the row number in the pinned revision (`row_id`). It is stable only because the revision is pinned.

## 4. Domains and counts
| Domain | Proposal | Audit | Check | Role in this project |
|---|---|---|---|---|
| Product | 4,439 | 4,439 | PASS | Source: train / validation / source test |
| Real World (`"Real World"`, with a space) | 4,357 | 4,357 | PASS | Held-out target: final evaluation only |
| Art | 2,427 | 2,427 | PASS | Optional extra target |
| Clipart | 4,365 | 4,365 | PASS | Not used |
| **Total** | 15,588 | 15,588 | PASS | |

**Classes:** 65 classes (PASS); no class is empty in any domain.
**Class distribution (Product):** 38 images (Postit_Notes) to 99 (Backpack) per class,
median 67. Full table in the appendix; figure `results/figures/product_class_distribution.png`.

**Image sizes** (width and height in pixels, min–median–max):

| Domain | Images | Width | Height | Non-RGB files |
|---|---|---|---|---|
| Art | 2,427 | 85–800–4438 | 117–678–2686 | 0 |
| Clipart | 4,365 | 18–426–2400 | 4–431–2400 | 0 |
| Product | 4,439 | 42–500–2560 | 42–500–2560 | 0 |
| Real World | 4,357 | 80–800–6000 | 63–662–6500 | 0 |

## 5. Splits (fixed once)
| Split | Images | Use |
|---|---|---|
| Product train | 3,551 | Training; nested label budgets |
| Product validation | 444 | Checkpoint selection (validation loss) and the LR pilot |
| Product test | 444 | In-domain reference |
| Real World | 4,357 | Final evaluation only, after the 16 Nov freeze |

- Class-stratified with split seed **0**; largest-remainder quotas make the totals exact.
  Every class has at least 3 validation and 4 test images.
- Near-duplicate groups (pHash Hamming distance ≤ 4) are kept inside one split;
  **0 groups cross splits**.
- Label budgets for training seeds 42, 123, 999: 355 / 888 / 1,776 / 3,551 images
  (10 / 25 / 50 / 100%) for every seed; nested (10% ⊂ 25% ⊂ 50% ⊂ 100%);
  at least 3 images per class at 10%.
- SHA-256 of `product_split.csv`: `df7fa0d4006df84ef427b53b41154454d383f2d1e99cde1986239ffeb9c5ba6c` (all file hashes in `split_summary.json`).

## 6. Quality audit
- **Class names:** 65 classes, no `Clock` class. The original Office-Home CSV lists an empty
  `Clock` folder; the Hugging Face copy dropped it, and `Alarm_Clock` is the real class.
- **Unreadable files:** 0.
- **Colour modes:** 0 images are not stored as RGB (none); all are converted to RGB at load time.
- **Odd shapes:** 118 images with a side under 64 px; 393 with an aspect ratio above 3.
- **Near-duplicates inside Product:** 422 pairs (268 with identical hashes) forming
  284 groups that hold 628 images (14% of Product);
  largest group 8 images. Each group stays inside one split, so no near-duplicate of a training
  image (as detected by pHash) is in Product validation or test.
- **Same picture, different labels:** the hash grouped 16 sets of Product images (38 images)
  that carry more than one label. Checked by eye and by pixel correlation: **10 groups (22 images)
  are the same picture filed under two classes**: Computer / Laptop (4), File_Cabinet / Folder (4), Calendar / Notebook (1), Desk_Lamp / Lamp_Shade (1). These are ambiguous class pairs
  (a laptop is also a computer; a file cabinet holds folders) rather than random mistakes, and they are likely
  to show up as confusions between those classes. The other 6 groups are different objects that only look
  alike to the hash (mostly thin objects on white backgrounds), so their labels are correct. Splits of the
  10 true groups: train: 8, val: 2. Labels are left unchanged, to stay comparable with published Office-Home results;
  this label noise affects every method equally. List: `results/data_audit/conflicting_label_review.csv`.
- **Product vs Real World near-duplicates:** 14 candidate pairs (pHash); 2 are the same photo (Product copy in: train: 1, test: 1). These Real World images are not strictly unseen, but they are 2 of 4,357 target images (0.05%) and affect both selection rules equally, so they are kept and reported as a limitation.
  List: `results/data_audit/cross_domain_review.csv`.
- **Flagged samples:** 457 rows in `results/data_audit/flagged_samples.csv` (colour mode, size, shape, or
  near-duplicate with a different label).
- **Not checked automatically:** watermarks, overlaid text and mislabelled single images.

## 7. Preprocessing and augmentation (identical for every run)
- Train: `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip()`, `ToTensor()`, ImageNet normalisation.
- Validation / test / Real World: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, ImageNet normalisation.
- Mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; no statistics are computed from our data.

## 8. Privacy, bias and limits
- **People:** an automatic face detector (OpenCV Haar cascade) flagged 275 Real World and 137 Product images,
  but most hits are false alarms: in a visual check of 40 Real World hits, only 2 showed a real face (a child
  wearing glasses in the Glasses class and a historical portrait in the Radio class); the rest were bottles,
  flowers, wheels and labels. People therefore appear in only a small share of Real World images, mostly
  incidentally, and include at least one child. Product hits were not checked; catalog shots rarely show people
  beyond a hand. No image is labelled by person, and we do not identify, crop or redistribute anyone.
  Unverified detector hits: `results/data_audit/faces_detected.csv`.
- **Bias (not measured):** the objects appear to be mostly from Western offices, homes and online stores, and
  Product images are catalog shots on white backgrounds. Results may not transfer to other regions, object styles
  or camera conditions.
- **ImageNet overlap:** our ResNet-18 is ImageNet-pretrained, and many Office-Home classes (for example laptop, mug,
  keyboard, monitor) are also ImageNet categories, so the pretrained features may already know them.
- **Validation size:** at the 10% budget the 444 validation images outnumber the 355 training images, so validation-loss selection has more data to judge with than the model had to learn from.
- **Not for:** deployment, or any claim about people.

## 9. Representative examples
`results/figures/examples_product_vs_realworld.png` shows the same classes in Product and Real World.
Review sheets used for this card (not committed): `outputs/audit/review/`.

## Appendix: images per class and domain
| Class | Product | Real World | Art | Clipart |
|---|---|---|---|---|
| Alarm_Clock | 79 | 86 | 74 | 60 |
| Backpack | 99 | 99 | 41 | 56 |
| Batteries | 62 | 64 | 27 | 64 |
| Bed | 43 | 83 | 40 | 98 |
| Bike | 44 | 99 | 75 | 99 |
| Bottle | 62 | 78 | 99 | 99 |
| Bucket | 47 | 80 | 40 | 73 |
| Calculator | 81 | 73 | 33 | 46 |
| Calendar | 81 | 68 | 20 | 78 |
| Candles | 56 | 99 | 76 | 99 |
| Chair | 99 | 96 | 69 | 99 |
| Clipboards | 65 | 65 | 25 | 40 |
| Computer | 96 | 64 | 44 | 99 |
| Couch | 88 | 76 | 40 | 64 |
| Curtains | 75 | 73 | 40 | 42 |
| Desk_Lamp | 83 | 62 | 23 | 41 |
| Drill | 67 | 51 | 15 | 48 |
| Eraser | 41 | 43 | 18 | 40 |
| Exit_Sign | 67 | 81 | 21 | 41 |
| Fan | 58 | 60 | 45 | 50 |
| File_Cabinet | 71 | 58 | 22 | 40 |
| Flipflops | 99 | 85 | 46 | 40 |
| Flowers | 91 | 75 | 90 | 99 |
| Folder | 90 | 57 | 20 | 99 |
| Fork | 41 | 36 | 46 | 61 |
| Glasses | 67 | 60 | 40 | 52 |
| Hammer | 57 | 52 | 40 | 99 |
| Helmet | 90 | 60 | 79 | 69 |
| Kettle | 72 | 72 | 46 | 40 |
| Keyboard | 99 | 75 | 18 | 99 |
| Knives | 41 | 83 | 72 | 53 |
| Lamp_Shade | 54 | 78 | 49 | 40 |
| Laptop | 99 | 67 | 51 | 99 |
| Marker | 56 | 23 | 20 | 71 |
| Monitor | 98 | 71 | 42 | 99 |
| Mop | 72 | 46 | 32 | 40 |
| Mouse | 96 | 60 | 18 | 76 |
| Mug | 41 | 58 | 49 | 99 |
| Notebook | 93 | 68 | 21 | 83 |
| Oven | 68 | 64 | 20 | 40 |
| Pan | 70 | 30 | 19 | 51 |
| Paper_Clip | 47 | 68 | 19 | 40 |
| Pen | 60 | 65 | 20 | 99 |
| Pencil | 40 | 59 | 26 | 99 |
| Postit_Notes | 38 | 67 | 19 | 41 |
| Printer | 99 | 52 | 18 | 87 |
| Push_Pin | 43 | 53 | 24 | 40 |
| Radio | 43 | 66 | 47 | 46 |
| Refrigerator | 59 | 75 | 49 | 40 |
| Ruler | 58 | 41 | 15 | 67 |
| Scissors | 99 | 77 | 20 | 99 |
| Screwdriver | 40 | 51 | 30 | 75 |
| Shelf | 49 | 66 | 42 | 42 |
| Sink | 46 | 77 | 41 | 42 |
| Sneakers | 99 | 88 | 46 | 43 |
| Soda | 43 | 63 | 40 | 61 |
| Speaker | 99 | 81 | 20 | 90 |
| Spoon | 47 | 54 | 46 | 60 |
| TV | 76 | 53 | 40 | 99 |
| Table | 60 | 59 | 16 | 80 |
| Telephone | 58 | 82 | 44 | 99 |
| ToothBrush | 42 | 85 | 43 | 39 |
| Toys | 45 | 67 | 20 | 99 |
| Trash_Can | 93 | 81 | 21 | 53 |
| Webcam | 98 | 49 | 16 | 40 |
| **Total** | 4439 | 4357 | 2427 | 4365 |
