# Progress: reproducibility and extensions

## Progress-report paragraph
The split pipeline (`scripts/01_data.sh`) was rerun from scratch on a different platform (Kaggle, Python 3.13). All checks passed: 15,588 images, 65 classes, 0 unreadable images, a Product split of 3,551 / 444 / 444 with no duplicate groups crossing splits, and nested label budgets (355 / 888 / 1,776 / 3,551) for all three seeds. Every image was assigned to the same split as in the original; the only difference was 22 perceptual-hash values (library versions), which changed no split decision. Details are in `docs/reproduction.md`. Next: the ablation-2 script (Rule W trigger patience 1, 2 or 3), run after the Nov 16 freeze.

## Representative examples
**Figure 1** (`results/figures/examples_product_vs_realworld.png`). Product (top) vs Real World (bottom) images for six of the 65 classes. Product images show centered objects on white backgrounds; Real World images have clutter, natural lighting and varied viewpoints. This is the domain shift the selection rule must handle.
