# Responsible-use and data limitations (report section, draft)

**Label ambiguity.** Ten groups of Product images (22 images) are the same picture filed under two classes, in four
class pairs: Computer/Laptop, File_Cabinet/Folder, Calendar/Notebook and Desk_Lamp/Lamp_Shade. These reflect genuinely
overlapping categories (a laptop is a computer) rather than random errors. Labels were left unchanged to stay comparable
with published Office-Home results. Eight groups are in training and two in validation, so they add a little noise to
training and to validation-loss selection. Both selection rules see the same checkpoints and validation set, so the
comparison stays fair, but errors between these pairs partly reflect the labels, not the model.

**Leakage and overlap.** Near-duplicate Product images were kept within one split, so Product validation and test
scores are not inflated by copies. Two Real World images (0.05%) are the same photo as a Product image; they are kept
and affect both rules equally. The ResNet-18 backbone is ImageNet-pretrained, and many Office-Home classes (laptop,
mug, keyboard, monitor) are ImageNet categories, so part of the Real World performance comes from pretraining rather
than from fine-tuning on Product.

**Validation-set caveat.** At the 10% label budget, the 444 labelled validation images outnumber the 355 training
images. A practitioner with few labels would rarely have this much validation data, which favours the validation-loss
baseline; any advantage of Rule W at low budgets is therefore conservative.

**Privacy.** No new data were collected. A face detector flagged about 400 images, but a visual check of 40 Real World
hits found only 2 real faces, including a child. People appear rarely and incidentally. Images are not redistributed,
no image is labelled by person, and figures in this report exclude identifiable people.

**Bias and scope of claims.** The objects appear to come mostly from Western offices, homes and online stores, and
Product images are catalog shots on white backgrounds; bias was not measured. Our evidence covers one architecture
(ResNet-18), one main domain shift (Product to Real World), three seeds and one alpha threshold (2.0). With three seeds
we report spread, not statistical significance. The results do not show that WeightWatcher-based selection works for
other models, domains or safety-critical uses. Rule W needs no labelled target data, but evaluating it did; in
deployment there would be no way to check whether the chosen checkpoint transfers well.

**License.** Office-Home is distributed under a fair-use notice for research, not an open license; we use it only for
this course project, with attribution, and our repository contains row IDs, labels and summaries but no images.

