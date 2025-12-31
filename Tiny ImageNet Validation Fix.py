import os
import shutil

base = "../data/tiny-imagenet-200/val"
ann_file = os.path.join(base, "val_annotations.txt")

with open(ann_file) as f:
    for line in f:
        img, cls = line.split('\t')[:2]
        cls_dir = os.path.join(base, cls)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.move(
            os.path.join(base, "images", img),
            os.path.join(cls_dir, img)
        )

shutil.rmtree(os.path.join(base, "images"))