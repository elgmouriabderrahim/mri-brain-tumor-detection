import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_utils import read_boxes, CLASS_TO_ID, IMG_EXTS

IMG_SIZE = 224


def get_class_names(_=None):
    return list(CLASS_TO_ID) 

def build_table(data_root="../data", val_ok_paths=None):
    """One row per image: path, label, split (train/val/test), n_boxes."""
    root = Path(data_root)
    sources = {"train": root / "split" / "Train",
               "val":   root / "Val",
               "test":  root / "split" / "Test"}
    rows = []
    for split, folder in sources.items():
        for cls in CLASS_TO_ID:
            for img in sorted((folder / cls / "images").glob("*")):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                if split == "val" and val_ok_paths is not None and str(img) not in val_ok_paths:
                    continue
                n_boxes = len(read_boxes(label_path(img))[0])
                rows.append({"path": str(img), "label": cls, "split": split, "n_boxes": n_boxes})
    return pd.DataFrame(rows)


def label_path(img_path):
    p = Path(img_path)
    return p.parent.parent / "labels" / f"{p.stem}.txt"


def get_transforms(train):
    """Train: resize + random geometry/contrast. Val/test: resize only (deterministic)."""
    steps = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        steps += [
            A.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
        ]
    bbox = A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.3)
    return A.Compose(steps, bbox_params=bbox)


def normalize(img):
    return img.astype("float32") / 127.5 - 1.0       # [0,255] -> [-1,1]


def load_sample(path, transform):
    """Return the transformed image (uint8), its adapted boxes and class ids."""
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    boxes, _ = read_boxes(label_path(path))
    out = transform(image=img,
                    bboxes=[b[1:] for b in boxes],
                    class_ids=[b[0] for b in boxes])
    return out["image"], out["bboxes"], out["class_ids"]


def make_dataset(df, class_names, train, batch_size=32):
    """tf.data pipeline for classification: yields (image, integer label)."""
    import tensorflow as tf
    transform = get_transforms(train)
    paths = df["path"].tolist()
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = df["label"].map(class_to_idx).tolist()

    def gen():
        order = np.random.permutation(len(paths)) if train else range(len(paths))
        for i in order:
            img, _, _ = load_sample(paths[i], transform)
            yield normalize(img), labels[i]

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
                               tf.TensorSpec((), tf.int32)))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def show_augmentation(path, n=4):
    """Original + n random augmentations, with boxes following the image."""
    plain, aug = get_transforms(False), get_transforms(True)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 3.5))
    for k, ax in enumerate(axes):
        img, boxes, _ = load_sample(path, plain if k == 0 else aug)
        ax.imshow(img)
        for xc, yc, w, h in boxes:
            ax.add_patch(patches.Rectangle(((xc - w/2) * IMG_SIZE, (yc - h/2) * IMG_SIZE),
                                           w * IMG_SIZE, h * IMG_SIZE,
                                           fill=False, edgecolor="red", linewidth=2))
        ax.set_title("original" if k == 0 else f"augmented {k}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()