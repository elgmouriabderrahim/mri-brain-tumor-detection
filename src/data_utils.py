from pathlib import Path
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_boxes(lbl_path):
    """Return (list of boxes, number of malformed lines) for one label file."""
    boxes, bad = [], 0
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                p = line.split()
                assert len(p) == 5
                boxes.append((int(p[0]), *map(float, p[1:])))
            except Exception:
                bad += 1
    return boxes, bad


def box_is_valid(b, eps=1e-6):
    _, xc, yc, w, h = b
    return (w > 0 and h > 0 and
            xc - w/2 >= -eps and yc - h/2 >= -eps and
            xc + w/2 <= 1 + eps and yc + h/2 <= 1 + eps)


def scan_dataset(root="data"):
    """One row per image (df) and one row per box (df_boxes)."""
    rows, boxes = [], []
    for img in Path(root).rglob("*"):
        if img.suffix.lower() not in IMG_EXTS or img.parent.name != "images":
            continue
        cls_dir = img.parent.parent
        lbl = cls_dir / "labels" / f"{img.stem}.txt"
        row = {"path": str(img), "label": cls_dir.name,
               "folder": cls_dir.parent.name, "has_label": lbl.exists()}
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                row.update(width=im.width, height=im.height, mode=im.mode)
            row["md5"] = hashlib.md5(img.read_bytes()).hexdigest()
            row["corrupted"] = False
        except Exception:
            row["corrupted"] = True

        bs, n_malformed = read_boxes(lbl)
        row["n_boxes"] = len(bs)
        row["n_bad_boxes"] = n_malformed + sum(not box_is_valid(b) for b in bs)
        rows.append(row)
        for c, xc, yc, w, h in bs:
            boxes.append({"path": str(img), "label": cls_dir.name,
                          "class_id": c, "xc": xc, "yc": yc, "w": w, "h": h})
    return pd.DataFrame(rows), pd.DataFrame(boxes)


def show_samples(df, df_boxes, n=4, seed=42):
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(labels), n, figsize=(3.5 * n, 3.5 * len(labels)))
    for i, lab in enumerate(labels):
        sub = df[(df.label == lab) & (~df.corrupted)]
        for j, (_, r) in enumerate(sub.sample(min(n, len(sub)), random_state=seed).iterrows()):
            ax = axes[i][j]
            img = Image.open(r.path).convert("RGB")
            W, H = img.size
            ax.imshow(img)
            for _, b in df_boxes[df_boxes.path == r.path].iterrows():
                ax.add_patch(patches.Rectangle(((b.xc - b.w/2) * W, (b.yc - b.h/2) * H),
                                               b.w * W, b.h * H,
                                               fill=False, edgecolor="red", linewidth=2))
            ax.set_title(f"{lab} | {r.n_boxes} box", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def clean_and_split(df, seed=42):
    d = df[~df.corrupted]                                   # corrupted files
    d = d[d.n_bad_boxes == 0]                               # inconsistent annotations
    conflicts = d.groupby("md5")["label"].nunique()
    d = d[~d.md5.isin(conflicts[conflicts > 1].index)]      # same image, different labels
    d = d.drop_duplicates("md5")                            # duplicates (also across train/val)

    train, temp = train_test_split(d, test_size=0.30, stratify=d.label, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=seed)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return out[["path", "label", "md5", "n_boxes", "split"]].reset_index(drop=True)