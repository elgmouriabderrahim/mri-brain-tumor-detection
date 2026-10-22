from pathlib import Path
import hashlib
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_ID = {"Glioma": 0, "Meningioma": 1, "No Tumor": 2, "Pituitary": 3}


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
    root_dir = Path(root)
    root_path = root_dir.resolve()
    rows, boxes = [], []
    for img in root_dir.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS or img.parent.name != "images":
            continue
        cls_dir = img.parent.parent
        lbl = cls_dir / "labels" / f"{img.stem}.txt"
        source_split = cls_dir.parent.name
        # Ignore any output folders created by clean_and_split on later notebook runs.
        if (source_split not in {"Train", "Val"}
                or cls_dir.parent.parent.resolve() != root_path):
            continue
        row = {"path": str(img), "label": cls_dir.name,
               "folder": source_split, "has_label": lbl.exists()}
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
        n_bad_geometry = sum(not box_is_valid(b) for b in bs)
        n_class_mismatch = sum(c != CLASS_TO_ID.get(cls_dir.name) for c, *_ in bs)
        row["n_boxes"] = len(bs)
        row["n_class_mismatch"] = n_class_mismatch
        row["n_bad_boxes"] = n_malformed + n_bad_geometry + n_class_mismatch
        rows.append(row)
        for c, xc, yc, w, h in bs:
            boxes.append({"path": str(img), "label": cls_dir.name,
                          "class_id": c, "class_matches_folder": c == CLASS_TO_ID.get(cls_dir.name),
                          "xc": xc, "yc": yc, "w": w, "h": h})
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


def clean_and_split(df, output_root="../data/split", seed=42, test_size=0.20):
    """Split original Train into folders; preserve original Val unchanged."""
    d = df[(df.folder == "Train") & (~df.corrupted) & (df.n_bad_boxes == 0) & df.has_label].copy()

    # Keep identical images out of the new split if their copy is already in Val.
    val_hashes = set(df.loc[(df.folder == "Val") & (~df.corrupted), "md5"].dropna())
    d = d[~d.md5.isin(val_hashes)]

    # Exclude conflicting labels and retain one copy of each identical Train image.
    conflicts = d.groupby("md5")["label"].nunique()
    d = d[~d.md5.isin(conflicts[conflicts > 1].index)]
    d = d.drop_duplicates("md5")

    train, test = train_test_split(
        d, test_size=test_size, stratify=d.label, random_state=seed
    )
    out = pd.concat([train.assign(split="Train"), test.assign(split="Test")])

    output_root = Path(output_root)
    for split in ("Train", "Test"):
        for class_name in CLASS_TO_ID:
            for kind in ("images", "labels"):
                folder = output_root / split / class_name / kind
                folder.mkdir(parents=True, exist_ok=True)
                for old_file in folder.iterdir():
                    if old_file.is_file():
                        old_file.unlink()

    for row in out.itertuples(index=False):
        image_path = Path(row.path)
        image_out = output_root / row.split / row.label / "images" / image_path.name
        shutil.copy2(image_path, image_out)
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if label_path.exists():
            label_out = output_root / row.split / row.label / "labels" / label_path.name
            shutil.copy2(label_path, label_out)

    return out[["path", "label", "md5", "n_boxes", "split"]].reset_index(drop=True)
