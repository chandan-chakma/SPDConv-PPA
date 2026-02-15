import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) normalized format. x1, y1
    is the top-left corner, x2, y2 is the bottom-right corner.

    Args:
        x: box coordinates in (x1, y1, x2, y2) format
        w: image width
        h: image height
        clip: clip coordinates to image boundaries
        eps: epsilon value for numerical stability

    Returns:
        box in (x_center, y_center, width, height) normalized format
    """
    if clip:
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - eps)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - eps)

    # Convert to center format
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center (normalized)
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center (normalized)
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width (normalized)
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height (normalized)

    return y


def convert_labels(fname=Path("datasets/xView/xView_train.geojson")):
    """Converts xView geoJSON labels to YOLO format.

    xView has 60 classes (originally numbered 11-94 in the dataset). This function maps them to 0-59 for YOLO.
    """
    path = fname.parent

    print(f"\n{'=' * 70}")
    print("Converting xView Dataset to YOLO Format")
    print(f"{'=' * 70}\n")

    # Load geoJSON file
    with open(fname, encoding="utf-8") as f:
        print(f"📂 Loading {fname}...")
        data = json.load(f)

    print(f"✅ Found {len(data['features'])} annotations")

    # Create labels directory
    labels = Path(path / "labels" / "train")
    if labels.exists():
        print("🗑️  Removing existing labels directory...")
        shutil.rmtree(labels)
    labels.mkdir(parents=True, exist_ok=True)

    # xView class mapping: original class IDs (11-94) to YOLO indices (0-59)
    xview_class2index = [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,  # 0-10
        0,
        1,
        2,
        -1,
        3,
        -1,
        4,
        5,
        6,
        7,
        8,
        -1,
        9,
        10,
        11,  # 11-25
        12,
        13,
        14,
        15,
        -1,
        -1,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        -1,  # 26-39
        23,
        24,
        25,
        -1,
        26,
        27,
        -1,
        28,
        -1,
        29,
        30,
        31,
        32,
        33,  # 40-53
        34,
        35,
        36,
        37,
        -1,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        -1,  # 54-67
        -1,
        -1,
        -1,
        46,
        47,
        48,
        49,
        -1,
        50,
        51,
        -1,
        52,
        -1,
        -1,  # 68-81
        -1,
        53,
        54,
        -1,
        55,
        -1,
        -1,
        56,
        -1,
        57,
        -1,
        58,
        59,  # 82-94
    ]

    shapes = {}  # Cache for image dimensions
    skipped = 0
    converted = 0
    missing_images = set()

    # Process each annotation
    for feature in tqdm(data["features"], desc="Converting annotations"):
        p = feature["properties"]

        if p["bounds_imcoords"]:
            image_id = p["image_id"]
            image_file = path / "train_images" / image_id

            if not image_file.exists():
                if image_id not in missing_images:
                    missing_images.add(image_id)
                skipped += 1
                continue

            try:
                # Parse bounding box coordinates
                box = np.array([int(num) for num in p["bounds_imcoords"].split(",")])

                if box.shape[0] != 4:
                    print(f"⚠️  Skipping invalid box shape for {image_id}: {box.shape[0]}")
                    skipped += 1
                    continue

                # Get class index
                cls = p["type_id"]
                cls = xview_class2index[int(cls)]

                if not (0 <= cls <= 59):
                    print(f"⚠️  Skipping invalid class {cls} for {image_id}")
                    skipped += 1
                    continue

                # Get image dimensions (cache them)
                if image_id not in shapes:
                    shapes[image_id] = Image.open(image_file).size

                w, h = shapes[image_id]

                # Convert to YOLO format
                box = xyxy2xywhn(box[None].astype(np.float64), w=w, h=h, clip=True)

                # Write label to file
                label_file = (labels / image_id).with_suffix(".txt")
                with open(label_file, "a", encoding="utf-8") as f:
                    f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box[0])}\n")

                converted += 1

            except Exception as e:
                print(f"⚠️  Error processing {image_id}: {e}")
                skipped += 1

    print(f"\n{'=' * 70}")
    print("Conversion Complete!")
    print(f"{'=' * 70}")
    print(f"✅ Converted: {converted} annotations")
    print(f"⚠️  Skipped: {skipped} annotations")
    if missing_images:
        print(f"❌ Missing images: {len(missing_images)} (e.g., {list(missing_images)[:3]})")
    print(f"{'=' * 70}\n")


def reorganize_images(base_path):
    """Reorganize images into YOLO-compatible structure: datasets/xView/ ├── images/ │ ├── train/ │ └── val/ └── labels/
    └── train/.
    """
    base_path = Path(base_path)
    images_dir = base_path / "images"

    print(f"\n{'=' * 70}")
    print("Reorganizing Images")
    print(f"{'=' * 70}\n")

    # Create images directory
    images_dir.mkdir(parents=True, exist_ok=True)

    # Move training images
    train_src = base_path / "train_images"
    train_dst = images_dir / "train"

    if train_src.exists() and not train_dst.exists():
        print(f"📂 Moving training images: {train_src} → {train_dst}")
        shutil.move(str(train_src), str(train_dst))
        train_count = len(list(train_dst.glob("*.tif")))
        print(f"✅ Moved {train_count} training images")
    elif train_dst.exists():
        print("✅ Training images already in place")
    else:
        print(f"⚠️  Training images not found at {train_src}")

    # Move validation images
    val_src = base_path / "val_images"
    val_dst = images_dir / "val"

    if val_src.exists() and not val_dst.exists():
        print(f"📂 Moving validation images: {val_src} → {val_dst}")
        shutil.move(str(val_src), str(val_dst))
        val_count = len(list(val_dst.glob("*.tif")))
        print(f"✅ Moved {val_count} validation images")
    elif val_dst.exists():
        print("✅ Validation images already in place")
    else:
        print(f"⚠️  Validation images not found at {val_src}")

    print(f"\n{'=' * 70}")
    print("Image Reorganization Complete!")
    print(f"{'=' * 70}\n")


def create_autosplit_files(base_path, train_ratio=0.9):
    """Create train/val split files for YOLO. Since xView val set has no labels, we split the train set.
    """
    base_path = Path(base_path)
    images_dir = base_path / "images" / "train"

    print(f"\n{'=' * 70}")
    print("Creating Train/Val Split")
    print(f"{'=' * 70}\n")

    # Get all training images
    image_files = sorted(images_dir.glob("*.tif"))
    print(f"📊 Found {len(image_files)} training images")

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(image_files))

    split_idx = int(len(image_files) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"📊 Split: {len(train_indices)} train, {len(val_indices)} val")

    # Create autosplit files
    train_file = images_dir.parent / "autosplit_train.txt"
    val_file = images_dir.parent / "autosplit_val.txt"

    # Write train file
    with open(train_file, "w") as f:
        for idx in train_indices:
            rel_path = f"./train/{image_files[idx].name}"
            f.write(f"{rel_path}\n")

    print(f"✅ Created {train_file}")

    # Write val file
    with open(val_file, "w") as f:
        for idx in val_indices:
            rel_path = f"./train/{image_files[idx].name}"
            f.write(f"{rel_path}\n")

    print(f"✅ Created {val_file}")

    print(f"\n{'=' * 70}")
    print("Split Creation Complete!")
    print(f"{'=' * 70}\n")


def verify_conversion(base_path):
    """Verify the conversion was successful."""
    base_path = Path(base_path)

    print(f"\n{'=' * 70}")
    print("Verifying Conversion")
    print(f"{'=' * 70}\n")

    # Check images
    train_images = list((base_path / "images" / "train").glob("*.tif"))
    val_images = list((base_path / "images" / "val").glob("*.tif"))

    print(f"📸 Training images: {len(train_images)}")
    print(f"📸 Validation images: {len(val_images)}")

    # Check labels
    train_labels = list((base_path / "labels" / "train").glob("*.txt"))

    print(f"🏷️  Training labels: {len(train_labels)}")

    # Check split files
    train_split = base_path / "images" / "autosplit_train.txt"
    val_split = base_path / "images" / "autosplit_val.txt"

    if train_split.exists():
        with open(train_split) as f:
            train_count = len(f.readlines())
        print(f"📄 Train split file: {train_count} images")
    else:
        print("❌ Train split file missing!")

    if val_split.exists():
        with open(val_split) as f:
            val_count = len(f.readlines())
        print(f"📄 Val split file: {val_count} images")
    else:
        print("❌ Val split file missing!")

    # Sample label check
    if train_labels:
        sample_label = train_labels[0]
        print(f"\n📋 Sample label file: {sample_label.name}")
        with open(sample_label) as f:
            lines = f.readlines()
            print(f"   Objects: {len(lines)}")
            if lines:
                print(f"   First object: {lines[0].strip()}")

    print(f"\n{'=' * 70}")
    print("Verification Complete!")
    print(f"{'=' * 70}\n")


def main():
    """Main conversion function for xView dataset."""
    # Set your xView dataset path
    xview_path = Path("datasets/xView")

    print("\n" + "=" * 70)
    print("🛰️  xView Dataset → YOLO Format Converter")
    print("=" * 70)

    # Check if dataset exists
    if not xview_path.exists():
        print(f"\n❌ Error: Dataset not found at {xview_path}")
        print("Please download the xView dataset manually from:")
        print("https://challenge.xviewdataset.org")
        return

    # Check for required files
    geojson_file = xview_path / "xView_train.geojson"
    if not geojson_file.exists():
        print(f"\n❌ Error: Labels not found at {geojson_file}")
        print(f"Please ensure xView_train.geojson is in {xview_path}")
        return

    # Step 1: Convert labels
    print("\n📝 Step 1: Converting labels to YOLO format...")
    convert_labels(geojson_file)

    # Step 2: Reorganize images
    print("\n📁 Step 2: Reorganizing image directories...")
    reorganize_images(xview_path)

    # Step 3: Create train/val split
    print("\n✂️  Step 3: Creating train/val split...")
    create_autosplit_files(xview_path, train_ratio=0.9)

    # Step 4: Verify conversion
    print("\n✅ Step 4: Verifying conversion...")
    verify_conversion(xview_path)

    print("\n" + "=" * 70)
    print("🎉 xView Dataset Conversion Complete!")
    print("=" * 70)
    print("\nYou can now train with:")
    print("python train.py --data xView.yaml")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
