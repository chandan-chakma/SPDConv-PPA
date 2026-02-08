import os
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm

def visdrone2yolo(dir, split, source_name=None):
    """
    Convert VisDrone annotations to YOLO format with images/{split} and labels/{split} structure.
    
    Args:
        dir: Root directory containing VisDrone dataset
        split: Dataset split ('train', 'val', or 'test')
        source_name: Source folder name (e.g., 'VisDrone2019-DET-train')
    """
    source_dir = Path(dir) / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = Path(dir) / "images" / split
    labels_dir = Path(dir) / "labels" / split
    
    # Create directories
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split...")
    print(f"{'='*60}")
    print(f"Source directory: {source_dir}")
    print(f"Target images directory: {images_dir}")
    print(f"Target labels directory: {labels_dir}")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ ERROR: Source directory {source_dir} does not exist!")
        return
    
    # Move/copy images to new structure
    source_images_dir = source_dir / "images"
    if source_images_dir.exists():
        print(f"\n📸 Processing images from {source_images_dir}...")
        img_count = 0
        for img in tqdm(list(source_images_dir.glob("*.jpg")), desc="Copying images"):
            target_img = images_dir / img.name
            if not target_img.exists():
                shutil.copy2(img, target_img)  # Use copy2 to preserve metadata
            img_count += 1
        print(f"✅ Processed {img_count} images")
    else:
        print(f"⚠️  WARNING: Images directory {source_images_dir} does not exist!")
        return
    
    # Convert annotations (if they exist)
    annotations_dir = source_dir / "annotations"
    if not annotations_dir.exists():
        print(f"⚠️  WARNING: No annotations directory found at {annotations_dir}")
        print(f"   This is normal for test sets without labels.")
        print(f"   Images have been copied successfully.")
        return
    
    annotation_files = list(annotations_dir.glob("*.txt"))
    print(f"\n📝 Found {len(annotation_files)} annotation files")
    
    converted_count = 0
    skipped_count = 0
    empty_files = 0
    
    for f in tqdm(annotation_files, desc=f"Converting {split} annotations"):
        img_file = images_dir / f.with_suffix(".jpg").name
        
        # Check if corresponding image exists
        if not img_file.exists():
            skipped_count += 1
            continue
        
        try:
            # Get image dimensions
            img = Image.open(img_file)
            img_size = img.size
            dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
            
            lines = []
            with open(f, encoding="utf-8") as file:
                for row in [x.split(",") for x in file.read().strip().splitlines()]:
                    if len(row) < 6:
                        continue
                    
                    if row[4] != "0":  # Skip ignored regions (score = 0)
                        try:
                            x, y, w, h = map(int, row[:4])
                            cls = int(row[5]) - 1  # Convert to 0-indexed YOLO format
                            
                            # Skip invalid boxes
                            if w <= 0 or h <= 0 or cls < 0 or cls > 9:
                                continue
                            
                            # Convert to YOLO format (normalized center coordinates)
                            x_center = (x + w / 2) * dw
                            y_center = (y + h / 2) * dh
                            w_norm = w * dw
                            h_norm = h * dh
                            
                            # Clip values to [0, 1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            w_norm = max(0, min(1, w_norm))
                            h_norm = max(0, min(1, h_norm))
                            
                            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        except (ValueError, IndexError) as e:
                            continue
            
            # Write YOLO format annotations
            label_file = labels_dir / f.name
            if lines:
                label_file.write_text("".join(lines), encoding="utf-8")
                converted_count += 1
            else:
                # Create empty file for images with no valid objects
                label_file.write_text("", encoding="utf-8")
                empty_files += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {f.name}: {e}")
            skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed {split.upper()} split:")
    print(f"  ✅ Converted: {converted_count} files with objects")
    print(f"  📄 Empty: {empty_files} files (no valid objects)")
    print(f"  ⚠️  Skipped: {skipped_count} files")
    print(f"{'='*60}")


def main():
    """Main function to convert VisDrone dataset to YOLO format."""
    
    # MODIFY THIS PATH to where you extracted VisDrone dataset
    dataset_root = Path(r"datasets\VisDrone")
    
    print("\n" + "="*60)
    print("🚁 VisDrone to YOLO Converter")
    print("="*60)
    
    # Check if root directory exists
    if not dataset_root.exists():
        print(f"❌ ERROR: Dataset root directory does not exist: {dataset_root}")
        print("Please update the 'dataset_root' path in the script.")
        return
    
    # List available folders
    print(f"\n📂 Checking dataset root: {dataset_root}")
    available_folders = [f.name for f in dataset_root.iterdir() if f.is_dir() and f.name.startswith("VisDrone")]
    print(f"Found folders: {available_folders}")
    
    # Define splits and their source folders
    # Updated to match your actual folder names
    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test": "test"  # Changed from test-dev
    }
    
    # Convert each split
    for folder, split in splits.items():
        if (dataset_root / folder).exists():
            visdrone2yolo(dataset_root, split, folder)
        else:
            print(f"\n⚠️  Skipping {split}: folder '{folder}' not found")
    
    print("\n" + "="*60)
    print("✅ Conversion completed!")
    print("="*60)
    print(f"\n📁 Dataset structure created at: {dataset_root}")
    print("\n📂 Directory structure:")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  └── labels/")
    print("      ├── train/")
    print("      ├── val/")
    print("      └── test/")
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        img_dir = dataset_root / 'images' / split
        lbl_dir = dataset_root / 'labels' / split
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*.jpg')))
            lbl_count = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
            print(f"  {split:6s}: {img_count:5d} images, {lbl_count:5d} labels")
    
    # Optional: cleanup original directories
    print("\n" + "="*60)
    cleanup = input("🗑️  Remove original VisDrone folders? (yes/no): ").strip().lower()
    if cleanup == 'yes':
        for folder in splits.keys():
            folder_path = dataset_root / folder
            if folder_path.exists():
                print(f"  Removing {folder_path}...")
                shutil.rmtree(folder_path)
        print("✅ Cleanup completed!")
    else:
        print("⏭️  Skipping cleanup. Original folders preserved.")
    
    print("\n" + "="*60)
    print("🎯 Next steps:")
    print("  1. Update your VisDrone.yaml 'path' to point to this directory")
    print("  2. Run training: python train_yolov8.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()