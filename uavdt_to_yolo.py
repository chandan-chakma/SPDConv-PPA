import os
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

# UAVDT class mapping to VisDrone classes
# UAVDT: 1=car, 2=truck, 3=bus
# VisDrone: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
UAVDT_TO_VISDRONE = {
    1: 3,  # car -> car
    2: 5,  # truck -> truck
    3: 8   # bus -> bus
}

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    Convert UAVDT bbox (x, y, w, h) to YOLO format (x_center, y_center, w, h) normalized
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return x_center, y_center, w, h

def parse_gt_file(gt_file):
    """
    Parse UAVDT ground truth file
    Returns: dict with frame_id as key and list of annotations as value
    """
    annotations = {}
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            frame_id = int(parts[0])
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            out_of_view = int(parts[6])
            occlusion = int(parts[7])
            category = int(parts[8])
            
            # Skip objects that are out of view or invalid category
            if out_of_view == 1 or category not in UAVDT_TO_VISDRONE:
                continue
            
            if frame_id not in annotations:
                annotations[frame_id] = []
            
            annotations[frame_id].append({
                'bbox': bbox,
                'category': category,
                'occlusion': occlusion
            })
    
    return annotations

def convert_sequence(seq_name, img_dir, gt_file, output_img_dir, output_label_dir):
    """
    Convert one UAVDT sequence to YOLO format
    """
    print(f"Processing sequence: {seq_name}")
    
    # Parse annotations
    annotations = parse_gt_file(gt_file)
    
    # Get all images
    img_files = sorted(list(Path(img_dir).glob('*.jpg')))
    
    for img_file in tqdm(img_files, desc=f"Converting {seq_name}"):
        # Extract frame number from filename (e.g., img0001.jpg -> 1)
        frame_id = int(img_file.stem.replace('img', ''))
        
        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Create output filename
        output_img_name = f"{seq_name}_frame{frame_id:06d}.jpg"
        output_label_name = f"{seq_name}_frame{frame_id:06d}.txt"
        
        # Copy image
        output_img_path = os.path.join(output_img_dir, output_img_name)
        shutil.copy(str(img_file), output_img_path)
        
        # Write YOLO format labels
        output_label_path = os.path.join(output_label_dir, output_label_name)
        
        if frame_id in annotations:
            with open(output_label_path, 'w') as f:
                for ann in annotations[frame_id]:
                    # Convert category
                    yolo_class = UAVDT_TO_VISDRONE[ann['category']]
                    
                    # Convert bbox to YOLO format
                    x_center, y_center, w, h = convert_bbox_to_yolo(
                        img_width, img_height, ann['bbox']
                    )
                    
                    # Write in YOLO format: class x_center y_center width height
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        else:
            # Create empty label file if no annotations
            open(output_label_path, 'w').close()

def main():
    # Paths
    uavdt_root = r"datasets\UAVDT"  # CHANGE THIS
    output_root = r"datasets\UAVDT1"  # CHANGE THIS
    
    benchmark_dir = os.path.join(uavdt_root, "UAV-benchmark-M")
    gt_dir = os.path.join(uavdt_root, "UAV-benchmark-MOTD_v1.0\GT")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'labels', split), exist_ok=True)
    
    # Get all sequences
    sequences = sorted([d for d in os.listdir(benchmark_dir) 
                       if os.path.isdir(os.path.join(benchmark_dir, d))])
    
    print(f"Found {len(sequences)} sequences")
    
    # Split sequences: 70% train, 15% val, 15% test
    n_train = int(len(sequences) * 0.7)
    n_val = int(len(sequences) * 0.15)
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train+n_val]
    test_seqs = sequences[n_train+n_val:]
    
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
    
    # Process each split
    for split, seqs in [('train', train_seqs), ('val', val_seqs), ('test', test_seqs)]:
        print(f"\n{'='*50}")
        print(f"Processing {split} split ({len(seqs)} sequences)")
        print(f"{'='*50}")
        
        for seq_name in seqs:
            img_dir = os.path.join(benchmark_dir, seq_name)
            gt_file = os.path.join(gt_dir, f"{seq_name}_gt_whole.txt")
            
            if not os.path.exists(gt_file):
                print(f"Warning: GT file not found for {seq_name}")
                continue
            
            output_img_dir = os.path.join(output_root, 'images', split)
            output_label_dir = os.path.join(output_root, 'labels', split)
            
            convert_sequence(seq_name, img_dir, gt_file, output_img_dir, output_label_dir)
    
    # Create data.yaml
    yaml_content = f"""# UAVDT dataset in YOLO format
path: {output_root}
train: images/train
val: images/val
test: images/test

# Classes (using VisDrone class indices)
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# UAVDT only has 3 classes: car (4), truck (6), bus (9)
# Other classes won't appear in labels but kept for consistency with VisDrone
"""
    
    with open(os.path.join(output_root, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_root}")
    print("="*50)

if __name__ == "__main__":
    main()