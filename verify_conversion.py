# verify_conversion.py
import os
import cv2
import yaml

def verify_yolo_dataset(data_yaml_path):
    """Verify YOLO dataset format"""
    
    # Load yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    root = data['path']
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(root, 'images', split)
        label_dir = os.path.join(root, 'labels', split)
        
        img_files = list(Path(img_dir).glob('*.jpg'))
        label_files = list(Path(label_dir).glob('*.txt'))
        
        print(f"\n{split.upper()} split:")
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(label_files)}")
        
        # Check a few samples
        for img_file in img_files[:3]:
            label_file = Path(label_dir) / (img_file.stem + '.txt')
            
            # Read image
            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]
            
            print(f"\n  Sample: {img_file.name}")
            print(f"    Image size: {w}x{h}")
            
            # Read labels
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = f.readlines()
                print(f"    Objects: {len(labels)}")
                
                for label in labels[:3]:  # Show first 3 objects
                    parts = label.strip().split()
                    cls = int(parts[0])
                    print(f"      Class {cls}: {data['names'][cls]}")
            else:
                print(f"    No labels found!")

if __name__ == "__main__":
    verify_yolo_dataset("datasets\UAVDT1\data.yaml")