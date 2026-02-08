from pathlib import Path
from tqdm import tqdm
import os

def fix_all_labels():
    """
    Fix ALL label files in UAVDT-YOLO dataset
    Convert class indices: 3→0, 5→1, 8→2
    """
    
    # Class mapping
    class_map = {
        '3': '0',  # car
        '5': '1',  # truck
        '8': '2'   # bus
    }
    
    uavdt_root = Path('UAVDT-YOLO')
    
    if not uavdt_root.exists():
        print("❌ UAVDT-YOLO folder not found!")
        return
    
    print("="*60)
    print("FIXING UAVDT CLASS INDICES")
    print("="*60)
    print("Mapping: class 3→0 (car), 5→1 (truck), 8→2 (bus)\n")
    
    total_files = 0
    total_fixed = 0
    
    for split in ['train', 'val', 'test']:
        label_dir = uavdt_root / split / 'labels'
        
        if not label_dir.exists():
            print(f"⚠ {split} labels not found, skipping...")
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        print(f"\nProcessing {split} split: {len(label_files)} files")
        
        fixed_in_split = 0
        
        for label_file in tqdm(label_files, desc=f'Fixing {split}'):
            try:
                # Read file
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                new_lines = []
                modified = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = parts[0]
                        
                        # Remap class if needed
                        if old_class in class_map:
                            parts[0] = class_map[old_class]
                            modified = True
                        
                        new_lines.append(' '.join(parts) + '\n')
                
                # Write back
                if modified:
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    fixed_in_split += 1
                    
            except Exception as e:
                print(f"\n❌ Error with {label_file.name}: {e}")
        
        print(f"✓ Fixed {fixed_in_split} files in {split}/")
        total_files += len(label_files)
        total_fixed += fixed_in_split
    
    # Delete cache files
    print("\nDeleting cache files...")
    for cache_file in uavdt_root.glob('**/*.cache'):
        try:
            cache_file.unlink()
            print(f"  Deleted: {cache_file}")
        except:
            pass
    
    print("\n" + "="*60)
    print("✓✓✓ FIX COMPLETE! ✓✓✓")
    print(f"Total files processed: {total_files}")
    print(f"Total files fixed: {total_fixed}")
    print("="*60)
    
    # Verify fix
    print("\nVerifying fix on sample files...")
    verify_fix()

def verify_fix():
    """Verify that classes are now 0, 1, 2"""
    uavdt_root = Path('UAVDT-YOLO')
    label_dir = uavdt_root / 'val' / 'labels'
    
    if not label_dir.exists():
        return
    
    # Check 5 random files
    label_files = list(label_dir.glob('*.txt'))
    if not label_files:
        print("No label files found!")
        return
    
    import random
    sample_files = random.sample(label_files, min(5, len(label_files)))
    
    all_classes = set()
    
    for label_file in sample_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    all_classes.add(parts[0])
    
    print(f"Classes found in samples: {sorted(all_classes)}")
    
    # Check for invalid classes
    valid_classes = {'0', '1', '2'}
    invalid = all_classes - valid_classes
    
    if invalid:
        print(f"⚠ WARNING: Still found invalid classes: {invalid}")
        print("You may need to reconvert the dataset!")
    else:
        print("✓ All classes are valid (0, 1, 2)!")

if __name__ == '__main__':
    fix_all_labels()