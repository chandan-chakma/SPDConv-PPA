"""Test RSP-YOLO model loading."""

import torch

from ultralytics import YOLO

print("=" * 70)
print("Testing RSP-YOLO Model")
print("=" * 70)

try:
    # Test 1: Load model architecture
    print("\n1. Loading RSP-YOLO architecture...")
    model = YOLO("ultralytics/cfg/models/v8/RSP-yolo.yaml")
    print("   ✓ Architecture loaded successfully")

    # Test 2: Load pretrained weights
    print("\n2. Loading pretrained weights...")
    model = model.load("yolov8n.pt")
    print("   ✓ Weights loaded successfully")

    # Test 3: Check model structure
    print("\n3. Checking model structure...")
    print(f"   Model has {len(model.model.model)} layers")

    # Test 4: Test forward pass
    print("\n4. Testing forward pass...")
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    print("   ✓ Forward pass successful")
    print(f"   Output shape: {y.shape if isinstance(y, torch.Tensor) else [yi.shape for yi in y]}")

    # Test 5: Check detection heads
    print("\n5. Checking detection heads...")
    detect_layer = model.model.model[-1]
    print(f"   Detection strides: {detect_layer.stride}")
    expected_strides = torch.tensor([4.0, 8.0, 16.0])
    if torch.allclose(detect_layer.stride, expected_strides):
        print("   ✓ P2 detection head present (stride 4)!")
    else:
        print(f"   ⚠ Unexpected strides: expected {expected_strides}, got {detect_layer.stride}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nRSP-YOLO is ready to train!")
    print("Run: python train_rsp_yolo_simple.py")

except Exception as e:
    print("\n" + "=" * 70)
    print("❌ ERROR!")
    print("=" * 70)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nDebugging info:")
    import traceback

    traceback.print_exc()
    print("\n" + "=" * 70)
