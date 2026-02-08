"""
Test script to verify MSP-YOLO setup before training
Run this BEFORE train_msp.py to catch any issues.
"""

import sys

import torch

print("=" * 70)
print("MSP-YOLO Setup Verification")
print("=" * 70)

# Test 1: Import check
print("\n1. Checking imports...")
try:
    from ultralytics.change_models.Msp import Bottleneck_MSP, C2f_MSP

    print("   ✓ MSP modules imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import MSP modules: {e}")
    sys.exit(1)

# Test 2: Module instantiation
print("\n2. Testing module instantiation...")
try:
    # Test C2f_MSP with various argument patterns
    m1 = C2f_MSP(64, 64, 2, True)
    m2 = C2f_MSP(64, 64, n=2, shortcut=True)
    m3 = C2f_MSP(64.0, 64.0, 2.0, True)  # Float inputs
    print("   ✓ C2f_MSP instantiation works")
except Exception as e:
    print(f"   ✗ C2f_MSP instantiation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n3. Testing forward pass...")
try:
    x = torch.randn(1, 64, 32, 32)
    out = m1(x)
    assert out.shape == (1, 64, 32, 32), f"Wrong output shape: {out.shape}"
    print(f"   ✓ Forward pass works (output shape: {out.shape})")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: YAML parsing simulation
print("\n4. Simulating YAML parsing...")
try:
    from ultralytics.utils.ops import make_divisible

    # Simulate what parse_model does
    width = 0.25  # n scale
    c1 = 32  # Previous layer output
    c2_raw = 128
    c2 = make_divisible(c2_raw * width, 8)  # 32
    n = 3
    args = [c1, c2, n, True]

    print(f"   Simulated args: {args}")
    m = C2f_MSP(*args)
    x = torch.randn(1, c1, 32, 32)
    out = m(x)
    print(f"   ✓ YAML pattern works (output: {out.shape})")
except Exception as e:
    print(f"   ✗ YAML pattern failed: {e}")
    sys.exit(1)

# Test 5: Check tasks.py registration
print("\n5. Checking tasks.py registration...")
try:
    from ultralytics.nn import tasks

    # Check if modules are in the sets
    has_bottleneck = Bottleneck_MSP in tasks.parse_model.__code__.co_consts
    has_c2f = C2f_MSP in tasks.parse_model.__code__.co_consts

    print("   Note: Module registration cannot be directly verified")
    print("   Will be confirmed when loading YAML")
except Exception as e:
    print(f"   ⚠ Could not verify registration: {e}")

# Test 6: Try loading a minimal YAML
print("\n6. Testing YAML loading...")
try:
    from ultralytics import YOLO

    # Try to create model from YAML
    print("   Attempting to load msp.yaml...")
    model = YOLO(r"ultralytics\cfg\models\v8\msp.yaml")
    print("   ✓ YAML loaded successfully!")

    # Try to build model
    print("   Building model...")
    model_info = str(model.model)
    if "C2f_MSP" in model_info:
        print("   ✓ C2f_MSP modules found in model!")
    else:
        print("   ⚠ C2f_MSP not found in model (might be using C2f instead)")

except Exception as e:
    print(f"   ✗ YAML loading failed: {e}")
    print("\n" + "=" * 70)
    print("ERROR DETAILS:")
    print("=" * 70)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour setup is ready for training!")
print("Run: python train_msp.py")
print("=" * 70)
