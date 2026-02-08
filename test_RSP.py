"""
Complete Testing Script for RSP-YOLO
Evaluate on VisDrone and xView datasets.
"""

import os

import pandas as pd

from ultralytics import YOLO


def test_model(weights, data="VisDrone.yaml", split="val", imgsz=640, batch=8, device=0, save_json=False):
    """Test model on dataset.

    Args:
        weights: Path to model weights
        data: Dataset YAML file
        split: 'val' or 'test'
        imgsz: Image size for testing
        batch: Batch size
        device: GPU device ID
        save_json: Save results in JSON format

    Returns:
        Validation results
    """
    print("=" * 70)
    print(f"Testing RSP-YOLO on {data} ({split} split)")
    print("=" * 70)
    print(f"Weights:     {weights}")
    print(f"Image size:  {imgsz}")
    print(f"Batch size:  {batch}")
    print(f"Device:      cuda:{device}")
    print("=" * 70)

    if not os.path.exists(weights):
        print(f"\n❌ Error: Weights file not found: {weights}")
        return None

    # Load model
    model = YOLO(weights)

    # Run validation
    results = model.val(
        data=data, split=split, imgsz=imgsz, batch=batch, device=device, save_json=save_json, plots=True, verbose=True
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"mAP@0.5:      {results.box.map50:.4f} ({results.box.map50 * 100:.2f}%)")
    print(f"mAP@0.5-0.95: {results.box.map:.4f} ({results.box.map * 100:.2f}%)")
    print(f"Precision:    {results.box.mp:.4f}")
    print(f"Recall:       {results.box.mr:.4f}")
    print("=" * 70)

    # Per-class results
    if hasattr(results.box, "ap_class_index"):
        print("\nPer-class AP@0.5:")
        print("-" * 70)
        for i, cls_idx in enumerate(results.box.ap_class_index):
            ap = results.box.ap50[i]
            print(f"Class {cls_idx:2d}: {ap:.4f} ({ap * 100:.2f}%)")
        print("=" * 70)

    return results


def compare_models(rsp_weights, baseline_weights, data="VisDrone.yaml"):
    """Compare RSP-YOLO with baseline YOLOv8n.

    Args:
        rsp_weights: Path to RSP-YOLO weights
        baseline_weights: Path to baseline weights
        data: Dataset YAML file
    """
    print("\n" + "=" * 70)
    print("COMPARING RSP-YOLO vs BASELINE YOLOv8n")
    print("=" * 70)

    # Test both models
    print("\n1. Testing Baseline YOLOv8n...")
    baseline_results = test_model(baseline_weights, data, split="val")

    print("\n2. Testing RSP-YOLO...")
    rsp_results = test_model(rsp_weights, data, split="val")

    if baseline_results is None or rsp_results is None:
        print("❌ Comparison failed - missing results")
        return

    # Create comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    comparison = {
        "Metric": ["mAP@0.5", "mAP@0.5-0.95", "Precision", "Recall"],
        "Baseline": [
            f"{baseline_results.box.map50 * 100:.2f}%",
            f"{baseline_results.box.map * 100:.2f}%",
            f"{baseline_results.box.mp:.4f}",
            f"{baseline_results.box.mr:.4f}",
        ],
        "RSP-YOLO": [
            f"{rsp_results.box.map50 * 100:.2f}%",
            f"{rsp_results.box.map * 100:.2f}%",
            f"{rsp_results.box.mp:.4f}",
            f"{rsp_results.box.mr:.4f}",
        ],
        "Improvement": [
            f"+{(rsp_results.box.map50 - baseline_results.box.map50) * 100:.2f}%",
            f"+{(rsp_results.box.map - baseline_results.box.map) * 100:.2f}%",
            f"+{rsp_results.box.mp - baseline_results.box.mp:.4f}",
            f"+{rsp_results.box.mr - baseline_results.box.mr:.4f}",
        ],
    }

    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    print("=" * 70)

    # Calculate improvement
    map50_gain = (rsp_results.box.map50 - baseline_results.box.map50) * 100
    map_gain = (rsp_results.box.map - baseline_results.box.map) * 100

    print("\n🎯 Overall Improvement:")
    print(f"   mAP@0.5: +{map50_gain:.2f} percentage points")
    print(f"   mAP@0.5-0.95: +{map_gain:.2f} percentage points")

    if map50_gain >= 10:
        print("\n🎉 EXCELLENT! RSP-YOLO shows significant improvement (+10% or more)!")
    elif map50_gain >= 5:
        print("\n✅ GOOD! RSP-YOLO shows solid improvement (+5-10%)!")
    elif map50_gain >= 2:
        print("\n⚠️  RSP-YOLO shows modest improvement (+2-5%).")
    else:
        print("\n❌ RSP-YOLO shows minimal improvement (<2%). Check training.")

    print("=" * 70)

    return comparison


def test_multiple_scales(weights, data="VisDrone.yaml"):
    """Test at multiple image sizes to find optimal scale.

    Args:
        weights: Path to model weights
        data: Dataset YAML file
    """
    print("\n" + "=" * 70)
    print("MULTI-SCALE TESTING")
    print("=" * 70)

    scales = [640, 800, 1024]
    results_dict = {}

    for imgsz in scales:
        print(f"\nTesting at {imgsz}px...")
        results = test_model(weights, data, split="val", imgsz=imgsz, batch=4)
        if results:
            results_dict[imgsz] = {"mAP@0.5": results.box.map50 * 100, "mAP@0.5-0.95": results.box.map * 100}

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-SCALE RESULTS")
    print("=" * 70)

    for imgsz, metrics in results_dict.items():
        print(f"{imgsz}px: mAP@0.5={metrics['mAP@0.5']:.2f}%, mAP@0.5-0.95={metrics['mAP@0.5-0.95']:.2f}%")

    # Find best scale
    best_scale = max(results_dict.items(), key=lambda x: x[1]["mAP@0.5"])[0]
    print(f"\n🎯 Best scale: {best_scale}px")
    print("=" * 70)

    return results_dict


def visualize_predictions(weights, source, save_dir="predictions"):
    """Visualize predictions on sample images.

    Args:
        weights: Path to model weights
        source: Path to images or video
        save_dir: Directory to save predictions
    """
    print("\n" + "=" * 70)
    print("VISUALIZING PREDICTIONS")
    print("=" * 70)

    model = YOLO(weights)

    results = model.predict(
        source=source,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        iou=0.45,
        project=save_dir,
        name="vis",
        exist_ok=True,
    )

    print(f"\n✅ Predictions saved to: {save_dir}/vis/")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RSP-YOLO")

    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, default="VisDrone.yaml", help="Dataset YAML file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for testing")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")

    # Comparison mode
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline weights for comparison")

    # Multi-scale testing
    parser.add_argument("--multi-scale", action="store_true", help="Test at multiple scales (640, 800, 1024)")

    # Visualization
    parser.add_argument("--visualize", type=str, default=None, help="Path to images/video for visualization")

    args = parser.parse_args()

    # Mode 1: Compare with baseline
    if args.compare:
        compare_models(rsp_weights=args.weights, baseline_weights=args.compare, data=args.data)

    # Mode 2: Multi-scale testing
    elif args.multi_scale:
        test_multiple_scales(weights=args.weights, data=args.data)

    # Mode 3: Visualization
    elif args.visualize:
        visualize_predictions(weights=args.weights, source=args.visualize)

    # Mode 4: Standard testing
    else:
        results = test_model(
            weights=args.weights,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

        if results:
            print("\n✅ Testing complete!")
            print(f"\nFinal mAP@0.5: {results.box.map50 * 100:.2f}%")
            print(f"Final mAP@0.5-0.95: {results.box.map * 100:.2f}%")
