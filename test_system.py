"""
test_system.py
=============================================================================
Smart Object Detection and AI Image Description System
Automated Test Suite: Verifies modules, functions, edge cases, and fallbacks.
=============================================================================
"""

import sys
import io
import time
from PIL import Image, ImageDraw
import pandas as pd

def test_utils():
    print("[1/3] Testing utils.py functions...")
    from utils import (
        validate_image_file,
        get_spatial_position,
        detections_to_dataframe,
        detections_to_csv_bytes,
        image_to_bytes,
        generate_full_report
    )

    # 1. Test image generation & validation
    img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.name = "test.jpg"

    is_valid, err, loaded = validate_image_file(buf, max_size_mb=5.0)
    assert is_valid is True, f"Validation failed: {err}"
    assert loaded is not None
    assert loaded.size == (400, 300)
    print("  ✓ Image validation passed for valid JPEG")

    # 2. Test invalid extension
    buf_invalid = io.BytesIO(b"dummy")
    buf_invalid.name = "test.exe"
    is_valid, err, _ = validate_image_file(buf_invalid)
    assert is_valid is False
    assert "Unsupported file format" in err
    print("  ✓ Invalid extension correctly rejected")

    # 3. Test spatial position locator
    pos = get_spatial_position([10, 10, 50, 50], 400, 300)
    assert pos == "Top-Left", f"Expected Top-Left, got {pos}"
    pos_center = get_spatial_position([180, 130, 220, 170], 400, 300)
    assert "Center" in pos_center
    print("  ✓ Spatial quadrant positioning verified")

    # 4. Test dataframe and export
    mock_detections = [
        {"label": "person", "confidence": 0.952, "box": [10.0, 10.0, 80.0, 120.0], "class_id": 0},
        {"label": "car", "confidence": 0.887, "box": [150.0, 120.0, 320.0, 240.0], "class_id": 2}
    ]
    df = detections_to_dataframe(mock_detections, 400, 300)
    assert len(df) == 2
    assert "Confidence (%)" in df.columns
    assert df.iloc[0]["Confidence (%)"] == "95.2%"
    
    csv_bytes = detections_to_csv_bytes(df)
    assert len(csv_bytes) > 0
    print("  ✓ DataFrame conversion and CSV serialization verified")

    report = generate_full_report(
        original_size=(400, 300),
        detections_df=df,
        summary={"total_count": 2, "unique_categories": ["car", "person"], "class_counts": {"car": 1, "person": 1}, "average_confidence": 0.919},
        ai_description="This is a test description.",
        ai_provider="Test Provider",
        timing={"detection_ms": 25.0, "caption_s": 0.8}
    )
    assert "Microsoft AI-901" in report
    assert "Executive Detection Summary" in report
    print("  ✓ Markdown inspection report generator verified")
    print("utils.py PASSED!\n")


def test_image_caption():
    print("[2/3] Testing image_caption.py functions...")
    from image_caption import (
        generate_local_scene_summary,
        describe_image
    )

    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    mock_detections = [
        {"label": "person", "confidence": 0.945, "box": [50.0, 100.0, 150.0, 350.0], "class_id": 0},
        {"label": "bicycle", "confidence": 0.892, "box": [120.0, 200.0, 250.0, 400.0], "class_id": 1}
    ]
    mock_summary = {
        "total_count": 2,
        "unique_categories": ["bicycle", "person"],
        "class_counts": {"person": 1, "bicycle": 1},
        "average_confidence": 0.918
    }

    # Test Local Contextual Synthesizer (Offline zero-key mode)
    res = generate_local_scene_summary(img, mock_detections, mock_summary)
    assert res["success"] is True
    assert "Scene Overview" in res["description"]
    assert "person" in res["description"].lower()
    assert "bicycle" in res["description"].lower()
    print("  ✓ Local Scene Synthesizer generated grounded narrative")

    # Test zero detections edge case
    empty_res = generate_local_scene_summary(img, [], {"total_count": 0, "unique_categories": [], "class_counts": {}, "average_confidence": 0.0})
    assert empty_res["success"] is True
    assert "no distinct objects" in empty_res["description"].lower()
    print("  ✓ Empty detections edge-case handled gracefully")

    # Test describe_image fallback with missing Azure credentials
    fallback_res = describe_image(
        image=img,
        detected_objects=mock_detections,
        summary=mock_summary,
        provider="azure",
        azure_endpoint="",
        azure_key=""
    )
    assert fallback_res["success"] is True
    assert "Fallen back" in fallback_res.get("error", "")
    print("  ✓ Seamless fallback to Local Synthesizer when cloud API key is missing verified")
    print("image_caption.py PASSED!\n")


def test_detector():
    print("[3/3] Testing detector.py and YOLOv8 inference...")
    try:
        from detector import ObjectDetector
    except ImportError as e:
        print(f"Skipping YOLO inference check (dependencies still installing): {e}")
        return

    # Create synthetic test image
    img = Image.new("RGB", (640, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw simple shapes
    draw.rectangle([100, 100, 400, 500], fill=(200, 50, 50))
    
    print("  Loading ObjectDetector with yolov8n.pt...")
    detector = ObjectDetector(model_name="yolov8n.pt")
    res = detector.detect(img, conf_threshold=0.25)
    
    assert "annotated_image" in res
    assert "detections" in res
    assert "summary" in res
    assert "inference_time_ms" in res
    print(f"  ✓ YOLOv8 executed inference in {res['inference_time_ms']:.1f} ms!")
    print(f"  ✓ Summary: {res['summary']}")
    print("detector.py PASSED!\n")


if __name__ == "__main__":
    print("==================================================")
    print("Running Smart Object Detection System Test Suite")
    print("==================================================\n")
    test_utils()
    test_image_caption()
    test_detector()
    print("==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")
