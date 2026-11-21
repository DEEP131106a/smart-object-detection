"""
detector.py
=============================================================================
Smart Object Detection and AI Image Description System
Object detection module powered by Ultralytics YOLOv8.
Handles model loading, inference, bounding box visualization, and statistics.
=============================================================================
"""

import time
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

# Ensure Ultralytics configuration directory is set to a local writable location
if "YOLO_CONFIG_DIR" not in os.environ:
    local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics")
    try:
        os.makedirs(local_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = local_cfg
    except Exception:
        os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()

# Distinct high-contrast color palette for aesthetic bounding boxes
PALETTE = [
    (235, 87, 87),   # Coral Red
    (39, 174, 96),   # Emerald Green
    (47, 128, 237),  # Azure Blue
    (242, 153, 74),  # Tangerine
    (155, 81, 224),  # Amethyst Purple
    (0, 184, 212),   # Cyan
    (245, 0, 87),    # Vivid Pink
    (255, 214, 0),   # Lemon Yellow
    (0, 200, 83),    # Bright Green
    (101, 31, 255),  # Deep Indigo
    (255, 109, 0),   # Amber Orange
    (213, 0, 249),   # Magenta
    (0, 229, 255),   # Light Cyan
    (255, 82, 82),   # Soft Red
    (100, 221, 23),  # Lime Green
    (98, 0, 234),    # Electric Violet
]


class ObjectDetector:
    """
    Wrapper around Ultralytics YOLOv8 for object detection.
    Encapsulates model initialization, inference, bounding box annotation, and metrics.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initializes the YOLO model.
        Args:
            model_name: Path or identifier for YOLO model (defaults to lightweight YOLOv8 nano).
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the YOLO model into memory."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {str(e)}")

    def get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Returns a deterministic RGB color from the palette for a given class ID."""
        color = PALETTE[class_id % len(PALETTE)]
        return color

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        draw_boxes: bool = True,
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs object detection on the provided PIL image.

        Args:
            image: Input PIL Image (RGB).
            conf_threshold: Minimum confidence score (0.0 to 1.0) to filter detections.
            iou_threshold: Intersection-over-Union threshold for Non-Maximum Suppression (NMS).
            draw_boxes: Whether to draw bounding boxes and labels onto the image.
            line_thickness: Custom bounding box border thickness (auto-calculated if None).
            font_scale: Custom text font scale (auto-calculated if None).

        Returns:
            Dict containing:
                - 'annotated_image': PIL.Image with rendered annotations
                - 'detections': List of detection dicts (label, confidence, box, class_id)
                - 'summary': Dict of aggregate metrics (total, unique categories, counts, avg confidence)
                - 'inference_time_ms': Inference time in milliseconds
        """
        if self.model is None:
            self._load_model()

        # Convert PIL to RGB numpy array for YOLO
        img_np = np.array(image.convert("RGB"))
        img_h, img_w = img_np.shape[:2]

        start_time = time.perf_counter()
        
        # Run YOLO inference
        results = self.model.predict(
            source=img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}
        confidences: List[float] = []

        # Parse detection results
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy_tensor = boxes.xyxy.cpu().numpy()
            conf_tensor = boxes.conf.cpu().numpy()
            cls_tensor = boxes.cls.cpu().numpy().astype(int)
            names_dict = result.names

            for i in range(len(xyxy_tensor)):
                box = xyxy_tensor[i].tolist()  # [x1, y1, x2, y2]
                conf = float(conf_tensor[i])
                class_id = int(cls_tensor[i])
                label = names_dict.get(class_id, f"class_{class_id}")

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [round(coord, 1) for coord in box],
                    "class_id": class_id,
                })

                class_counts[label] = class_counts.get(label, 0) + 1
                confidences.append(conf)

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Compute summary metrics
        total_count = len(detections)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        unique_categories = sorted(list(class_counts.keys()))

        summary = {
            "total_count": total_count,
            "unique_categories": unique_categories,
            "category_count": len(unique_categories),
            "class_counts": class_counts,
            "average_confidence": avg_confidence,
        }

        # Annotate image with high-visibility aesthetic bounding boxes
        if draw_boxes:
            annotated_pil = self._draw_annotations(
                image=image,
                detections=detections,
                line_thickness=line_thickness,
                font_scale=font_scale
            )
        else:
            annotated_pil = image.copy()

        return {
            "annotated_image": annotated_pil,
            "detections": detections,
            "summary": summary,
            "inference_time_ms": inference_time_ms,
        }

    def _draw_annotations(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None
    ) -> Image.Image:
        """
        Draws high-contrast, professional bounding boxes with pill-style labels.
        Works in OpenCV BGR space and converts back to PIL RGB.
        """
        # Convert PIL to BGR array for OpenCV drawing
        rgb_array = np.array(image.convert("RGB"))
        cv_img = rgb_array[:, :, ::-1].copy()
        h, w = cv_img.shape[:2]

        # Auto-compute dynamic scale based on image resolution
        diag = np.sqrt(w ** 2 + h ** 2)
        thickness = line_thickness if line_thickness is not None else max(2, int(diag / 500))
        scale = font_scale if font_scale is not None else max(0.45, min(1.0, diag / 1600))
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            class_id = det["class_id"]
            # Get RGB from palette and convert to BGR for OpenCV
            r, g, b = self.get_color_for_class(class_id)
            color_bgr = (b, g, r)

            # Draw outer bounding box
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color_bgr, thickness, lineType=cv2.LINE_AA)

            # Label text: e.g. "Person 94.2%"
            conf_pct = det["confidence"] * 100.0
            label_text = f"{det['label'].capitalize()} {conf_pct:.1f}%"

            # Measure text size for pill background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, scale, max(1, thickness - 1))

            # Determine pill background position (above box if space allows, otherwise inside)
            pad_x = 6
            pad_y = 4
            pill_y1 = y1 - text_h - (pad_y * 2) - 2
            pill_y2 = y1

            if pill_y1 < 0:
                # Place label inside the box if it overflows image top
                pill_y1 = y1
                pill_y2 = y1 + text_h + (pad_y * 2) + 2

            pill_x1 = x1
            pill_x2 = min(w - 1, x1 + text_w + (pad_x * 2))

            # Draw solid pill banner for maximum legibility
            cv2.rectangle(
                cv_img,
                (pill_x1, pill_y1),
                (pill_x2, pill_y2),
                color_bgr,
                cv2.FILLED
            )

            # Text color: white or black depending on color luminance
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)

            # Put text
            text_x = pill_x1 + pad_x
            text_y = pill_y2 - pad_y - 2
            cv2.putText(
                cv_img,
                label_text,
                (text_x, text_y),
                font,
                scale,
                text_color,
                thickness=max(1, int(thickness / 2)),
                lineType=cv2.LINE_AA
            )

        # Convert back to PIL RGB
        annotated_rgb = cv_img[:, :, ::-1]
        return Image.fromarray(annotated_rgb)


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("🔍 YOLOv8 Standalone Object Detector Test")
    print("=" * 60)

    # 1. Initialize detector
    print("1. Loading YOLOv8 nano model (yolov8n.pt)...")
    detector = ObjectDetector(model_name="yolov8n.pt")
    print("   ✓ Model successfully loaded into memory!")

    # 2. Determine target image
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
        print(f"2. Loading input image: {image_path}")
        test_img = Image.open(image_path).convert("RGB")
    else:
        print("2. Generating test visual scene...")
        from PIL import ImageDraw
        test_img = Image.new("RGB", (800, 600), color=(240, 244, 248))
        draw = ImageDraw.Draw(test_img)
        # Background elements
        draw.rectangle([0, 350, 800, 600], fill=(70, 70, 70))
        # Car representation
        draw.rectangle([180, 320, 500, 480], fill=(220, 40, 40))
        draw.rectangle([230, 260, 450, 320], fill=(160, 200, 240))
        draw.ellipse([210, 450, 270, 510], fill=(20, 20, 20))
        draw.ellipse([410, 450, 470, 510], fill=(20, 20, 20))
        # Person representation
        draw.ellipse([580, 280, 620, 320], fill=(240, 200, 160))
        draw.rectangle([580, 320, 620, 440], fill=(40, 80, 200))

    # 3. Execute detection
    print("3. Executing inference (Confidence: 0.25, IoU: 0.45)...")
    result = detector.detect(test_img, conf_threshold=0.25, iou_threshold=0.45)

    print(f"   ✓ Inference completed in {result['inference_time_ms']:.1f} ms!")
    summary = result["summary"]
    print(f"   ✓ Total objects detected: {summary['total_count']}")
    print(f"   ✓ Unique categories: {summary['unique_categories']}")
    print(f"   ✓ Average confidence: {summary['average_confidence']*100:.1f}%\n")

    if result["detections"]:
        print("4. Detection Details:")
        for idx, det in enumerate(result["detections"], 1):
            print(f"   [{idx}] {det['label'].capitalize()} - {det['confidence']*100:.1f}% | Box: {det['box']}")
    else:
        print("   (No objects met the 0.25 confidence threshold)")

    # 4. Save output
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "standalone_detection_test.jpg")
    result["annotated_image"].save(out_file)
    print(f"\n5. Annotated image saved to: {out_file}")
    print("=" * 60)
    print("✨ Standalone test completed successfully!")
    print("=" * 60)
