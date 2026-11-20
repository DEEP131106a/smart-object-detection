"""
utils.py
=============================================================================
Smart Object Detection and AI Image Description System
Utility functions for image processing, validation, formatting, and UI styling.
=============================================================================
"""

import io
import base64
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


# Supported image MIME types and extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_MB_DEFAULT = 15.0


def validate_image_file(
    uploaded_file,
    max_size_mb: float = MAX_FILE_SIZE_MB_DEFAULT
) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded file format, size, and image integrity.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object.
        max_size_mb: Maximum allowed file size in Megabytes.

    Returns:
        Tuple of (is_valid: bool, error_message: str | None, pil_image: Image.Image | None)
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    # Check filename extension
    filename = getattr(uploaded_file, "name", "upload.jpg").lower()
    _, ext = os.path.splitext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file format '{ext}'. Allowed formats are JPG, JPEG, PNG, WEBP, and BMP.",
            None,
        )

    # Check file size
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return (
            False,
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB.",
            None,
        )

    # Verify PIL image integrity
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Handle EXIF orientation if present
        image = ImageOps.exif_transpose(image)
        # Convert to RGB (handles RGBA, grayscale, CMYK gracefully)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Quick load check
        image.load()
        return True, None, image
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}", None


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL RGB Image to an OpenCV BGR numpy array."""
    rgb_array = np.array(image)
    return rgb_array[:, :, ::-1].copy()


def cv2_to_pil(image_cv: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR numpy array to a PIL RGB Image."""
    rgb_array = image_cv[:, :, ::-1]
    return Image.fromarray(rgb_array)


def pil_to_base64_data_uri(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL Image to a base64 data URI string for API transmission."""
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() in ["JPEG", "JPG"] else f"image/{format.lower()}"
    return f"data:{mime};base64,{img_b64}"


def optimize_image_for_api(image: Image.Image, max_dim: int = 1536) -> Image.Image:
    """
    Downsamples the image proportionally if either dimension exceeds max_dim.
    Reduces API payload size and network latency while maintaining visual clarity.
    """
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image
    
    scaling_factor = min(max_dim / width, max_dim / height)
    new_width = max(1, int(width * scaling_factor))
    new_height = max(1, int(height * scaling_factor))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_spatial_position(box: List[float], img_w: int, img_h: int) -> str:
    """
    Translates bounding box coordinates [x1, y1, x2, y2] to a human-friendly spatial description.
    Example: 'Center', 'Top-Left', 'Bottom-Right'.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Horizontal position
    if cx < img_w * 0.35:
        h_pos = "Left"
    elif cx > img_w * 0.65:
        h_pos = "Right"
    else:
        h_pos = "Center"

    # Vertical position
    if cy < img_h * 0.35:
        v_pos = "Top"
    elif cy > img_h * 0.65:
        v_pos = "Bottom"
    else:
        v_pos = "Middle"

    if h_pos == "Center" and v_pos == "Middle":
        return "Center-Foreground"
    return f"{v_pos}-{h_pos}"


def detections_to_dataframe(detections: List[Dict[str, Any]], img_w: int, img_h: int) -> pd.DataFrame:
    """
    Converts raw detection list into a clean, formatted Pandas DataFrame.
    """
    if not detections:
        return pd.DataFrame(columns=[
            "Index", "Object Label", "Confidence Score", "Confidence (%)", "Location", "Bounding Box"
        ])

    rows = []
    for idx, det in enumerate(detections, start=1):
        box = det["box"]
        conf = det["confidence"]
        loc = get_spatial_position(box, img_w, img_h)
        box_str = f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
        
        rows.append({
            "Index": idx,
            "Object Label": det["label"].capitalize(),
            "Confidence Score": round(conf, 4),
            "Confidence (%)": f"{conf * 100:.1f}%",
            "Location": loc,
            "Bounding Box": box_str
        })
    return pd.DataFrame(rows)


def detections_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Returns UTF-8 CSV bytes from the detection DataFrame."""
    return df.to_csv(index=False).encode("utf-8")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Encodes PIL Image into downloadable byte array."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame into a clean Markdown table string
    without requiring external optional dependencies like tabulate.
    """
    if df.empty:
        return "_No records to display._"
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---" for _ in headers]) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_line = "| " + " | ".join(str(row[col]) for col in df.columns) + " |"
        data_lines.append(row_line)
    return "\n".join([header_line, sep_line] + data_lines)


def generate_full_report(
    original_size: Tuple[int, int],
    detections_df: pd.DataFrame,
    summary: Dict[str, Any],
    ai_description: str,
    ai_provider: str,
    timing: Dict[str, float]
) -> str:
    """
    Generates a comprehensive Markdown inspection report for download.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append("# Smart Object Detection & AI Image Description System")
    report.append(f"**Inspection Timestamp:** {timestamp}")
    report.append(f"**Image Dimensions:** {original_size[0]} x {original_size[1]} pixels\n")
    report.append("---")
    report.append("## 1. Executive Detection Summary")
    report.append(f"- **Total Objects Detected:** {summary.get('total_count', 0)}")
    report.append(f"- **Unique Object Categories:** {len(summary.get('unique_categories', []))} ({', '.join(summary.get('unique_categories', []))})")
    report.append(f"- **Average Confidence Score:** {summary.get('average_confidence', 0.0) * 100:.1f}%")
    report.append(f"- **Object Detection Latency:** {timing.get('detection_ms', 0):.1f} ms")
    report.append(f"- **AI Description Latency:** {timing.get('caption_s', 0):.2f} s\n")

    report.append("## 2. Object Frequency Breakdown")
    for obj, count in summary.get("class_counts", {}).items():
        report.append(f"- **{obj.capitalize()}**: {count} instance{'s' if count > 1 else ''}")
    report.append("")

    report.append("## 3. Detected Objects Detail Table")
    if not detections_df.empty:
        report.append(dataframe_to_markdown_table(detections_df))
    else:
        report.append("_No objects detected above the selected confidence threshold._")
    report.append("\n---")

    report.append("## 4. Generative AI Scene Description")
    report.append(f"**AI Engine / Provider:** {ai_provider.upper()}\n")
    report.append(f"> {ai_description}\n")
    report.append("---")
    report.append("## 5. Microsoft AI-901 Syllabus Alignment")
    report.append("- **Domain:** Computer Vision & Image Analysis")
    report.append("- **Tasks:** Object Detection (Bounding Box Localization + Classification), Scene Understanding (Vision-Language Generative AI)")
    report.append("- **Principles:** Responsible AI (Transparency, Reliability, Safety & Fallback Resilience)")
    report.append("\n_Report automatically generated by Smart Object Detection System._")

    return "\n".join(report)


def inject_custom_css():
    """Injects modern, polished CSS styles for a professional Streamlit dashboard."""
    st.markdown(
        """
        <style>
        /* Modern aesthetic styling */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Glassmorphism & card containers */
        .metric-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }

        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888888;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #1E88E5;
            margin: 0;
        }

        .badge-ai901 {
            display: inline-block;
            background: linear-gradient(90deg, #0078D4 0%, #00B4FF 100%);
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-right: 8px;
            box-shadow: 0 2px 6px rgba(0, 120, 212, 0.3);
        }

        .badge-model {
            display: inline-block;
            background: #2E7D32;
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .description-box {
            background: linear-gradient(135deg, rgba(0, 120, 212, 0.06) 0%, rgba(0, 180, 255, 0.03) 100%);
            border-left: 4px solid #0078D4;
            border-radius: 8px;
            padding: 18px 22px;
            margin: 12px 0 20px 0;
            line-height: 1.65;
            font-size: 1.05rem;
        }

        /* Clean tab and button tweaks */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stDownloadButton button {
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }

        code {
            font-family: 'JetBrains Mono', monospace;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
