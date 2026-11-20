"""
app.py
=============================================================================
Smart Object Detection and AI Image Description System
Main Streamlit Application Dashboard.
Integrated with Ultralytics YOLOv8 and Generative AI Vision Models.
Compliant with Microsoft AI-901 Syllabus Concepts.
=============================================================================
"""

import os
import tempfile
from typing import Optional
from PIL import Image
import pandas as pd
import streamlit as st

# Ensure Ultralytics config is stored in a local writable directory
if "YOLO_CONFIG_DIR" not in os.environ:
    local_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics")
    try:
        os.makedirs(local_cfg, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = local_cfg
    except Exception:
        os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()

# Local modules
from detector import ObjectDetector
from image_caption import describe_image
from utils import (
    validate_image_file,
    detections_to_dataframe,
    detections_to_csv_bytes,
    image_to_bytes,
    generate_full_report,
    inject_custom_css,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Aesthetic Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Object Detection & AI Image Description",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern CSS
inject_custom_css()

# Ensure outputs directory exists
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector(model_name: str = "yolov8n.pt") -> ObjectDetector:
    """Loads and caches the YOLO object detector."""
    return ObjectDetector(model_name=model_name)


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = None


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logomark_Reverse.png",
        width=60,
    )
    st.title("System Control")
    st.markdown("Configure computer vision and generative AI inference parameters.")

    st.markdown("---")
    st.subheader("🎯 Object Detection (YOLOv8)")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", 0.25)),
        step=0.05,
        help="Minimum confidence probability for YOLO to register an object detection."
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.45)),
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression to filter duplicate overlapping boxes."
    )

    with st.expander("🎨 Visual Box Styling", expanded=False):
        draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)
        line_thickness = st.slider("Box Border Thickness", min_value=1, max_value=8, value=3)
        font_scale = st.slider("Label Font Scale", min_value=0.3, max_value=1.5, value=0.6, step=0.05)

    st.markdown("---")
    st.subheader("🤖 Generative AI Description")

    ai_provider_options = [
        "Local Contextual Synthesizer (Zero-Key)",
        "Azure AI Foundry (GPT-4o / Phi-3.5)",
        "OpenAI (GPT-4o / GPT-4o-mini)",
        "Google Gemini (Gemini 1.5 Flash)",
    ]
    
    default_provider_env = os.getenv("DEFAULT_AI_PROVIDER", "local").lower()
    default_idx = 0
    if "azure" in default_provider_env:
        default_idx = 1
    elif "openai" in default_provider_env:
        default_idx = 2
    elif "gemini" in default_provider_env:
        default_idx = 3

    selected_provider_label = st.selectbox(
        "Vision-Language Provider",
        options=ai_provider_options,
        index=default_idx,
        help="Choose the Generative AI provider to formulate natural-language scene descriptions."
    )

    # Provider credential inputs
    azure_endpoint = None
    azure_key = None
    azure_model = None
    openai_key = None
    openai_model = None
    gemini_key = None
    gemini_model = None

    if "Azure" in selected_provider_label:
        provider_key = "azure"
        st.info("Configuring Azure AI Foundry / Azure OpenAI endpoint.")
        azure_endpoint = st.text_input(
            "Azure Endpoint URL",
            value=os.getenv("AZURE_AI_ENDPOINT", ""),
            type="default",
            placeholder="https://<endpoint>.services.ai.azure.com/models/chat/completions"
        )
        azure_key = st.text_input(
            "Azure API Key",
            value=os.getenv("AZURE_AI_KEY", ""),
            type="password",
            help="Your Azure AI Foundry or Azure OpenAI resource key."
        )
        azure_model = st.text_input(
            "Model / Deployment",
            value=os.getenv("AZURE_AI_MODEL", "gpt-4o"),
            placeholder="gpt-4o"
        )

    elif "OpenAI" in selected_provider_label:
        provider_key = "openai"
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-..."
        )
        openai_model = st.text_input(
            "Model Name",
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            placeholder="gpt-4o-mini"
        )

    elif "Gemini" in selected_provider_label:
        provider_key = "gemini"
        gemini_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            placeholder="AIzaSy..."
        )
        gemini_model = st.text_input(
            "Model Name",
            value=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            placeholder="gemini-1.5-flash"
        )

    else:
        provider_key = "local"
        st.success("✅ **Zero-Key Local Mode Active**: Functions 100% offline with spatial scene heuristics!")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #888;">
        <strong>Microsoft AI-901 Mini-Project</strong><br>
        Demonstrating: Computer Vision, Object Localization, Confidence Metrics & Multimodal Generative AI.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 5. Main App Header & Educational Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <span class="badge-ai901">Microsoft AI-901 Certified</span>
        <span class="badge-model">YOLOv8n Real-Time Engine</span>
        <h1 style="margin: 10px 0 6px 0; font-size: 2.3rem; font-weight: 800;">
            Smart Object Detection & AI Image Description System
        </h1>
        <p style="font-size: 1.05rem; color: #888; margin-bottom: 0;">
            Upload an image to localize multiple objects with high-precision bounding boxes, quantify confidence metrics,
            and generate an intelligent natural-language scene understanding description using Generative AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Educational Expander on AI-901 Concepts
with st.expander("📘 Microsoft AI-901 Syllabus Concept Mapping & Architecture", expanded=False):
    st.markdown(
        """
        This project directly illustrates fundamental learning pillars of the **Microsoft Certified: Azure AI Fundamentals (AI-901)** syllabus:

        | Syllabus Module | Fundamental Concept | How It Is Implemented Here |
        | :--- | :--- | :--- |
        | **Computer Vision Fundamentals** | Image processing & visual feature extraction | OpenCV & Pillow decode image matrices, calculate spatial aspect ratios, and format tensor data. |
        | **Object Detection vs Classification** | Detecting *what* is in the image and *where* it is located | Pretrained YOLOv8 identifies individual object classes and returns precise bounding box coordinates `[x1, y1, x2, y2]`. |
        | **Confidence & NMS** | Evaluating model certainty & resolving duplicates | Users adjust confidence thresholds and Non-Maximum Suppression (IoU) to filter spurious predictions. |
        | **Multimodal Generative AI** | Scene understanding & vision-language reasoning | Generative AI models (Azure AI Foundry / OpenAI / Gemini / Synthesizer) synthesize holistic narratives explaining relationships between detected entities. |
        | **Responsible AI Principles** | Transparency, reliability, safety & fallback | System provides explicit confidence scores, execution latencies, and a local offline fallback engine when cloud APIs are unreachable. |
        """
    )


# -----------------------------------------------------------------------------
# 6. File Upload & Image Acquisition Section
# -----------------------------------------------------------------------------
st.markdown("### 📤 Step 1: Upload or Choose an Image")

col_upload, col_sample = st.columns([2.5, 1.5])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG, WEBP, BMP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload an image up to 15MB. The image will be validated before processing."
    )

with col_sample:
    st.markdown("**Or load a built-in sample scene:**")
    sample_choice = st.selectbox(
        "Select Sample Image",
        options=["(Custom Upload)", "Urban Street & Vehicles", "Domestic Living Room", "Workstation & Tech"],
        index=0
    )

# Sample image generators if user chooses sample
def create_sample_image(sample_type: str) -> Image.Image:
    """Generates synthetic high-quality test images with recognizable shapes for immediate testing."""
    from PIL import ImageDraw
    img = Image.new("RGB", (800, 600), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)

    if sample_type == "Urban Street & Vehicles":
        # Sky & road
        draw.rectangle([0, 0, 800, 350], fill=(135, 206, 235))
        draw.rectangle([0, 350, 800, 600], fill=(80, 80, 80))
        # Road lane lines
        for x in range(50, 800, 150):
            draw.rectangle([x, 470, x + 80, 480], fill=(255, 255, 255))
        # Car representation
        draw.rectangle([200, 360, 480, 480], fill=(220, 50, 50))
        draw.rectangle([240, 320, 440, 360], fill=(180, 220, 255))
        draw.ellipse([230, 460, 290, 520], fill=(30, 30, 30))
        draw.ellipse([390, 460, 450, 520], fill=(30, 30, 30))
        # Person representation
        draw.ellipse([580, 280, 620, 320], fill=(240, 200, 160))
        draw.rectangle([580, 320, 620, 440], fill=(30, 80, 180))
        draw.line([590, 440, 590, 510], fill=(40, 40, 40), width=6)
        draw.line([610, 440, 610, 510], fill=(40, 40, 40), width=6)

    elif sample_type == "Domestic Living Room":
        # Room wall & floor
        draw.rectangle([0, 0, 800, 420], fill=(230, 230, 220))
        draw.rectangle([0, 420, 800, 600], fill=(160, 110, 70))
        # TV / Monitor on wall
        draw.rectangle([280, 80, 520, 240], fill=(20, 20, 20))
        draw.rectangle([290, 90, 510, 230], fill=(50, 120, 200))
        # Couch
        draw.rectangle([200, 340, 600, 470], fill=(70, 130, 180))
        draw.rectangle([180, 370, 220, 470], fill=(50, 100, 150))
        draw.rectangle([580, 370, 620, 470], fill=(50, 100, 150))
        # Potted plant
        draw.rectangle([70, 320, 130, 440], fill=(180, 100, 60))
        draw.ellipse([50, 240, 150, 340], fill=(46, 139, 87))

    else:  # Workstation & Tech
        # Desk surface
        draw.rectangle([0, 0, 800, 600], fill=(245, 245, 240))
        draw.rectangle([50, 100, 750, 580], fill=(210, 180, 140))
        # Laptop
        draw.rectangle([250, 180, 550, 380], fill=(160, 160, 160))
        draw.rectangle([260, 190, 540, 370], fill=(30, 30, 40))
        draw.rectangle([220, 380, 580, 440], fill=(190, 190, 190))
        # Cup / Mug
        draw.ellipse([620, 280, 700, 360], fill=(255, 255, 255))
        draw.ellipse([630, 290, 690, 350], fill=(100, 60, 30))
        # Cell phone
        draw.rectangle([110, 260, 180, 380], fill=(40, 40, 40))

    return img


# Image Validation & Selection Logic
current_image: Optional[Image.Image] = None
filename = "sample.jpg"

if sample_choice != "(Custom Upload)":
    current_image = create_sample_image(sample_choice)
    filename = f"{sample_choice.lower().replace(' ', '_')}.png"
elif uploaded_file is not None:
    is_valid, err_msg, loaded_img = validate_image_file(uploaded_file)
    if not is_valid:
        st.error(f"❌ **Validation Error**: {err_msg}")
    else:
        current_image = loaded_img
        filename = uploaded_file.name

# Synchronize with session state
if current_image is not None and (
    st.session_state.active_image is None or st.session_state.image_filename != filename
):
    st.session_state.active_image = current_image
    st.session_state.image_filename = filename
    st.session_state.analysis_results = None  # Reset prior results on new image


# -----------------------------------------------------------------------------
# 7. Image Preview & Action Controls
# -----------------------------------------------------------------------------
if st.session_state.active_image is not None:
    img = st.session_state.active_image
    w, h = img.size

    st.markdown("---")
    st.markdown(f"### 🖼️ Step 2: Image Preview (`{w} × {h}` px)")
    
    col_preview, col_actions = st.columns([2.5, 1.5])
    
    with col_preview:
        st.image(img, caption=f"Active Image: {st.session_state.image_filename} ({w}x{h} px)", use_container_width=True)

    with col_actions:
        st.markdown("#### Ready for Analysis")
        st.write(
            f"**File Name:** `{st.session_state.image_filename}`\n\n"
            f"**Resolution:** `{w} × {h}` pixels\n\n"
            f"**Confidence Threshold:** `{conf_threshold * 100:.0f}%`\n\n"
            f"**AI Description Provider:** `{selected_provider_label}`"
        )
        
        btn_analyze = st.button("🚀 Analyze Image", type="primary", use_container_width=True)
        
        btn_reset = st.button("🔄 Reset Image & Results", use_container_width=True)
        if btn_reset:
            st.session_state.active_image = None
            st.session_state.image_filename = None
            st.session_state.analysis_results = None
            st.rerun()

    # -------------------------------------------------------------------------
    # 8. Execution Pipeline (Object Detection + AI Scene Description)
    # -------------------------------------------------------------------------
    if btn_analyze:
        with st.status("🔍 Processing Image Pipeline...", expanded=True) as status:
            # 1. Load YOLO model
            st.write("Loading YOLOv8 object detection model...")
            detector = get_detector("yolov8n.pt")

            # 2. Run Object Detection
            st.write(f"Executing object detection (Confidence threshold: {conf_threshold:.2f})...")
            detection_res = detector.detect(
                image=img,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                draw_boxes=draw_boxes,
                line_thickness=line_thickness,
                font_scale=font_scale
            )
            det_time = detection_res["inference_time_ms"]
            detections = detection_res["detections"]
            summary = detection_res["summary"]
            st.write(f"✅ Object detection complete in **{det_time:.1f} ms**! Found **{summary['total_count']}** objects.")

            # 3. Run Generative AI Scene Description
            st.write(f"Synthesizing natural-language description using **{selected_provider_label}**...")
            ai_res = describe_image(
                image=img,
                detected_objects=detections,
                summary=summary,
                provider=provider_key,
                azure_endpoint=azure_endpoint,
                azure_key=azure_key,
                azure_model=azure_model,
                openai_key=openai_key,
                openai_model=openai_model,
                gemini_key=gemini_key,
                gemini_model=gemini_model,
            )
            caption_time = ai_res.get("latency_seconds", 0.0)

            if ai_res.get("error"):
                st.warning(f"⚠️ {ai_res['error']}")
            st.write(f"✅ Scene description generated in **{caption_time:.2f} s**.")

            # Save state
            st.session_state.analysis_results = {
                "detection_res": detection_res,
                "ai_res": ai_res,
                "timing": {
                    "detection_ms": det_time,
                    "caption_s": caption_time
                }
            }
            status.update(label="🎉 Analysis Complete!", state="complete", expanded=False)


# -----------------------------------------------------------------------------
# 9. Results Dashboard
# -----------------------------------------------------------------------------
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    detection_res = results["detection_res"]
    ai_res = results["ai_res"]
    timing = results["timing"]
    summary = detection_res["summary"]
    detections = detection_res["detections"]
    annotated_img = detection_res["annotated_image"]
    img = st.session_state.active_image
    w, h = img.size

    # Convert to DataFrame
    df_detections = detections_to_dataframe(detections, w, h)

    st.markdown("---")
    st.markdown("## 📊 Detection & Scene Analysis Dashboard")

    # Metrics Summary Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Detected Objects</div>
                <div class="metric-value">{summary['total_count']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Unique Categories</div>
                <div class="metric-value">{summary['category_count']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        avg_conf_pct = summary["average_confidence"] * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Confidence</div>
                <div class="metric-value">{avg_conf_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Processing Time</div>
                <div class="metric-value" style="color: #2E7D32;">{timing['detection_ms']:.0f}ms <span style="font-size: 1rem; color: #888;">+ {timing['caption_s']:.1f}s</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Dashboard Tabs
    tab_visual, tab_ai, tab_analytics, tab_data = st.tabs([
        "🖼️ Visual Detection",
        "🧠 AI Image Description",
        "📈 Detection Analytics",
        "📋 Results Table & Export"
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Visual Comparison
    # -------------------------------------------------------------------------
    with tab_visual:
        st.markdown("#### Side-by-Side Visual Comparison")
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("**Original Input Image**")
            st.image(img, use_container_width=True)
        with col_annot:
            st.markdown(f"**Detected Objects ({summary['total_count']}) with Bounding Boxes**")
            st.image(annotated_img, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 2: AI Generative Scene Description
    # -------------------------------------------------------------------------
    with tab_ai:
        st.markdown("#### Generative AI Scene Understanding")
        provider_badge = ai_res.get("provider", "Generative AI")
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span class="badge-ai901">Generative AI</span>
                <span class="badge-model">{provider_badge}</span>
                <span style="font-size: 0.85rem; color: #888;">Latency: {timing['caption_s']:.2f}s</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="description-box">
                {ai_res.get('description', 'No description generated.')}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "💡 **AI-901 Alignment**: This description leverages multimodal Generative AI to formulate "
            "coherent natural language summarizing visual attributes, spatial relationships, and contextual intent."
        )

    # -------------------------------------------------------------------------
    # Tab 3: Analytics & Category Distribution
    # -------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("#### Detection Frequency by Category")
        if summary["class_counts"]:
            chart_data = pd.DataFrame([
                {"Category": k.capitalize(), "Occurrences": v}
                for k, v in summary["class_counts"].items()
            ]).sort_values("Occurrences", ascending=False)
            
            st.bar_chart(chart_data.set_index("Category"))
            
            # Confidence spread
            st.markdown("#### Confidence Distribution")
            conf_df = pd.DataFrame([
                {"Object": d["label"].capitalize(), "Confidence (%)": round(d["confidence"] * 100, 1)}
                for d in detections
            ])
            st.line_chart(conf_df.set_index("Object"))
        else:
            st.warning("No objects were detected at the selected confidence threshold.")

    # -------------------------------------------------------------------------
    # Tab 4: Results Table & Export Center
    # -------------------------------------------------------------------------
    with tab_data:
        st.markdown("#### Complete Object Detection Registry")
        if not df_detections.empty:
            st.dataframe(df_detections, use_container_width=True, hide_index=True)
        else:
            st.info("No detections registered.")

        st.markdown("---")
        st.markdown("### 📥 Download & Export Center")
        
        col_down1, col_down2, col_down3 = st.columns(3)
        
        with col_down1:
            img_bytes = image_to_bytes(annotated_img, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image (PNG)",
                data=img_bytes,
                file_name=f"annotated_{st.session_state.image_filename}",
                mime="image/png",
                use_container_width=True
            )

        with col_down2:
            csv_bytes = detections_to_csv_bytes(df_detections)
            st.download_button(
                label="📊 Download Detection Results (CSV)",
                data=csv_bytes,
                file_name=f"detections_{st.session_state.image_filename}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_down3:
            report_md = generate_full_report(
                original_size=(w, h),
                detections_df=df_detections,
                summary=summary,
                ai_description=ai_res.get("description", "N/A"),
                ai_provider=ai_res.get("provider", "Unknown"),
                timing=timing
            )
            st.download_button(
                label="📄 Download Full Inspection Report (MD)",
                data=report_md.encode("utf-8"),
                file_name=f"inspection_report_{st.session_state.image_filename}.md",
                mime="text/markdown",
                use_container_width=True
            )

else:
    # Empty State Guide
    st.info(
        "👋 **Welcome!** Upload an image file or pick one of the sample scenes from the dropdown above, "
        "then click **'🚀 Analyze Image'** to begin object detection and AI description generation."
    )
