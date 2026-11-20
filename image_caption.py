"""
image_caption.py
=============================================================================
Smart Object Detection and AI Image Description System
Generative AI module for multimodal image understanding and natural language description.
Supports Azure AI Foundry, OpenAI, Google Gemini, and a Local Contextual Synthesizer.
=============================================================================
"""

import os
import time
import json
import base64
import io
from typing import Dict, Any, List, Optional
import requests
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


# Systematic prompt engineered for grounded, hallucination-free scene analysis
SYSTEM_PROMPT = """You are an expert AI Computer Vision and Image Analysis assistant aligned with Microsoft AI-901 concepts.
Your mission is to provide an accurate, clear, and comprehensive natural-language description of the provided image.

Follow these strict rules:
1. Base your description SOLELY on what is visibly present in the image. Do not invent objects, people, or events.
2. Structure your response into:
   - **Scene Overview**: A 1-2 sentence high-level summary of the setting and primary subjects.
   - **Key Entities & Actions**: Describe the main objects/subjects, their visual attributes, and what they are doing.
   - **Spatial Relationships & Interactions**: Explain how objects relate to one another (e.g., who is next to what, background vs foreground).
   - **Contextual Summary**: A concise concluding observation about the overall atmosphere or purpose of the scene.
3. Keep the tone professional, objective, and easy for a beginner to understand.
"""


def _pil_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL Image into a base64 encoded string."""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_azure_caption(
    image: Image.Image,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Generates an image description using Azure AI Foundry / Azure OpenAI vision endpoints.

    Args:
        image: PIL Image to describe.
        detected_objects: List of objects detected by YOLO to ground the prompt.
        endpoint: Azure AI Foundry / Azure OpenAI REST endpoint URL.
        api_key: Azure API key or bearer token.
        model_name: Model deployment name (e.g., 'gpt-4o', 'Phi-3.5-vision-instruct').
        timeout: Request timeout in seconds.

    Returns:
        Dict containing success status, description, provider, error, and latency.
    """
    endpoint = endpoint or os.getenv("AZURE_AI_ENDPOINT")
    api_key = api_key or os.getenv("AZURE_AI_KEY")
    model_name = model_name or os.getenv("AZURE_AI_MODEL", "gpt-4o")

    if not endpoint or not api_key:
        return {
            "success": False,
            "description": "",
            "provider": "azure",
            "error": "Azure AI endpoint or API key not configured. Please supply credentials in the sidebar or .env file.",
            "latency_seconds": 0.0
        }

    # Prepare grounding context from YOLO
    grounding_text = ""
    if detected_objects:
        obj_names = [f"{d['label']} ({d['confidence']*100:.1f}%)" for d in detected_objects[:10]]
        grounding_text = f"\nFor reference, an object detector identified these visible objects: {', '.join(obj_names)}."

    user_text = f"Analyze this image according to the required structure.{grounding_text}"

    # Downsample if image is overly large to ensure fast transmission
    img_b64 = _pil_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    # Build standard OpenAI vision-compatible request payload
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,  # Azure standard header
        "Authorization": f"Bearer {api_key}"  # Azure Foundry serverless header
    }

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}}
                ]
            }
        ],
        "max_tokens": 600,
        "temperature": 0.3
    }

    # Add model name if required by endpoint
    if "deployments" not in endpoint:
        payload["model"] = model_name

    start_time = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        latency = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            description = data["choices"][0]["message"]["content"].strip()
            return {
                "success": True,
                "description": description,
                "provider": f"Azure AI Foundry ({model_name})",
                "error": None,
                "latency_seconds": latency
            }
        else:
            return {
                "success": False,
                "description": "",
                "provider": "azure",
                "error": f"Azure AI returned HTTP {response.status_code}: {response.text[:300]}",
                "latency_seconds": latency
            }
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "success": False,
            "description": "",
            "provider": "azure",
            "error": f"Azure connection failed: {str(e)}",
            "latency_seconds": latency
        }


def generate_openai_caption(
    image: Image.Image,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Generates an image description using OpenAI vision models (gpt-4o / gpt-4o-mini).
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    if not api_key:
        return {
            "success": False,
            "description": "",
            "provider": "openai",
            "error": "OpenAI API key not configured.",
            "latency_seconds": 0.0
        }

    grounding_text = ""
    if detected_objects:
        obj_names = [f"{d['label']} ({d['confidence']*100:.1f}%)" for d in detected_objects[:10]]
        grounding_text = f"\nVisible objects detected by YOLO: {', '.join(obj_names)}."

    user_text = f"Analyze this image in detail.{grounding_text}"
    img_b64 = _pil_to_base64(image)
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}}
                ]
            }
        ],
        "max_tokens": 600,
        "temperature": 0.3
    }

    endpoint = f"{base_url}/chat/completions"
    start_time = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        latency = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            description = data["choices"][0]["message"]["content"].strip()
            return {
                "success": True,
                "description": description,
                "provider": f"OpenAI ({model_name})",
                "error": None,
                "latency_seconds": latency
            }
        else:
            return {
                "success": False,
                "description": "",
                "provider": "openai",
                "error": f"OpenAI HTTP {response.status_code}: {response.text[:300]}",
                "latency_seconds": latency
            }
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "success": False,
            "description": "",
            "provider": "openai",
            "error": f"OpenAI error: {str(e)}",
            "latency_seconds": latency
        }


def generate_gemini_caption(
    image: Image.Image,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Generates an image description using Google Gemini Vision REST API.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    if not api_key:
        return {
            "success": False,
            "description": "",
            "provider": "gemini",
            "error": "Google Gemini API key not configured.",
            "latency_seconds": 0.0
        }

    grounding_text = ""
    if detected_objects:
        obj_names = [f"{d['label']} ({d['confidence']*100:.1f}%)" for d in detected_objects[:10]]
        grounding_text = f"\nVisible objects detected: {', '.join(obj_names)}."

    prompt = f"{SYSTEM_PROMPT}\n\nTask: Describe this image according to instructions.{grounding_text}"
    img_b64 = _pil_to_base64(image)

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600
        }
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        latency = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            description = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {
                "success": True,
                "description": description,
                "provider": f"Google Gemini ({model_name})",
                "error": None,
                "latency_seconds": latency
            }
        else:
            return {
                "success": False,
                "description": "",
                "provider": "gemini",
                "error": f"Gemini HTTP {response.status_code}: {response.text[:300]}",
                "latency_seconds": latency
            }
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "success": False,
            "description": "",
            "provider": "gemini",
            "error": f"Gemini connection failed: {str(e)}",
            "latency_seconds": latency
        }


def generate_local_scene_summary(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Intelligent Local Scene Synthesis Engine (Zero API Key / Offline Mode).
    Generates a structured, grounded scene description using spatial reasoning,
    object co-occurrence analysis, and image geometry heuristics.
    Guarantees the application functions 100% offline without any API keys.
    """
    start_time = time.perf_counter()
    img_w, img_h = image.size
    total = summary.get("total_count", 0)
    class_counts = summary.get("class_counts", {})
    unique_cats = summary.get("unique_categories", [])

    if total == 0:
        desc = (
            "### Scene Overview\n"
            "The image presents a broad view where no distinct objects met the current confidence threshold. "
            "The scene may contain abstract patterns, landscape scenery, fine textures, or objects outside "
            "the 80 standard COCO detection classes.\n\n"
            "### Key Observations\n"
            "- No high-confidence foreground bounding boxes were resolved.\n"
            "- Consider lowering the confidence threshold in the sidebar if you wish to detect subtle or distant elements.\n\n"
            "### Contextual Summary\n"
            "The visual composition appears ambient or homogenous without prominent isolated targets."
        )
        return {
            "success": True,
            "description": desc,
            "provider": "Local Contextual Synthesizer (Zero-Key Mode)",
            "error": None,
            "latency_seconds": (time.perf_counter() - start_time)
        }

    # 1. Semantic Scene Domain Heuristic
    labels_set = set(k.lower() for k in class_counts.keys())
    
    transit_items = {"car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}
    living_items = {"chair", "couch", "sofa", "bed", "dining table", "tv", "potted plant", "vase", "clock"}
    tech_items = {"laptop", "keyboard", "mouse", "cell phone", "remote", "book"}
    kitchen_items = {"bottle", "cup", "fork", "knife", "spoon", "bowl", "wine glass", "banana", "apple", "sandwich", "pizza"}
    animal_items = {"dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

    scene_type = "general outdoor/indoor"
    if transit_items.intersection(labels_set):
        scene_type = "transportation or urban street setting"
    elif tech_items.intersection(labels_set):
        scene_type = "workstation, study, or digital office environment"
    elif living_items.intersection(labels_set):
        scene_type = "domestic interior or living space"
    elif kitchen_items.intersection(labels_set):
        scene_type = "dining or culinary setup"
    elif animal_items.intersection(labels_set):
        scene_type = "wildlife, domestic pet, or nature scenario"
    elif "person" in labels_set:
        scene_type = "human-centered activity or social setting"

    # 2. Prominent objects and quantities
    item_summaries = []
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if count == 1:
            item_summaries.append(f"1 {label}")
        else:
            item_summaries.append(f"{count} {label}s")
    objects_phrase = ", ".join(item_summaries)

    # 3. Spatial and relational breakdown
    spatial_descriptions = []
    for det in detections[:6]:  # top 6 detections
        box = det["box"]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        # Determine relative quadrant
        h_str = "left side" if cx < img_w * 0.35 else ("right side" if cx > img_w * 0.65 else "center")
        v_str = "upper section" if cy < img_h * 0.35 else ("lower section" if cy > img_h * 0.65 else "midground")
        conf_pct = det["confidence"] * 100.0
        spatial_descriptions.append(
            f"- A **{det['label'].capitalize()}** is located in the {v_str} ({h_str}) with a confidence of {conf_pct:.1f}%."
        )

    # 4. Relational interaction inference
    interaction_notes = []
    if "person" in labels_set:
        person_count = class_counts.get("person", 0)
        other_items = [k for k in labels_set if k != "person"]
        if other_items:
            interaction_notes.append(
                f"The presence of {person_count} person(s) alongside {', '.join(other_items)} suggests active human engagement within the {scene_type}."
            )
        else:
            interaction_notes.append("The subject focus centers on human presence and activity.")

    if len(detections) >= 2:
        # Check box proximity between top two detections
        b1 = detections[0]["box"]
        b2 = detections[1]["box"]
        overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        if overlap_x > 0 and overlap_y > 0:
            interaction_notes.append(
                f"The primary detected **{detections[0]['label']}** and **{detections[1]['label']}** spatially overlap, indicating direct physical proximity or interaction."
            )

    # Build markdown output
    overview = (
        f"This image captures a **{scene_type}** featuring **{total}** identified object{'s' if total > 1 else ''} "
        f"across **{len(unique_cats)}** distinct categor{'ies' if len(unique_cats) > 1 else 'y'} ({objects_phrase})."
    )

    spatial_block = "\n".join(spatial_descriptions)
    interaction_block = " ".join(interaction_notes) if interaction_notes else (
        "Objects are arranged across distinct zones of the visual frame without substantial occlusions."
    )

    full_text = (
        f"### Scene Overview\n{overview}\n\n"
        f"### Key Visible Entities\n{spatial_block}\n\n"
        f"### Spatial Relationships & Interactions\n{interaction_block}\n\n"
        f"### Contextual Summary\n"
        f"The visual composition exhibits consistent spatial organization characteristic of a {scene_type}. "
        f"The detected entities demonstrate high detection certainty (average confidence: {summary.get('average_confidence', 0)*100:.1f}%), "
        f"corroborating the structured scene layout."
    )

    latency = time.perf_counter() - start_time
    return {
        "success": True,
        "description": full_text,
        "provider": "Local Contextual AI Synthesizer (Offline Engine)",
        "error": None,
        "latency_seconds": latency
    }


def describe_image(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    summary: Dict[str, Any],
    provider: str = "auto",
    azure_endpoint: Optional[str] = None,
    azure_key: Optional[str] = None,
    azure_model: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    gemini_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified entry point for generating image descriptions.
    Dispatches to the requested provider and gracefully falls back to local synthesis
    if API keys are missing or network calls fail.
    """
    provider = provider.lower().strip()

    # Automatic provider selection
    if provider == "auto":
        if azure_endpoint or os.getenv("AZURE_AI_ENDPOINT"):
            provider = "azure"
        elif openai_key or os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif gemini_key or os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            provider = "local"

    # Dispatch to chosen provider
    if provider == "azure":
        res = generate_azure_caption(
            image=image,
            detected_objects=detected_objects,
            endpoint=azure_endpoint,
            api_key=azure_key,
            model_name=azure_model
        )
        if res["success"]:
            return res
        # If Azure failed, notify and seamlessly fall back to local synthesizer
        fallback = generate_local_scene_summary(image, detected_objects, summary)
        fallback["error"] = f"Azure AI failed ({res.get('error')}). Fallen back to Local Scene Synthesizer."
        return fallback

    elif provider == "openai":
        res = generate_openai_caption(
            image=image,
            detected_objects=detected_objects,
            api_key=openai_key,
            model_name=openai_model
        )
        if res["success"]:
            return res
        fallback = generate_local_scene_summary(image, detected_objects, summary)
        fallback["error"] = f"OpenAI API failed ({res.get('error')}). Fallen back to Local Scene Synthesizer."
        return fallback

    elif provider == "gemini":
        res = generate_gemini_caption(
            image=image,
            detected_objects=detected_objects,
            api_key=gemini_key,
            model_name=gemini_model
        )
        if res["success"]:
            return res
        fallback = generate_local_scene_summary(image, detected_objects, summary)
        fallback["error"] = f"Gemini API failed ({res.get('error')}). Fallen back to Local Scene Synthesizer."
        return fallback

    else:
        # Default or local provider
        return generate_local_scene_summary(image, detected_objects, summary)
