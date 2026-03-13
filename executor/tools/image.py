"""
Image Generation Tools -- two-path image generation for the code executor.

WHAT: Two generation functions:
      - generate_companion_image(): RunComfy with the companion's LoRA model
        for images featuring the companion herself
      - generate_image(): Google Gemini/Imagen for everything else (scenery,
        objects, other people)
      Plus check_image_status() for polling async RunComfy jobs.

WHY:  Images of the companion require a fine-tuned LoRA model (RunComfy) to
      look like her. All other images use Gemini which is cheaper and faster.
      The companion's code chooses the right function based on whether she
      appears in the image.

HOW:  RunComfy: POST workflow to personal deployment -> poll for completion
      -> return image URL. Gemini: Call Imagen model -> return URL or base64.
      Both return {"status": "success/error", "image_url": "..."} dicts.
"""
import os
import time
import requests
import json
from typing import Dict, Optional

# RunComfy for image generation
RUNCOMFY_API_KEY = os.environ.get("RUNCOMFY_API_KEY")
RUNCOMFY_USER_ID = os.environ.get("RUNCOMFY_USER_ID", "")
RUNCOMFY_PERSONAL_DEPLOYMENT_ID = os.environ.get("RUNCOMFY_PERSONAL_DEPLOYMENT_ID")
RUNCOMFY_BASE_URL = "https://api.runcomfy.com/api"

# Google Gemini for general images
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")


def generate_companion_image(prompt: str, wait: bool = True, timeout: int = 120) -> Dict:
    """Generate an image of the companion using her LoRA model (RunComfy).

    Use this ONLY for images featuring the companion herself.

    Args:
        prompt: Description of the companion in the image (e.g., "companion reading at a coffee shop")
        wait: Whether to wait for completion (default True)
        timeout: Max seconds to wait (default 120)

    Returns:
        {"status": "success", "image_url": "https://..."}
        or {"status": "error", "message": "..."}

    Example:
        >>> from tools import image
        >>> result = image.generate_companion_image("companion smiling at sunset")
        >>> print(result['image_url'])
    """
    if not RUNCOMFY_API_KEY or not RUNCOMFY_PERSONAL_DEPLOYMENT_ID:
        return {"status": "error", "message": "RunComfy not configured"}

    # Build workflow input for personal deployment (uses the companion's LoRA)
    workflow_input = {
        "positive_prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted, deformed",
        "seed": int(time.time()) % 1000000,  # Random seed
        "width": 1024,
        "height": 1024,
    }

    try:
        # Submit workflow
        response = requests.post(
            f"{RUNCOMFY_BASE_URL}/workflows/run",
            headers={
                "Authorization": f"Bearer {RUNCOMFY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "deployment_id": RUNCOMFY_PERSONAL_DEPLOYMENT_ID,
                "input": workflow_input
            },
            timeout=30
        )

        if response.status_code != 200:
            return {"status": "error", "message": f"API error: {response.status_code}"}

        data = response.json()
        run_id = data.get("run_id")

        if not run_id:
            return {"status": "error", "message": "No run_id returned"}

        if not wait:
            return {"status": "pending", "run_id": run_id}

        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = requests.get(
                f"{RUNCOMFY_BASE_URL}/runs/{run_id}",
                headers={"Authorization": f"Bearer {RUNCOMFY_API_KEY}"},
                timeout=10
            )

            if status_response.status_code != 200:
                time.sleep(3)
                continue

            status_data = status_response.json()
            status = status_data.get("status")

            if status == "completed":
                outputs = status_data.get("outputs", {})
                # Look for image URL in outputs
                for key, value in outputs.items():
                    if isinstance(value, str) and value.startswith("http"):
                        return {"status": "success", "image_url": value}
                    elif isinstance(value, list) and value:
                        for item in value:
                            if isinstance(item, str) and item.startswith("http"):
                                return {"status": "success", "image_url": item}

                return {"status": "error", "message": "No image URL in output"}

            elif status == "failed":
                return {"status": "error", "message": status_data.get("error", "Unknown error")}

            time.sleep(5)  # Wait before polling again

        return {"status": "error", "message": f"Timeout after {timeout}s"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_image(prompt: str) -> Dict:
    """Generate an image NOT featuring the companion (uses Google Gemini/Nanobanana).

    Use this for general images, scenery, objects, other people, etc.

    Args:
        prompt: Description of the image to generate

    Returns:
        {"status": "success", "image_url": "https://..."} or base64 data
        or {"status": "error", "message": "..."}

    Example:
        >>> from tools import image
        >>> result = image.generate_image("a cozy reading nook with warm lighting")
        >>> print(result)
    """
    if not GOOGLE_GEMINI_API_KEY:
        return {"status": "error", "message": "Gemini API not configured"}

    try:
        import google.generativeai as genai

        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

        # Use Imagen model for image generation
        model = genai.ImageGenerationModel("imagen-3.0-generate-002")

        result = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_only_high",
            person_generation="allow_adult"
        )

        if result.images:
            # Get first image
            image_data = result.images[0]

            # If we have a URL, return it
            if hasattr(image_data, 'url') and image_data.url:
                return {"status": "success", "image_url": image_data.url}

            # Otherwise return base64 data
            if hasattr(image_data, '_pil_image'):
                import base64
                from io import BytesIO
                buffer = BytesIO()
                image_data._pil_image.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode()
                return {
                    "status": "success",
                    "image_data": f"data:image/png;base64,{b64[:50]}...",
                    "note": "Base64 image data (truncated in output)"
                }

            return {"status": "success", "message": "Image generated (check Gemini console)"}

        return {"status": "error", "message": "No image generated"}

    except ImportError:
        return {"status": "error", "message": "google-generativeai not installed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_image_status(run_id: str) -> Dict:
    """Check the status of a pending RunComfy image generation.

    Args:
        run_id: The run_id from a previous generate_companion_image call

    Returns:
        {"status": "pending|completed|failed", ...}
    """
    if not RUNCOMFY_API_KEY:
        return {"status": "error", "message": "RunComfy not configured"}

    try:
        response = requests.get(
            f"{RUNCOMFY_BASE_URL}/runs/{run_id}",
            headers={"Authorization": f"Bearer {RUNCOMFY_API_KEY}"},
            timeout=10
        )

        if response.status_code != 200:
            return {"status": "error", "message": f"API error: {response.status_code}"}

        data = response.json()
        status = data.get("status", "unknown")

        if status == "completed":
            outputs = data.get("outputs", {})
            for key, value in outputs.items():
                if isinstance(value, str) and value.startswith("http"):
                    return {"status": "completed", "image_url": value}
            return {"status": "completed", "outputs": outputs}

        return {"status": status}

    except Exception as e:
        return {"status": "error", "message": str(e)}
