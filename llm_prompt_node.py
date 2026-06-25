import base64
import hashlib
import io
import logging
import threading
from collections import OrderedDict
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

from .config import load_config, update_config


LOGGER = logging.getLogger(__name__)
MAX_IMAGES = 30
PROMPT_CACHE_LIMIT = 256
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL_CHOICES = [
    "mistral: pixtral-large-latest",
    "gemini: gemini-2.5-flash",
    "gemini: gemini-2.5-pro",
]
DEFAULT_MODEL_CHOICE = DEFAULT_MODEL_CHOICES[0]

_sessions = {
    "mistral": requests.Session(),
    "gemini": requests.Session(),
    "lmstudio": requests.Session(),
}
_model_choices = list(DEFAULT_MODEL_CHOICES)


def build_prompt_cache_key(
    unique_id,
    prompt,
    model,
    temperature,
    max_tokens,
    top_p,
    prompt_to_improve,
    auto_unload_lmstudio,
    images,
):
    digest = hashlib.sha256()
    values = (
        str(unique_id or ""),
        prompt,
        model,
        float(temperature),
        int(max_tokens),
        float(top_p),
        str(prompt_to_improve or ""),
        bool(auto_unload_lmstudio),
    )
    digest.update(repr(values).encode("utf-8"))

    for image in images:
        digest.update(image.mode.encode("ascii"))
        digest.update(repr(image.size).encode("ascii"))
        digest.update(image.tobytes())

    return digest.hexdigest()


def get_lmstudio_headers(config):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ComfyUI-LLM++/1.0",
    }
    if config["lmstudio_api_key"]:
        headers["Authorization"] = f"Bearer {config['lmstudio_api_key']}"
    return headers


def get_lmstudio_native_api_base(api_base):
    parsed = urlsplit(api_base)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/v1", "", ""))


def fetch_lmstudio_model_choices(timeout=5):
    config = load_config()
    response = _sessions["lmstudio"].get(
        f"{config['lmstudio_api_base']}/models",
        headers=get_lmstudio_headers(config),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    models = []
    for item in payload.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else None
        if model_id:
            models.append(f"lmstudio: {model_id}")
    return sorted(set(models))


def refresh_model_choices(timeout=5):
    global _model_choices
    lmstudio_models = fetch_lmstudio_model_choices(timeout=timeout)
    _model_choices = list(DEFAULT_MODEL_CHOICES) + lmstudio_models
    return list(_model_choices)


def normalize_model_choice(model_choice):
    value = (model_choice or DEFAULT_MODEL_CHOICE).strip()
    if ":" not in value:
        raise ValueError(f"Invalid model selection: {value}")
    provider, model = value.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"mistral", "gemini", "lmstudio"} or not model:
        raise ValueError(f"Invalid model selection: {value}")
    return provider, model


def is_every_run_mode(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "every run", "every_run"}


def tensor_batch_to_pil(images):
    if images is None:
        return []

    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    else:
        images = np.asarray(images)

    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {images.shape}")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images supported.")

    result = []
    for image in images:
        image = np.nan_to_num(image)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        if image.shape[-1] == 1:
            pil_image = Image.fromarray(image[:, :, 0], mode="L").convert("RGB")
        elif image.shape[-1] == 3:
            pil_image = Image.fromarray(image, mode="RGB")
        elif image.shape[-1] == 4:
            pil_image = Image.fromarray(image, mode="RGBA").convert("RGB")
        else:
            raise ValueError(f"Unsupported image channel count: {image.shape[-1]}")
        result.append(pil_image)
    return result


def encode_image_for_request(image, config):
    max_size = config["image_max_size"]
    max_kb = config["image_max_kb"]
    image = image.convert("RGB")

    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        size = (
            max(1, round(image.width * ratio)),
            max(1, round(image.height * ratio)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)

    quality = 90
    buffer = io.BytesIO()
    while True:
        buffer.seek(0)
        buffer.truncate()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        if buffer.tell() <= max_kb * 1024 or quality <= 40:
            break
        quality -= 5
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def send_to_mistral(model, prompt, images, temperature, max_tokens, top_p, config):
    if not config["mistral_api_key"]:
        raise ValueError("Set the Mistral API key in Settings > LLM++.")

    content = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = encode_image_for_request(image, config)
        content.append(
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"}
        )

    response = _sessions["mistral"].post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {config['mistral_api_key']}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-LLM++/1.0",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def send_to_gemini(model, prompt, images, temperature, max_tokens, top_p, config):
    if not config["gemini_api_key"]:
        raise ValueError("Set the Gemini API key in Settings > LLM++.")

    parts = []
    for image in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encode_image_for_request(image, config),
                }
            }
        )
    parts.append({"text": prompt})

    response = _sessions["gemini"].post(
        f"{GEMINI_API_BASE}/{quote(model, safe='')}:generateContent",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-LLM++/1.0",
            "x-goog-api-key": config["gemini_api_key"],
        },
        json={
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
                "topP": float(top_p),
            },
        },
        timeout=120,
    )
    if not response.ok:
        raise ValueError(f"Gemini API error {response.status_code}: {response.text}")

    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        raise ValueError(
            f"Gemini returned no candidates{f': {reason}' if reason else '.'}"
        )
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text.strip()


def send_to_lmstudio(model, prompt, images, temperature, max_tokens, top_p, config):
    content = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = encode_image_for_request(image, config)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )

    response = _sessions["lmstudio"].post(
        f"{config['lmstudio_api_base']}/chat/completions",
        headers=get_lmstudio_headers(config),
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        },
        timeout=120,
    )
    if not response.ok:
        raise ValueError(
            f"LM Studio API error {response.status_code}: {response.text}"
        )

    choices = response.json().get("choices") or []
    if not choices:
        raise ValueError("LM Studio returned no choices.")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    text = str(content or "").strip()
    if not text:
        raise ValueError("LM Studio returned an empty response.")
    return text


def unload_lmstudio_model(model, config):
    response = _sessions["lmstudio"].post(
        f"{get_lmstudio_native_api_base(config['lmstudio_api_base'])}/models/unload",
        headers=get_lmstudio_headers(config),
        json={"instance_id": model},
        timeout=10,
    )
    if not response.ok:
        raise ValueError(
            f"LM Studio unload error {response.status_code}: {response.text}"
        )


def build_model_prompt(instruction, prompt_to_improve):
    source_prompt = str(prompt_to_improve or "").strip()
    if not source_prompt:
        return instruction
    return (
        f"{instruction}\n\n"
        "Prompt to improve:\n"
        f"{source_prompt}"
    )


class LLMPlusPrompt:
    _prompt_cache = OrderedDict()
    _prompt_cache_lock = threading.RLock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe the image",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "model": (list(_model_choices), {"default": DEFAULT_MODEL_CHOICE}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.1},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 32768, "step": 1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "Prompt to Improve": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Paste the prompt you want the selected model to improve",
                    },
                ),
                "auto_unload_lmstudio": ("BOOLEAN", {"default": False}),
                "Every run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "LLM++"
    DESCRIPTION = (
        "Generates an image prompt with Mistral, Gemini, or an LM Studio model. "
        "Configure API keys and image limits in ComfyUI Settings > LLM++."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        every_run = kwargs.get("Every run", kwargs.get("every_run", True))
        generation_mode = kwargs.get("generation_mode")
        mode_value = generation_mode if generation_mode is not None else every_run
        if is_every_run_mode(mode_value):
            return float("nan")
        return False

    def generate(
        self,
        prompt,
        model,
        temperature,
        max_tokens,
        top_p,
        prompt_to_improve=None,
        auto_unload_lmstudio=False,
        image=None,
        unique_id=None,
        generation_mode=None,
        **kwargs,
    ):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        config = load_config()
        images = tensor_batch_to_pil(image)
        provider, model_name = normalize_model_choice(model)
        prompt_to_improve = kwargs.get(
            "Prompt to Improve",
            kwargs.get("prompt_to_improve", kwargs.get("append_text", prompt_to_improve)),
        )
        model_prompt = build_model_prompt(prompt, prompt_to_improve)
        cache_key = build_prompt_cache_key(
            unique_id,
            prompt,
            model,
            temperature,
            max_tokens,
            top_p,
            prompt_to_improve,
            auto_unload_lmstudio,
            images,
        )
        cache_slot = str(unique_id or cache_key)
        every_run = kwargs.get("Every run", kwargs.get("every_run", True))
        mode_value = generation_mode if generation_mode is not None else every_run

        if not is_every_run_mode(mode_value):
            with self._prompt_cache_lock:
                cached_entry = self._prompt_cache.get(cache_slot)
                if cached_entry is not None and cached_entry[0] == cache_key:
                    self._prompt_cache.move_to_end(cache_slot)
                    return (cached_entry[1],)

        if provider == "gemini":
            result = send_to_gemini(
                model_name, model_prompt, images, temperature, max_tokens, top_p, config
            )
        elif provider == "lmstudio":
            result = send_to_lmstudio(
                model_name, model_prompt, images, temperature, max_tokens, top_p, config
            )
        else:
            result = send_to_mistral(
                model_name, model_prompt, images, temperature, max_tokens, top_p, config
            )

        if provider == "lmstudio" and auto_unload_lmstudio:
            try:
                unload_lmstudio_model(model_name, config)
            except Exception as error:
                LOGGER.warning("LLM++ could not unload LM Studio model: %s", error)

        with self._prompt_cache_lock:
            self._prompt_cache[cache_slot] = (cache_key, result)
            self._prompt_cache.move_to_end(cache_slot)
            while len(self._prompt_cache) > PROMPT_CACHE_LIMIT:
                self._prompt_cache.popitem(last=False)

        return (result,)


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    routes = PromptServer.instance.routes

    @routes.post("/llm-plus/settings")
    async def save_settings(request):
        try:
            payload = await request.json()
            update_config(payload)
            return web.json_response({"ok": True})
        except Exception as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400)

    @routes.get("/llm-plus/models")
    async def get_models(_request):
        try:
            models = refresh_model_choices()
            return web.json_response({"ok": True, "models": models})
        except Exception as error:
            return web.json_response(
                {
                    "ok": False,
                    "models": list(DEFAULT_MODEL_CHOICES),
                    "error": str(error),
                },
                status=502,
            )


register_routes()

NODE_CLASS_MAPPINGS = {
    "LLMPlusPrompt": LLMPlusPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPlusPrompt": "LLM++ Prompt",
}
