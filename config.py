import json
import threading
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("llm_prompt_config.json")
DEFAULT_CONFIG = {
    "mistral_api_key": "",
    "gemini_api_key": "",
    "lmstudio_api_base": "http://127.0.0.1:1234/v1",
    "lmstudio_api_key": "",
    "image_max_size": 768,
    "image_max_kb": 400,
}
ALLOWED_KEYS = frozenset(DEFAULT_CONFIG)

_lock = threading.RLock()
_config = None


def _coerce_config(data):
    result = dict(DEFAULT_CONFIG)
    if isinstance(data, dict):
        for key in ALLOWED_KEYS:
            if key in data:
                result[key] = data[key]

    result["mistral_api_key"] = str(result["mistral_api_key"] or "").strip()
    result["gemini_api_key"] = str(result["gemini_api_key"] or "").strip()
    result["lmstudio_api_base"] = (
        str(result["lmstudio_api_base"] or DEFAULT_CONFIG["lmstudio_api_base"])
        .strip()
        .rstrip("/")
    )
    result["lmstudio_api_key"] = str(result["lmstudio_api_key"] or "").strip()
    result["image_max_size"] = max(64, int(result["image_max_size"]))
    result["image_max_kb"] = max(32, int(result["image_max_kb"]))
    return result


def load_config(force=False):
    global _config
    with _lock:
        if _config is not None and not force:
            return dict(_config)

        data = {}
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                data = {}

        _config = _coerce_config(data)
        return dict(_config)


def update_config(values):
    global _config
    with _lock:
        current = load_config()
        for key, value in values.items():
            if key in ALLOWED_KEYS:
                current[key] = value
        _config = _coerce_config(current)

        temporary_path = CONFIG_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(CONFIG_PATH)
        return dict(_config)
