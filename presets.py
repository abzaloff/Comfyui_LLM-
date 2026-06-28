import re
import threading
from pathlib import Path


PRESETS_DIR = Path(__file__).with_name("presets")
DEFAULT_PRESETS = {
    "Flux - Describe": "Describe the image",
    "SDXL - Tokens": "Describe the image using only comma-separated tokens",
    "LM Studio LLM": (
        "Rewrite the user's input into one clear, detailed English image prompt. "
        "Accept any input format, including comma-separated tokens, short notes, "
        "fragments, or natural language. Expand the idea into a coherent descriptive "
        "prompt with richer visual detail, while preserving the original subject, "
        "mood, style, composition, and important attributes. Do not add unrelated "
        "elements. Output only one final prompt in English. Do not include "
        "explanations, labels, introductions, alternatives, bullet points, quotation "
        "marks, or any extra text."
    ),
}

_INVALID_NAME_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_lock = threading.RLock()


def _validate_name(name):
    name = str(name or "").strip()
    if not name:
        raise ValueError("Preset name is empty.")
    if len(name) > 120:
        raise ValueError("Preset name is too long.")
    if name in {".", ".."} or _INVALID_NAME_PATTERN.search(name):
        raise ValueError("Preset name contains characters that cannot be used.")
    return name


def _preset_path(name):
    safe_name = _validate_name(name)
    path = (PRESETS_DIR / f"{safe_name}.txt").resolve()
    root = PRESETS_DIR.resolve()
    if root not in path.parents:
        raise ValueError("Preset path is outside the presets directory.")
    return path


def ensure_default_presets():
    if PRESETS_DIR.exists():
        return
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in DEFAULT_PRESETS.items():
        _write_preset_file(_preset_path(name), text)


def _write_preset_file(path, text):
    temporary_path = path.with_suffix(".txt.tmp")
    temporary_path.write_text(str(text or ""), encoding="utf-8")
    temporary_path.replace(path)


def list_presets():
    with _lock:
        ensure_default_presets()
        items = []
        for path in sorted(PRESETS_DIR.glob("*.txt"), key=lambda item: item.stem.lower()):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            items.append({"name": path.stem, "size": size})
        return items


def get_preset(name):
    with _lock:
        ensure_default_presets()
        path = _preset_path(name)
        if not path.exists():
            raise KeyError(f"Preset '{name}' was not found.")
        return {"name": path.stem, "text": path.read_text(encoding="utf-8")}


def save_preset(name, text, original_name=None):
    with _lock:
        ensure_default_presets()
        name = _validate_name(name)
        original_name = str(original_name or "").strip()
        path = _preset_path(name)

        if original_name and original_name != name:
            original_path = _preset_path(original_name)
            if path.exists():
                raise ValueError(f"Preset '{name}' already exists.")
            _write_preset_file(path, text)
            if original_path.exists():
                original_path.unlink()
            action = "renamed"
        else:
            action = "updated" if path.exists() else "created"
            _write_preset_file(path, text)

        return {"name": name, "action": action, "presets": list_presets()}


def delete_preset(name):
    with _lock:
        ensure_default_presets()
        path = _preset_path(name)
        if not path.exists():
            raise KeyError(f"Preset '{name}' was not found.")
        path.unlink()
        return {"name": name, "presets": list_presets()}
