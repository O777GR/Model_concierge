"""Model Concierge v3.27: ВСЕ КАРТИНКИ И ВИДЕО ИДУТ В output\ (отдельная папка
D:\\AI_Servers\\sd-cpp\\output\\) — корень sd-cpp больше не замусоривается.
img2vid и upscale корректно находят свои входы в output\\. Плюс всё из v3.26:
ручная память Civitai через браузер, анти-залипание LLM, помощник промптов
(🎲/✨/📝), тихий сервер, валидация wildcards, XMods-YAML, embeddings, Krea2,
CFG турбо."""

from __future__ import annotations

import base64
import json
import random
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ==================== НАСТРОЙКИ ====================
HOST, PORT = "127.0.0.1", 8090
MODELS = Path(r"D:\AI_Servers\sd-cpp\models")
SD_CCPP_ROOT = MODELS.parent
OUTPUT_DIR = SD_CCPP_ROOT / "output"
SD_CLI = str(SD_CCPP_ROOT / "sdc.bat") if (SD_CCPP_ROOT / "sdc.bat").exists() else str(SD_CCPP_ROOT / "sd-cli.exe")
SD_CLI_RAW = str(SD_CCPP_ROOT / "sd-cli.exe")
LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
VL_API = "http://127.0.0.1:8081/v1/chat/completions"
LORA_DIR = str(MODELS / "lora")
WILDCARD_DIR = MODELS / "wildcards"
LOG_DIR = SD_CCPP_ROOT / "queue_logs"
QUEUE_FILE = SD_CCPP_ROOT / "queue.json"
DEFAULT_PROMPT = "a photo of a red-haired girl sitting on the snow in a pine forest"

GENRES = {
    "portrait": {
        "ru": "Портрет",
        "base": "a portrait of a young woman, expressive face, detailed skin texture",
        "mods": ["XMods/ProPhoto/Lighting", "XMods/ProPhoto/Artists",
                 "XMods/ProPhoto/Aperture", "XMods/Helpers/begin"],
    },
    "landscape": {
        "ru": "Пейзаж",
        "base": "a breathtaking landscape, epic natural scenery, atmospheric perspective",
        "mods": ["XMods/Helpers/begin", "XMods/ProPhoto/Lighting",
                 "XMods/ProPhoto/PhotoTerms"],
    },
    "scifi": {
        "ru": "Sci-Fi",
        "base": "a sci-fi scene, futuristic technology, cinematic atmosphere",
        "mods": ["XMods/SciFi/Lighting", "XMods/SciFi/Style",
                 "XMods/SciFi/Genre", "XMods/SciFi/CyberScape",
                 "XMods/Helpers/begin"],
    },
    "fantasy": {
        "ru": "Фэнтези",
        "base": "a fantasy scene, magical atmosphere, mythical creatures, epic lighting",
        "mods": ["XMods/Fantasy/Lighting", "XMods/Fantasy/Effects",
                 "XMods/Fantasy/Styles", "XMods/Helpers/begin"],
    },
    "horror": {
        "ru": "Хоррор",
        "base": "a horror scene, eerie atmosphere, unsettling mood",
        "mods": ["XMods/Halloween/Lighting", "XMods/Halloween/Mood",
                 "XMods/Halloween/Directors", "XMods/Halloween/Visuals",
                 "XMods/Helpers/begin"],
    },
    "cyberpunk": {
        "ru": "Киберпанк",
        "base": "a cyberpunk scene, neon lights, rain-slicked streets, dystopian city",
        "mods": ["XMods/SciFi/Lighting", "XMods/SciFi/CyberScape",
                 "XMods/SciFi/CyberLights", "XMods/Helpers/begin"],
    },
}

EXTRAS = {
    "sd15": "sharp focus, highly detailed, natural skin texture",
    "sdxl": "sharp focus, highly detailed, natural skin texture",
    "qwen": "sharp focus, highly detailed, natural skin texture",
    "zimage": "sharp focus, highly detailed, natural skin texture",
}
NEG = {
    "sd15": "blurry, low quality, deformed, painting, illustration, anime, cartoon, 3d render, plastic skin",
    "sdxl": "blurry, low quality, deformed, painting, illustration, anime, cartoon, 3d render, plastic skin",
    "qwen": "blurry, low quality, deformed",
    "zimage": "blurry, low quality, deformed",
}
TRIGGERS = [
    ("detailed perfection", "ultra detailed, detailed skin pore, sharp, realistic style"),
    ("famegrid", "IGMODEL, rlskn"),
    ("zit-real", "photorealistic, sharp focus, detailed skin"),
]
STOP_TAGS = {"1girl", "solo", "looking at viewer", "looking_at_viewer", "highres",
             "absurdres", "realistic", "photo", "upper body", "full body"}

SOURCES = {
    "dit_qwen_image": "https://huggingface.co/QuantStack/Qwen-Image-GGUF/resolve/main/Qwen_Image-Q3_K_M.gguf",
    "vae_qwen": "https://huggingface.co/QuantStack/Qwen-Image-GGUF/resolve/main/VAE/Qwen_Image_VAE.safetensors",
    "llm_qwenvl": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/qwen2.5-vl-7b-instruct-q4_k_m.gguf",
    "dit_z_image": "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
    "llm_zimage": "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/qwen3_4b-Q8_0.gguf",
    "vae_zimage": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    "full_sd15": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "full_sdxl": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "esrgan": "https://huggingface.co/Kim2091/Upscaler/resolve/main/RealESRGAN_x4plus.pth",
}
ROLE_RU = {
    "full_sd15": "Полный чекпоинт SD 1.5 (solo)",
    "full_sdxl": "Полный чекпоинт SDXL (solo)",
    "lora_sd": "LoRA под SD 1.5/SDXL",
    "lora_qwen_image": "LoRA под Qwen-Image",
    "lora_z_image": "LoRA под Z-Image (ZIT)",
    "lora_flux": "LoRA под Flux",
    "dit_qwen_image": "DiT (тело) Qwen-Image",
    "dit_z_image": "DiT (тело) Z-Image-Turbo / Krea2",
    "dit_flux": "DiT Flux",
    "llm_qwenvl": "LLM-энкодер Qwen2.5-VL",
    "llm_zimage": "Текстовый энкодер Qwen3-4B (для Z-Image)",
    "llm_vision": "Визуальный проектор (mmproj) — «глаза» VL-модели",
    "llm": "LLM-энкодер",
    "vae": "VAE",
    "vae_qwen": "VAE Qwen-Image",
    "vae_zimage": "VAE Flux-стиль (ae.safetensors) для Z-Image",
    "animatediff_mm": "Motion-модуль AnimateDiff (видео)",
    "esrgan": "ESRGAN-апскейлер x4 (опционально, для upscale)",
    "embedding": "Textual Inversion embedding (встраивание)",
    "unknown": "Не опознано",
}
SUBDIR = {
    "full_sd15": "checkpoints", "full_sdxl": "checkpoints",
    "dit_qwen_image": "dit", "dit_z_image": "dit", "dit_flux": "dit",
    "llm_qwenvl": "llm", "llm_zimage": "llm", "llm_vision": "llm", "llm": "llm",
    "vae": "vae", "vae_qwen": "vae", "vae_zimage": "vae",
    "lora_sd": "lora", "lora_qwen_image": "lora", "lora_z_image": "lora", "lora_flux": "lora",
    "animatediff_mm": "motion",
    "esrgan": "esrgan",
    "embedding": "embeddings",
    "unknown": "unknown",
}
NEEDS = {
    "qwen": ["dit_qwen_image", "llm_qwenvl", "vae_qwen"],
    "zimage": ["dit_z_image", "llm_zimage", "vae_zimage"],
    "sd15": ["full_sd15"],
    "sdxl": ["full_sdxl"],
}
SCAN_EXT = (".safetensors", ".gguf", ".pth", ".pt")
DOWNLOADS: dict[str, dict] = {}

# ==================== ОЧЕРЕДЬ ====================
QUEUE: list[dict] = []
QUEUE_LOCK = threading.Lock()
CURRENT_PROC: subprocess.Popen | None = None
JOB_COUNTER = 0
LAST_ROLES: dict = {}
LAST_FILES: list[dict] = []
LORA_TRIGGERS: dict[str, list[str]] = {}
CIV_CACHE_FILE = SD_CCPP_ROOT / "civitai_cache.json"
CIV_CACHE: dict[str, dict] = {}
CIV_TOKEN_FILE = SD_CCPP_ROOT / "civitai_token.txt"
CIV_TOKEN = CIV_TOKEN_FILE.read_text(encoding="utf-8").strip() if CIV_TOKEN_FILE.exists() else ""
WC_YAML: dict[str, list[tuple[float, str]]] = {}
WC_ERRORS: list[str] = []
# =================================================================


def _out(name: str) -> str:
    """Путь в output\\ (для -o и -i аргументов sd-cli)."""
    return str(OUTPUT_DIR / name)


def read_keys(path: Path) -> tuple[list[str], bool]:
    """Читает имена тензоров из safetensors; для gguf возвращает ([], True)."""
    with open(path, "rb") as f:
        if f.read(4) == b"GGUF":
            return [], True
        f.seek(0)
        try:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 1_000_000_000:
                return [], False
            header = json.loads(f.read(n).decode("utf-8", "ignore"))
        except Exception:
            return [], False
    return [k for k in header if k != "__metadata__"], False


def read_meta(path: Path) -> dict:
    """Читает __metadata__ из заголовка safetensors (инфа об обучении, триггеры)."""
    with open(path, "rb") as f:
        if f.read(4) == b"GGUF":
            return {}
        f.seek(0)
        try:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 1_000_000_000:
                return {}
            header = json.loads(f.read(n).decode("utf-8", "ignore"))
        except Exception:
            return {}
    m = header.get("__metadata__", {})
    return m if isinstance(m, dict) else {}


def extract_triggers(path: Path) -> list[str]:
    """Достаёт триггеры лоры из метаданных: activation word, комментарий, топ-теги."""
    m = read_meta(path)
    if not m:
        return []
    out: list[str] = []
    for key in ("activation_word", "activation_words", "ss_activation_word",
                "trigger_words", "trigger words"):
        v = m.get(key)
        if v:
            out += [w.strip() for w in str(v).replace(",", " ").split() if w.strip()]
    comment = str(m.get("ss_training_comment", "")).strip()
    if comment and len(comment) < 200:
        out += [w.strip(",;") for w in comment.split() if w.strip(",;")]
    freq = m.get("ss_tag_frequency", "")
    if freq:
        try:
            data = json.loads(freq)
            counts: dict[str, int] = {}
            for bucket in data.values():
                if isinstance(bucket, dict):
                    for tag, c in bucket.items():
                        counts[tag] = counts.get(tag, 0) + int(c)
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
            out += [t for t, _ in top if t.lower() not in STOP_TAGS]
        except Exception:
            pass
    junk = {"none", "null", "true", "false", "n/a", ""}
    seen: set[str] = set()
    res: list[str] = []
    for w in out:
        if w.lower() not in seen and w.lower() not in junk:
            seen.add(w.lower())
            res.append(w)
    return res[:8]


def lora_family(path: Path, keys: list[str], name: str) -> str:
    """SD1.5 или SDXL: сигнатуры тензоров, метаданные, подсказки имени."""
    low = name.lower()
    s = " | ".join(keys[:3000])
    if "label_emb" in s:
        return "sdxl"
    if "input_blocks.10" in s or "input_blocks.11" in s:
        return "sd15"
    ver = str(read_meta(path).get("ss_base_model_version", "")).lower()
    if "sdxl" in ver:
        return "sdxl"
    if any(h in low for h in ("sdxl", "_xl", "xl_", "pony", "illustrious", "noobai", "animagine")):
        return "sdxl"
    if any(h in low for h in ("sd15", "sd1.5", "v1.5")):
        return "sd15"
    return ""


# ==================== WILDCARDS ====================
def wc_map() -> dict[str, Path]:
    if not WILDCARD_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(WILDCARD_DIR.rglob("*.txt"))}


def _wc_parse_block(raw: str) -> list[tuple[float, str]]:
    raw = raw.strip()
    if raw.startswith("{"):
        raw = raw[1:]
    if raw.endswith("}"):
        raw = raw[:-1]
    raw = raw.strip()
    if re.match(r"^\d+\$\$", raw):
        raw = re.sub(r"^\d+\$\$", "", raw).strip()
        return [(1.0, p.strip()) for p in raw.split("|") if p.strip()]
    if re.search(r"[\d.]+::", raw):
        opts: list[tuple[float, str]] = []
        for part in re.split(r"\s*\|\s*(?=[\d.]+::)", raw):
            m = re.match(r"^([\d.]+)::(.*)$", part.strip(), flags=re.S)
            if m:
                w = float(m.group(1))
                t = m.group(2).strip()
                if w > 0 and t:
                    opts.append((w, t))
        return opts
    return [(1.0, raw)] if raw else []


def wc_load_yaml() -> None:
    WC_YAML.clear()
    if not WILDCARD_DIR.exists():
        return
    for p in sorted(WILDCARD_DIR.rglob("*.yaml")):
        lines = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        root = group = leaf = ""
        i = 0
        while i < len(lines):
            s = lines[i].rstrip().strip()
            i += 1
            if not s or s.startswith("#") or s == "---":
                continue
            mk = re.match(r"^([A-Za-z0-9_\-]+):\s*$", s)
            if mk:
                key = mk.group(1)
                j = i
                while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#")):
                    j += 1
                nxt = lines[j].strip() if j < len(lines) else ""
                if nxt.startswith("- "):
                    leaf = key
                elif not root:
                    root = key
                else:
                    group, leaf = key, ""
                continue
            if s.startswith('- "'):
                buf = [s[2:]]
                while not buf[-1].rstrip().endswith('"') and i < len(lines):
                    buf.append(lines[i].rstrip())
                    i += 1
                raw = "\n".join(buf)
                if raw.rstrip().endswith('"'):
                    raw = raw.rstrip()[:-1]
                opts = _wc_parse_block(raw)
                if opts and (root or group or leaf):
                    WC_YAML["/".join(x for x in (root, group, leaf) if x)] = opts


def wc_pick(name: str) -> str | None:
    opts = WC_YAML.get(name)
    if opts is None:
        low = name.lower()
        cand = [k for k in WC_YAML if k.split("/")[-1].lower() == low]
        if len(cand) == 1:
            opts = WC_YAML[cand[0]]
    if not opts:
        return None
    texts = [t for _, t in opts]
    weights = [w for w, _ in opts]
    return random.choices(texts, weights=weights)[0]


def wc_validate() -> None:
    WC_ERRORS.clear()
    wm = wc_map()
    all_names = set(wm) | set(WC_YAML)
    paths = []
    if WILDCARD_DIR.exists():
        paths = sorted(list(WILDCARD_DIR.rglob("*.txt")) + list(WILDCARD_DIR.rglob("*.yaml")))
    for p in paths:
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
        depth = 0
        for i, c in enumerate(text):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth < 0:
                WC_ERRORS.append(f"{p.name}: лишняя '}}' в позиции {i}")
                break
        if depth > 0:
            WC_ERRORS.append(f"{p.name}: не закрыто {depth} '{{'")
        refs = re.findall(r"__([\w\-/ ]+?)__", text)
        for ref in refs:
            if ref not in all_names:
                WC_ERRORS.append(f"{p.name}: ссылка __{ref}__ не найдена")


def expand_wildcards(text: str, depth: int = 0) -> tuple[str, list[str]]:
    used: list[str] = []
    if depth > 3 or "__" not in text:
        return text, used
    wm = wc_map()

    def repl(m: re.Match) -> str:
        name = m.group(1)
        pick = wc_pick(name)
        if pick is None:
            p = wm.get(name)
            if p is None:
                return m.group(0)
            lines = [l.strip() for l in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            if not lines:
                return m.group(0)
            pick = random.choice(lines)
        pick = re.sub(r"\s+", " ", pick).replace("BREAK", " ").strip()
        used.append(f"wildcard {name} -> {pick[:80]}")
        return pick

    text = re.sub(r"__([\w\-/ ]+?)__", repl, text)
    text = re.sub(r"\{([^{}\n]+)\}", lambda mm: random.choice(mm.group(1).split("|")).strip(), text)
    if "__" in text and depth < 3:
        text2, used2 = expand_wildcards(text, depth + 1)
        return text2, used + used2
    return text, used


# ==================== ПОМОЩНИК ПРОМПТОВ ====================
def _clean_ai_prompt(text: str) -> str:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    seen: dict[str, int] = {}
    out: list[str] = []
    for p in parts:
        key = p.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 2:
            out.append(p)
        if len(out) >= 45:
            break
    return ", ".join(out)


def craft_prompt(base: str, mode: str, genre: str = "portrait") -> dict:
    if mode == "template":
        g = GENRES.get(genre, GENRES["portrait"])
        return {"prompt": g["base"], "source": f"шаблон: {g['ru']}"}

    if mode == "random":
        g = GENRES.get(genre, GENRES["portrait"])
        parts: list[str] = []
        for mod in g["mods"]:
            pick = wc_pick(mod)
            if pick:
                parts.append(pick)
        parts.append(g["base"])
        prompt = ", ".join(parts)
        prompt, _ = expand_wildcards(prompt)
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return {"prompt": prompt, "source": f"случайный шедевр: {g['ru']}"}

    if mode == "ai":
        sys_msg = ("Ты — эксперт по промптам для Stable Diffusion. "
                   "Превращай короткие описания пользователя в подробные SD-промпты. "
                   "Добавляй детали про освещение, стиль, композицию, камеру, атмосферу. "
                   "Пиши только теги через запятую на английском, без комментариев. "
                   "Используй веса (tag:1.3) для важных элементов. "
                   "Не используй BREAK, не пиши объяснений. "
                   "Не повторяй один тег дважды. "
                   "Ограничься ~40 словами.")
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": base.strip() or DEFAULT_PROMPT},
            ],
            "temperature": 0.6, "max_tokens": 400,
            "repeat_penalty": 1.25,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }).encode()
        req = urllib.request.Request(LLAMA_API, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"]["content"].strip()
            text = re.sub(r"\s+", " ", text).strip()
            return {"prompt": _clean_ai_prompt(text), "source": "улучшено локальным LLM"}
        except Exception as e:
            return {"prompt": base, "source": f"LLM недоступен: {e}"}

    return {"prompt": base, "source": "как есть"}
# =================================================================


def civ_load_cache() -> None:
    global CIV_CACHE
    if CIV_CACHE_FILE.exists():
        try:
            data = json.loads(CIV_CACHE_FILE.read_text(encoding="utf-8"))
            CIV_CACHE = {k: v for k, v in data.items() if "." in k}
        except Exception:
            CIV_CACHE = {}


def civ_save_cache() -> None:
    CIV_CACHE_FILE.write_text(json.dumps(CIV_CACHE, ensure_ascii=False, indent=1), encoding="utf-8")


CIV_HOSTS = ["https://civitai.com", "https://civitai.red"]


def _civ_api(query: str) -> dict:
    last_err: Exception | None = None
    for host in CIV_HOSTS:
        req = urllib.request.Request(host + "/api/v1/" + query)
        if CIV_TOKEN:
            req.add_header("Authorization", f"Bearer {CIV_TOKEN}")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
    raise last_err


def civ_query(name: str) -> list[str]:
    stem = Path(name).stem
    q = re.sub(r"[_\-\.]+", " ", stem)
    q = re.sub(r"\b(v\d+|fp\d+|bf16|f16|q\d_\w+|\d+)\b", " ", q, flags=re.I)
    q = " ".join(q.split())[:60]
    data = _civ_api("models?query=" + urllib.parse.quote(q) + "&types=LoRA&limit=5")
    for item in data.get("items", []):
        for ver in item.get("modelVersions", []):
            files = [f.get("name", "") for f in ver.get("files", [])]
            if any(Path(f).stem == stem for f in files):
                return ver.get("trainedWords", []) or []
    for item in data.get("items", []):
        for ver in item.get("modelVersions", []):
            if ver.get("trainedWords"):
                return ver["trainedWords"]
    return []


def civ_enrich() -> dict:
    if not LAST_FILES:
        scan()
    loras = [f["name"] for f in LAST_FILES if f["role"].startswith("lora")]
    report: dict = {"total": len(loras), "api": "ok",
                    "skipped_meta": 0, "cached": 0, "queried": {}}
    try:
        _civ_api("models?limit=1")
    except Exception as e:
        report["api"] = f"ошибка: {e}"
    for fname in loras:
        if LORA_TRIGGERS.get(fname):
            report["skipped_meta"] += 1
            continue
        if fname in CIV_CACHE:
            report["cached"] += 1
            continue
        if not str(report["api"]).startswith("ok"):
            report["queried"][fname] = "пропущено: API недоступен"
            break
        try:
            words = civ_query(fname)
        except Exception as e:
            words = []
            report["queried"][fname] = f"ошибка: {e}"
        else:
            report["queried"][fname] = words
        CIV_CACHE.setdefault(fname, {})["words"] = words
        time.sleep(1)
    civ_save_cache()
    return report


def classify_by_tensors(s: str) -> str:
    if any(t in s for t in ("lora_up", "lora_down", "lora_A", "lora_B", "lokr")):
        if "transformer_blocks" in s and "double_blocks" not in s:
            return "lora_qwen_image"
        if "double_blocks" in s or "single_blocks" in s:
            return "lora_flux"
        return "lora_sd"
    if "motion_modules" in s:
        return "animatediff_mm"
    if "visual." in s and ("lm_head" in s or "embed_tokens" in s):
        return "llm_qwenvl"
    if "model.layers" in s and ("lm_head" in s or "embed_tokens" in s):
        return "llm"
    if "double_blocks" in s:
        return "dit_flux"
    if "transformer_blocks" in s and "img_in" in s:
        return "dit_qwen_image"
    if "conditioner.embedders" in s:
        return "full_sdxl"
    if "cond_stage_model" in s and "model.diffusion_model" in s:
        return "full_sd15"
    if "model.diffusion_model" in s:
        return "dit_flux"
    if "encoder.down" in s or "decoder.up" in s or "quant_conv" in s:
        return "vae"
    return "unknown"


def classify(keys: list[str], name: str, is_gguf: bool) -> str:
    low = name.lower()
    s = " | ".join(keys[:3000]) if keys else ""
    if any(t in s for t in ("lora_up", "lora_down", "lora_A", "lora_B", "lokr")):
        if low.startswith("zit") or "zit-" in low or "z_image" in low or "zimage" in low or "krea" in low:
            return "lora_z_image"
        if "transformer_blocks" in s and "double_blocks" not in s:
            return "lora_qwen_image"
        if "double_blocks" in s or "single_blocks" in s:
            return "lora_flux"
        return "lora_sd"
    if "z_image" in low or "z-image" in low or "zimage" in low:
        if "qwen" in low:
            return "llm_zimage"
        if "ae." in low or "vae" in low:
            return "vae_zimage"
        return "dit_z_image"
    if "qwen3" in low:
        return "llm_zimage"
    if "krea" in low:
        return "dit_z_image"
    if "anima" in low and ("txt" in low or "text" in low):
        return "llm"
    if keys:
        role = classify_by_tensors(s)
        if role == "vae" and "qwen" in low:
            return "vae_qwen"
        if role != "unknown":
            return role
    if "esrgan" in low or "realesrgan" in low:
        return "esrgan"
    if low.endswith(".pt"):
        return "embedding"
    if is_gguf:
        if "mmproj" in low:
            return "llm_vision"
        if "qwen" in low and "image" in low:
            return "dit_qwen_image"
        if "vl" in low or "instruct" in low:
            return "llm_qwenvl"
        if "vae" in low:
            return "vae_qwen"
        return "unknown"
    if "vae" in low:
        return "vae_qwen" if "qwen" in low else "vae"
    return "unknown"


def analyze_file(path: Path) -> dict:
    keys, is_gguf = read_keys(path)
    role = classify(keys, path.name, is_gguf)
    d = {"name": path.name, "rel": str(path.relative_to(MODELS)),
         "size_gb": round(path.stat().st_size / 1e9, 2),
         "role": role, "role_ru": ROLE_RU.get(role, role)}
    if role == "unknown" and keys:
        d["keys_head"] = ", ".join(keys[:3])
    return d


def expected_path(role: str) -> Path:
    if role not in SOURCES:
        return MODELS / SUBDIR.get(role, "unknown") / f"<файл роли {role}>"
    return MODELS / SUBDIR[role] / SOURCES[role].rstrip("/").split("/")[-1]


def build_one(pipeline: str, ckpt_rel: str, lora_names: list, mode: str, roles: dict,
              custom_prompt: str = "", auto: bool = True, emb_rels: list | None = None) -> dict:
    """Собирает ОДНУ команду; ВСЕ выходные и входные файлы — в OUTPUT_DIR."""
    emb_rels = emb_rels or []
    neg = f'-n "{NEG.get(pipeline, NEG["sd15"])}"'
    prompt = custom_prompt.strip() or DEFAULT_PROMPT
    added = []
    if auto:
        ex = EXTRAS.get(pipeline, EXTRAS["sd15"])
        prompt += ", " + ex
        added.append("качество: " + ex)
        for l in lora_names:
            low = l.lower()
            words = None
            src = ""
            for key, w in TRIGGERS:
                if key in low:
                    words, src = w, "словарь"
                    break
            if words is None and LORA_TRIGGERS.get(l):
                words, src = ", ".join(LORA_TRIGGERS[l]), "метаданные"
            if words is None and CIV_CACHE.get(l, {}).get("words"):
                words, src = ", ".join(CIV_CACHE[l]["words"]), "civitai"
            if words:
                prompt += ", " + words
                added.append(f"{Path(l).stem} [{src}]: {words}")
    prompt, wused = expand_wildcards(prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    added.extend(wused)
    la, pl = "", prompt
    if lora_names:
        la = f' --lora-model-dir "{LORA_DIR}"'
        for l in lora_names:
            pl += f" <lora:{Path(l).stem}:1.0>"
    ea = ""
    if emb_rels:
        ea = f' --embd-dir "{MODELS / "embeddings"}"'
        for r in emb_rels:
            pl += f" {Path(r).stem}"
            added.append(f"embedding: {Path(r).stem}")
    stem = Path(ckpt_rel).stem if ckpt_rel else "base"
    ltag = "_" + "+".join(Path(l).stem for l in lora_names) if lora_names else ""
    img_name = f"{stem}{ltag}_img.png"
    img_path = _out(img_name)

    if mode == "upscale":
        esr = str(MODELS / roles["esrgan"]["rel"]) if "esrgan" in roles else str(expected_path("esrgan"))
        up_name = f"{stem}{ltag}_img_4x.png"
        return {"name": f"UPSCALE · x4 {img_name}", "added": added,
                "cmd": f'{SD_CLI} -M upscale -i "{img_path}" --upscale-model "{esr}" -o "{_out(up_name)}"'}

    if pipeline == "qwen":
        dit = str(MODELS / ckpt_rel) if ckpt_rel else str(expected_path("dit_qwen_image"))
        llm = str(MODELS / roles["llm_qwenvl"]["rel"]) if "llm_qwenvl" in roles else str(expected_path("llm_qwenvl"))
        vae = str(MODELS / roles["vae_qwen"]["rel"]) if "vae_qwen" in roles else str(expected_path("vae_qwen"))
        return {"name": f"QWEN · Картинка · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} --diffusion-model "{dit}" --llm "{llm}" --vae "{vae}"{la} '
                       f'-p "{pl}" {neg} --steps 20 --cfg-scale 4.0 -W 512 -H 512 -o "{img_path}"'}

    if pipeline == "zimage":
        dit = str(MODELS / ckpt_rel) if ckpt_rel else str(expected_path("dit_z_image"))
        if "krea" in stem.lower():
            return {"name": f"KREA2 · Картинка · {stem}{ltag} · 8 шагов", "added": added,
                    "cmd": f'{SD_CLI} --diffusion-model "{dit}"{la} -p "{pl}" {neg} '
                           f'--steps 8 --cfg-scale 1.0 -W 1024 -H 1024 -o "{img_path}"'}
        llm = str(MODELS / roles["llm_zimage"]["rel"]) if "llm_zimage" in roles else str(expected_path("llm_zimage"))
        vae = str(MODELS / roles["vae_zimage"]["rel"]) if "vae_zimage" in roles else str(expected_path("vae_zimage"))
        return {"name": f"Z-IMAGE · Картинка · {stem}{ltag} · 8 шагов", "added": added,
                "cmd": f'{SD_CLI} --diffusion-model "{dit}" --llm "{llm}" --vae "{vae}"{la} '
                       f'-p "{pl}" {neg} --steps 8 --cfg-scale 1.0 -W 1024 -H 1024 -o "{img_path}"'}

    m = f'-m "{MODELS / ckpt_rel}"'
    if pipeline == "sdxl":
        if mode == "hires":
            hr_name = f"{stem}{ltag}_hires.png"
            return {"name": f"SDXL · hires · {stem}{ltag}", "added": added,
                    "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} --steps 20 -W 1024 -H 1024 '
                           f'--hires --hires-width 1920 --hires-height 1080 --hires-steps 10 -o "{_out(hr_name)}"'}
        return {"name": f"SDXL · Картинка · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} --steps 20 -W 1024 -H 1024 -o "{img_path}"'}

    if mode == "hires":
        hr_name = f"{stem}{ltag}_hires.png"
        return {"name": f"SD1.5 · hires · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} --steps 20 -W 512 -H 512 '
                       f'--hires --hires-width 1366 --hires-height 768 --hires-steps 10 -o "{_out(hr_name)}"'}
    if mode in ("txt2vid", "img2vid") and "animatediff_mm" in roles:
        mm = f'--motion-module "{MODELS / roles["animatediff_mm"]["rel"]}"'
        if mode == "txt2vid":
            tv_name = f"{stem}{ltag}_t2v.webm"
            return {"name": f"SD1.5 · ТЕКСТ→ВИДЕО · {stem}{ltag}", "added": added,
                    "cmd": f'{SD_CLI} {m}{ea} -M vid_gen {mm} --video-frames 32 --fps 8 -p "{pl}" {neg} '
                           f'--steps 12 -W 512 -H 512 -o "{_out(tv_name)}"'}
        iv_name = f"{stem}{ltag}_i2v.webm"
        return {"name": f"SD1.5 · КАРТИНКА→ВИДЕО · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{ea} -M vid_gen {mm} --video-frames 32 --fps 8 -i "{img_path}" --strength 0.5 '
                       f'-p "{pl}" {neg} --steps 12 -W 512 -H 512 -o "{_out(iv_name)}"'}
    return {"name": f"SD1.5 · Картинка · {stem}{ltag}", "added": added,
            "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} --steps 20 -W 512 -H 512 -o "{img_path}"'}


def build_all_cmds(files: list, roles: dict) -> list:
    out = []
    neg = f'-n "{NEG["sd15"]}"'
    prompt = DEFAULT_PROMPT

    def p(role: str) -> str:
        if role in roles:
            return str(MODELS / roles[role]["rel"])
        return str(expected_path(role))

    def lora_part(names: list) -> tuple[str, str]:
        if not names:
            return "", prompt
        args = f' --lora-model-dir "{LORA_DIR}"'
        pr = prompt
        for l in names:
            pr += f" <lora:{Path(l).stem}:1.0>"
        return args, pr

    qwen_loras = [f["name"] for f in files if f["role"] == "lora_qwen_image"]
    zimg_loras = [f["name"] for f in files if f["role"] == "lora_z_image"]

    if "full_sd15" in roles:
        m = f'-m "{p("full_sd15")}"'
        out.append({"name": "SD1.5 · Картинка · лёгкая · 512/15 · RAM ~3 ГБ",
                    "cmd": f'{SD_CLI} {m} -p "{prompt}" {neg} --steps 15 -W 512 -H 512 -o "{_out("out_light.png")}"'})
        if "animatediff_mm" in roles:
            mm = f'--motion-module "{p("animatediff_mm")}"'
            out.append({"name": "SD1.5 · КАРТИНКА→ВИДЕО · из out_light.png · RAM ~5 ГБ",
                        "cmd": f'{SD_CLI} {m} -M vid_gen {mm} --video-frames 32 --fps 8 -i "{_out("out_light.png")}" '
                               f'--strength 0.5 -p "snow falls, soft light, camera slowly zooms" {neg} '
                               f'--steps 12 -W 512 -H 512 -o "{_out("video_i2v.webm")}"'})

    if "full_sdxl" in roles:
        m = f'-m "{p("full_sdxl")}"'
        out.append({"name": "SDXL · Картинка · лёгкая · 1024/15 · RAM ~8 ГБ",
                    "cmd": f'{SD_CLI} {m} -p "{prompt}" {neg} --steps 15 -W 1024 -H 1024 -o "{_out("out_light.png")}"'})

    if "dit_qwen_image" in roles or qwen_loras:
        trio = (f'--diffusion-model "{p("dit_qwen_image")}" '
                f'--llm "{p("llm_qwenvl")}" --vae "{p("vae_qwen")}"')
        out.append({"name": "QWEN · Картинка · лёгкая · 512/15 · RAM ~15 ГБ",
                    "cmd": f'{SD_CLI} {trio} -p "{prompt}" {neg} --steps 15 --cfg-scale 4.0 -W 512 -H 512 -o "{_out("out_light.png")}"'})

    if "dit_z_image" in roles or zimg_loras:
        trio = (f'--diffusion-model "{p("dit_z_image")}" '
                f'--llm "{p("llm_zimage")}" --vae "{p("vae_zimage")}"')
        la, pl = lora_part(zimg_loras)
        out.append({"name": "Z-IMAGE · Картинка · 8 шагов · 1024 · RAM ~11.5 ГБ",
                    "cmd": f'{SD_CLI} {trio}{la} -p "{pl}" {neg} --steps 8 --cfg-scale 1.0 -W 1024 -H 1024 -o "{_out("zimage_out.png")}"'})

    esr = p("esrgan")
    out.append({"name": "UPSCALE · x4 лёгкой out_light.png · RAM ~1-2 ГБ",
                "cmd": f'{SD_CLI} -M upscale -i "{_out("out_light.png")}" --upscale-model "{esr}" -o "{_out("out_light_4x.png")}"'})
    return out


RAM_TIPS = """ПАМЯТКА ПО RAM:
  - SD 1.5 512 ~ 3 ГБ, SDXL 1024 ~ 8 ГБ, Qwen ~ 15 ГБ, Z-Image GGUF Q8 ~ 11.5 ГБ
  - Krea2 — всё-в-одном (~13 ГБ): LLM и VAE встроены, снаружи не подаются
  - Очередь гоняет задачи ПОСЛЕДОВАТЕЛЬНО — драк за RAM нет
  - Если впритык: закрой браузер, снижай разрешение, --vae-tiling
  - ВСЕ КАРТИНКИ И ВИДЕО СОХРАНЯЮТСЯ В: output\\"""


def read_metadata(image_path: str) -> str:
    try:
        r = subprocess.run([SD_CLI_RAW, "-M", "metadata", "--image", image_path],
                           capture_output=True, timeout=120)
        text = (r.stdout + r.stderr).decode("utf-8", "ignore")
    except Exception:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"(?i)^prompt\s*[:=]", s):
            return re.split(r"[:=]", s, 1)[1].strip().strip('"')
    return text.strip()[:2000]


def ask_vl(image_path: str) -> str:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image as a stable diffusion prompt: "
                                     "comma-separated English tags, no comments."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0.2, "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(VL_API, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# ==================== ОЧЕРЕДЬ: функции ====================
def load_queue() -> None:
    global JOB_COUNTER
    if not QUEUE_FILE.exists():
        return
    try:
        data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        for j in data:
            if j.get("status") == "running":
                j["status"] = "waiting"
            QUEUE.append(j)
        JOB_COUNTER = max((j["id"] for j in QUEUE), default=0)
    except Exception:
        pass


def save_queue() -> None:
    with QUEUE_LOCK:
        data = [{k: j[k] for k in ("id", "name", "cmd", "status")} for j in QUEUE]
    QUEUE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def add_job(name: str, cmd: str) -> int:
    global JOB_COUNTER
    with QUEUE_LOCK:
        JOB_COUNTER += 1
        QUEUE.append({"id": JOB_COUNTER, "name": name or f"job {JOB_COUNTER}",
                      "cmd": cmd, "status": "waiting"})
    save_queue()
    return JOB_COUNTER


def stop_current() -> str:
    global CURRENT_PROC
    if CURRENT_PROC is not None:
        CURRENT_PROC.terminate()
        return "останавливаю текущую"
    return "ничего не крутится"


def clear_done() -> str:
    with QUEUE_LOCK:
        QUEUE[:] = [j for j in QUEUE if j["status"] in ("running", "waiting")]
    save_queue()
    return "ok"


def remove_job(job_id: int) -> str:
    with QUEUE_LOCK:
        job = next((j for j in QUEUE if j["id"] == job_id), None)
        if job is None or job["status"] == "running":
            return "не найдена или уже бежит"
        QUEUE.remove(job)
    save_queue()
    return "убрана"


def read_tail(job_id: int, n: int = 12) -> str:
    p = LOG_DIR / f"job_{job_id:03d}.log"
    if not p.exists():
        return "(лога пока нет)"
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return "\n".join(text.splitlines()[-n:])


def worker() -> None:
    global CURRENT_PROC
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        job = None
        with QUEUE_LOCK:
            for j in QUEUE:
                if j["status"] == "waiting":
                    j["status"] = "running"
                    job = j
                    break
        if job is None:
            time.sleep(2)
            continue
        log_path = LOG_DIR / f"job_{job['id']:03d}.log"
        try:
            with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
                proc = subprocess.Popen(job["cmd"], shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                        cwd=str(SD_CCPP_ROOT))
                CURRENT_PROC = proc
                proc.wait()
                job["status"] = "done" if proc.returncode == 0 else f"error {proc.returncode}"
        except Exception as e:
            job["status"] = f"error: {e}"
        finally:
            CURRENT_PROC = None
            save_queue()


def scan() -> dict:
    global LAST_ROLES, LAST_FILES
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wc_load_yaml()
    wc_validate()
    files = [analyze_file(p) for p in sorted(MODELS.rglob("*"))
             if p.is_file() and p.suffix in SCAN_EXT]
    LAST_FILES = files
    if not files:
        return {"files": [], "target": "empty", "missing": [], "commands": [], "options": {},
                "command": f"# Папка {MODELS} пуста — положи модели и нажми ещё раз."}
    roles = {}
    for f in files:
        roles.setdefault(f["role"], f)
    LAST_ROLES = roles
    LORA_TRIGGERS.clear()
    triggers_opt: dict[str, list[str]] = {}
    compat_opt: dict[str, str] = {}
    for f in files:
        if f["role"].startswith("lora"):
            t = extract_triggers(MODELS / f["rel"])
            if not t and CIV_CACHE.get(f["name"], {}).get("words"):
                t = CIV_CACHE[f["name"]]["words"]
            if t:
                LORA_TRIGGERS[f["name"]] = t
                triggers_opt[f["name"]] = t
            if f["role"] == "lora_sd":
                keys, _ = read_keys(MODELS / f["rel"])
                fam = lora_family(MODELS / f["rel"], keys, f["name"])
                if fam:
                    compat_opt[f["name"]] = fam
    if "dit_z_image" in roles or any(f["role"] == "lora_z_image" for f in files):
        target = "zimage"
    elif any(r in roles for r in ("dit_qwen_image", "lora_qwen_image")):
        target = "qwen"
    elif "full_sdxl" in roles:
        target = "sdxl"
    else:
        target = "sd15"
    present, missing = {}, []
    for need in NEEDS[target]:
        if need in roles:
            present[need] = roles[need]["rel"]
        else:
            missing.append({"role": need, "role_ru": ROLE_RU[need],
                            "url": SOURCES.get(need, ""),
                            "put_to": str(expected_path(need))})
    if "esrgan" not in roles:
        missing.append({"role": "esrgan", "role_ru": ROLE_RU["esrgan"],
                        "url": SOURCES["esrgan"], "put_to": str(expected_path("esrgan"))})
    options = {
        "sd15": [f["rel"] for f in files if f["role"] == "full_sd15"],
        "sdxl": [f["rel"] for f in files if f["role"] == "full_sdxl"],
        "qwenDit": [f["rel"] for f in files if f["role"] == "dit_qwen_image"],
        "zimageDit": [f["rel"] for f in files if f["role"] == "dit_z_image"],
        "lorasSd": [f["name"] for f in files if f["role"] == "lora_sd"],
        "lorasQwen": [f["name"] for f in files if f["role"] == "lora_qwen_image"],
        "lorasZimg": [f["name"] for f in files if f["role"] == "lora_z_image"],
        "embeds": [f["rel"] for f in files if f["role"] == "embedding"],
        "wildcards": sorted(set(list(wc_map()) + list(WC_YAML))),
        "wcErrors": WC_ERRORS,
        "hasMM": "animatediff_mm" in roles,
        "hasEsr": "esrgan" in roles,
        "triggers": triggers_opt,
        "loraCompat": compat_opt,
        "outputDir": str(OUTPUT_DIR),
    }
    commands = build_all_cmds(files, roles)
    return {"files": files, "target": target, "present": present,
            "missing": missing, "commands": commands, "options": options,
            "command": commands[0]["cmd"] if commands else ""}


def organize() -> list:
    moved = []
    for p in sorted(MODELS.rglob("*")):
        if not p.is_file() or p.suffix not in SCAN_EXT:
            continue
        keys, is_gguf = read_keys(p)
        role = classify(keys, p.name, is_gguf)
        dest_dir = MODELS / SUBDIR.get(role, "unknown")
        dest = dest_dir / p.name
        if p.parent != dest_dir and not dest.exists():
            dest_dir.mkdir(exist_ok=True)
            p.rename(dest)
            moved.append(f"{p.relative_to(MODELS)} -> {dest.relative_to(MODELS)}")
    return moved


def ask_llm(text: str) -> str:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "Ты — дворецкий моделей. Объясни по-русски в 3-4 предложениях, что найдено и чего не хватает."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2, "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(LLAMA_API, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# ==================== ТИХИЙ СЕРВЕР ====================
class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if sys.exc_info()[0] in (ConnectionAbortedError, ConnectionResetError,
                                 BrokenPipeError, TimeoutError):
            return
        super().handle_error(request, client_address)


HTML = """<!doctype html><html><head><meta charset='utf-8'><title>Model Concierge</title>
<style>body{background:#1e1e1e;color:#ddd;font-family:Consolas,monospace;margin:20px}
button{background:#333;color:#ddd;border:1px solid #555;padding:6px 12px;cursor:pointer;margin:2px}
input,select{background:#111;color:#ddd;border:1px solid #555;padding:6px;font-family:Consolas,monospace}
pre{background:#111;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}
label{margin-right:10px}
.miss{color:#f66}.ok{color:#6f6}.hdr{color:#6cf}.run{color:#fc6}</style></head><body>
<h2>Model Concierge v3.27</h2>
<button onclick='scan()'>Сканировать папку models</button>
<button onclick='organize()'>Раскидать по папкам</button>
<button onclick='civScan()'>🔎 Civitai триггеры</button>
<button onclick='openOutput()'>📁 Папка output</button>
<button onclick='explain()'>Объяснить через ИИ</button>
<button onclick='showQueue()'>Очередь</button>
<label><input type=checkbox id=auto onchange='toggleAuto()'> автообновление</label>
<button onclick='stopQ()'>Стоп текущей</button>
<button onclick='clearQ()'>Убрать завершённые</button>
<br><br>
<span class='hdr'>КОНСТРУКТОР КОМАНДЫ:</span>
<select id='pipe' onchange='fillCons()'></select>
<select id='ckpt'></select>
<select id='mode'></select>
<br>
<input id='pr' size='90' value='a photo of a red-haired girl sitting on the snow in a pine forest'>
<select id='genre'>
  <option value=portrait>Портрет</option>
  <option value=landscape>Пейзаж</option>
  <option value=scifi>Sci-Fi</option>
  <option value=fantasy>Фэнтези</option>
  <option value=horror>Хоррор</option>
  <option value=cyberpunk>Киберпанк</option>
</select>
<button onclick='craftPrompt("template")'>📝 Шаблон</button>
<button onclick='craftPrompt("random")'>🎲 Случайный шедевр</button>
<button onclick='craftPrompt("ai")'>✨ Улучшить через ИИ</button>
<br>
<div id='lorabox'></div>
<div id='embbox'></div>
<label><input type=checkbox id=autox checked> авто-добавки (качество + триггеры лор)</label>
<button onclick='buildCmd()'>Собрать команду</button>
<br><br>
<span class='hdr'>🔖 ТРИГГЕРЫ ВРУЧНУЮ (Civitai через браузер):</span>
<select id='civlora'></select>
<button onclick='openCiv()'>🔗 страница лоры</button>
<input id='civwords' size='70' placeholder='вставь сюда триггеры со страницы (через запятую)'>
<button onclick='saveCiv()'>💾 в память</button>
<br><br>
<input id='img' placeholder='путь к картинке (png/jpg)' size='70'>
<button onclick='guess()'>Узнать промпт</button>
<br><br>
<input id='qname' placeholder='имя задачи' size='24'>
<input id='qcmd' placeholder='команда целиком (своя)' size='60'>
<button onclick='addCustom()'>В очередь</button>
<pre id='out'>Жми «Сканировать»...</pre>
<script>
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function toggleAuto(){
  if(document.getElementById('auto').checked){
    window.autoTimer = setInterval(showQueue, 5000);
    showQueue();
  } else { clearInterval(window.autoTimer); }
}
function fillCivSelect(){
  const o = window.opts || {};
  const all = (o.lorasSd||[]).concat(o.lorasQwen||[]).concat(o.lorasZimg||[]);
  document.getElementById('civlora').innerHTML =
    all.map(l=>'<option>'+esc(l)+'</option>').join('') || '<option>(нет лор)</option>';
}
async function openCiv(){
  const n = document.getElementById('civlora').value;
  const q = encodeURIComponent(n.replace(/\\.safetensors$/,'').replace(/[_\\-\\.]+/g,' '));
  window.open('https://civitai.red/models?query='+q, '_blank');
}
async function saveCiv(){
  const n = document.getElementById('civlora').value;
  const w = document.getElementById('civwords').value;
  const d = await (await fetch('/api/civitai/manual',{method:'POST',
    body:JSON.stringify({name:n, words:w})})).json();
  document.getElementById('out').innerHTML =
    '<span class="ok">Сохранено:</span> ' + esc(n) + ' -> ' + esc((d.words||[]).join(', ')) +
    '\\nЖми «Сканировать», чтобы конструктор подхватил слова.';
}
async function openOutput(){
  const path = (window.opts||{}).outputDir || '';
  if(!path) return;
  // file:// ссылки на Windows-пути не открываются в большинстве браузеров;
  // копируем путь в буфер, чтобы пользователь вставил в Проводник
  const ta = document.createElement('textarea');
  ta.value = path;
  document.body.appendChild(ta); ta.select();
  try{ document.execCommand('copy'); }catch(e){}
  document.body.removeChild(ta);
  document.getElementById('out').innerHTML =
    '<span class="ok">Скопировано в буфер:</span> ' + esc(path) +
    '\\nОткрой Проводник (Win+E) и вставь (Ctrl+V) в адресную строку.';
}
function fillCons(){
  const p = document.getElementById('pipe').value;
  const o = window.opts || {};
  const cks = p=='sd15' ? (o.sd15||[]) :
              p=='sdxl' ? (o.sdxl||[]) :
              p=='qwen' ? (o.qwenDit||[]) : (o.zimageDit||[]);
  document.getElementById('ckpt').innerHTML =
    cks.map(c=>'<option>'+esc(c)+'</option>').join('') || '<option>(нет моделей)</option>';
  let modes = ['img'];
  if(p=='sd15' || p=='sdxl') modes = modes.concat(['hires']);
  if(o.hasMM && p=='sd15') modes = modes.concat(['txt2vid','img2vid']);
  if(o.hasEsr) modes = modes.concat(['upscale']);
  const ru = {img:'картинка',hires:'hires',txt2vid:'текст->видео',img2vid:'картинка->видео',upscale:'upscale x4'};
  document.getElementById('mode').innerHTML = modes.map(m=>'<option value='+m+'>'+ru[m]+'</option>').join('');
  const loras = p=='qwen' ? (o.lorasQwen||[]) : p=='zimage' ? (o.lorasZimg||[]) : (o.lorasSd||[]);
  const tr = o.triggers || {};
  const comp = o.loraCompat || {};
  document.getElementById('lorabox').innerHTML =
    loras.map(l => {
      const t = (tr[l]||[]).slice(0,4).join(', ');
      const fam = comp[l] || '';
      const bad = (p=='sd15' && fam=='sdxl') || (p=='sdxl' && fam=='sd15');
      return '<label style="'+(bad?'opacity:.45':'')+'">'+
        '<input type=checkbox class=lora value="'+esc(l)+'">'+esc(l)+
        (fam ? ' <span class="'+(bad?'miss':'ok')+'">'+fam.toUpperCase()+'</span>' : '')+
        (t ? ' <span class="ok">['+esc(t)+']</span>' : '') + '</label>';
    }).join('');
  const embs = (p=='sd15'||p=='sdxl') ? (o.embeds||[]) : [];
  document.getElementById('embbox').innerHTML =
    embs.map(e => '<label><input type=checkbox class=emb value="'+esc(e)+'">🧬 '+esc(e.split(/[\\\\/]/).pop())+'</label>').join('');
}
async function craftPrompt(mode){
  const base = document.getElementById('pr').value;
  const genre = document.getElementById('genre').value;
  document.getElementById('out').innerText = 'Готовлю промпт...';
  const d = await (await fetch('/api/craft',{method:'POST',
    body:JSON.stringify({base:base, mode:mode, genre:genre})})).json();
  document.getElementById('pr').value = d.prompt || '';
  document.getElementById('out').innerHTML =
    '<span class="hdr">ИСТОЧНИК:</span> ' + esc(d.source || '?');
}
async function buildCmd(){
  const loras = [...document.querySelectorAll('.lora:checked')].map(c=>c.value);
  const embs = [...document.querySelectorAll('.emb:checked')].map(c=>c.value);
  const body = {pipeline:document.getElementById('pipe').value,
                ckpt:document.getElementById('ckpt').value,
                mode:document.getElementById('mode').value, loras:loras, embs:embs,
                prompt:document.getElementById('pr').value,
                auto:document.getElementById('autox').checked};
  const d = await (await fetch('/api/build',{method:'POST',body:JSON.stringify(body)})).json();
  window.built = d;
  let t = esc(d.name) + ' <button onclick="addBuilt()">в очередь</button>\\n' + esc(d.cmd);
  if((d.added||[]).length)
    t += '\\n<span class="hdr">АВТО-ДОБАВЛЕНО В ПРОМПТ:</span>\\n' +
         d.added.map(a=>'  + '+esc(a)).join('\\n');
  document.getElementById('out').innerHTML = t;
}
async function addBuilt(){
  await fetch('/api/queue/add',{method:'POST',body:JSON.stringify({name:window.built.name,cmd:window.built.cmd})});
  showQueue();
}
async function scan(){
  const d = await (await fetch('/api/scan',{method:'POST'})).json();
  window.opts = d.options || {};
  document.getElementById('pipe').innerHTML =
    '<option value=sd15>SD 1.5</option>' +
    '<option value=sdxl>SDXL</option>' +
    '<option value=qwen>QWEN</option>' +
    '<option value=zimage>Z-IMAGE</option>';
  fillCons();
  fillCivSelect();
  let t = 'ЦЕЛЬ: ' + esc(d.target) + '\\n';
  t += '<span class="hdr">ВЫХОДЫ (output):</span> ' + esc(d.options.outputDir||'') + '\\n\\nНАЙДЕНО:\\n';
  (d.files||[]).forEach(f => t += '  ' + esc(f.rel) + ' (' + f.size_gb + ' ГБ) -> ' + esc(f.role_ru) +
    (f.keys_head ? ' <span class="miss">(ключи: ' + esc(f.keys_head) + ')</span>' : '') + '\\n');
  t += '\\nНЕ ХВАТАЕТ:\\n';
  if(!(d.missing||[]).length) t += '  <span class="ok">всё на месте!</span>\\n';
  (d.missing||[]).forEach(m => t += '  <span class="miss">' + esc(m.role_ru) +
    (m.url ? '\\n    url: <a href="' + esc(m.url) + '" target="_blank" style="color:#6cf">' + esc(m.url.split('/').pop()) + '</a>' : '') +
    '\\n    положить в: ' + esc(m.put_to) + '</span>\\n');
  const wc = (d.options||{}).wildcards||[];
  if(wc.length)
    t += '\\n<span class="hdr">WILDCARDS (пиши в промпте __имя__ или __XMods/...__):</span>\\n  ' +
         esc(wc.slice(0,30).join(', ')) + (wc.length>30 ? '... (всего '+wc.length+')' : '') + '\\n';
  if(((d.options||{}).wcErrors||[]).length)
    t += '\\n<span class="miss">⚠️ ПРОБЛЕМЫ С WILDCARDS:</span>\\n' +
         (d.options.wcErrors||[]).map(e=>'  ' + esc(e)).join('\\n') + '\\n';
  t += '\\n' + esc(d.ram_tips || '');
  document.getElementById('out').innerHTML = t;
  window.lastScan = JSON.stringify(d);
}
async function civScan(){
  const out = document.getElementById('out');
  out.innerText = 'Спрашиваю Civitai... (~1 сек на новую лору)';
  try{
    const d = await (await fetch('/api/civitai',{method:'POST'})).json();
    const r = d.report || {};
    let t = 'ОТЧЁТ CIVITAI:\\n';
    t += '  API: ' + esc(String(r.api||'?')) + '\\n';
    t += '  лор всего: ' + (r.total||0) +
         ' | с триггерами из метаданных: ' + (r.skipped_meta||0) +
         ' | уже в кэше: ' + (r.cached||0) + '\\n';
    const q = r.queried || {};
    t += '  опрошено сейчас: ' + Object.keys(q).length + '\\n';
    for(const [k,v] of Object.entries(q))
      t += '    ' + esc(k) + ': ' + (Array.isArray(v) ? (v.join(', ') || '(слов нет)') : esc(String(v))) + '\\n';
    if(!Object.keys(q).length) t += '    (все лоры уже с триггерами или в кэше)\\n';
    t += '\\nЖми «Сканировать», чтобы конструктор подхватил слова.';
    out.innerHTML = t;
  }catch(e){
    out.innerHTML = 'ОШИБКА CIVITAI: ' + esc(String(e));
  }
}
async function addCustom(){
  const n = document.getElementById('qname').value;
  const c = document.getElementById('qcmd').value;
  await fetch('/api/queue/add',{method:'POST',body:JSON.stringify({name:n,cmd:c})});
  showQueue();
}
async function showQueue(){
  const out = document.getElementById('out');
  try{
    const d = await (await fetch('/api/queue',{method:'POST'})).json();
    const jobs = d.jobs || [];
    let t = '<span class="hdr">ОЧЕРЕДЬ:</span>\\n';
    if(!jobs.length) t += '  пусто\\n';
    jobs.forEach(j => t += '  [' + j.id + '] <span class="' +
      (j.status=='running'?'run':(j.status=='done'?'ok':'miss')) + '">' + esc(j.status) +
      '</span> — ' + esc(j.name) + ' <button onclick="showLog(' + j.id + ')">лог</button>' +
      (j.status=='waiting' ? ' <button onclick="rmJob(' + j.id + ')">убрать</button>' : '') + '\\n');
    const run = jobs.find(j => j.status=='running');
    if(run){
      const l = await (await fetch('/api/queue/log',{method:'POST',body:JSON.stringify({id:run.id})})).json();
      t += '\\n<span class="hdr">СЕЙЧАС ВЫПОЛНЯЕТСЯ [' + run.id + '] ' + esc(run.name) + ':</span>\\n' + esc(l.log || '');
    }
    out.innerHTML = t;
  }catch(e){
    out.innerHTML = 'ОШИБКА при показе очереди: ' + esc(String(e));
  }
}
async function showLog(id){
  const d = await (await fetch('/api/queue/log',{method:'POST',body:JSON.stringify({id:id})})).json();
  document.getElementById('out').innerHTML = 'ЛОГ задачи ' + id + ':\\n' + esc(d.log);
}
async function rmJob(id){
  await fetch('/api/queue/remove',{method:'POST',body:JSON.stringify({id:id})});
  showQueue();
}
async function stopQ(){ await fetch('/api/queue/stop',{method:'POST'}); showQueue(); }
async function clearQ(){ await fetch('/api/queue/clear',{method:'POST'}); showQueue(); }
async function organize(){
  const d = await (await fetch('/api/organize',{method:'POST'})).json();
  document.getElementById('out').innerHTML =
    'Перемещено:\\n' + esc(d.moved.join('\\n') || 'нечего раскидывать');
}
async function explain(){
  const d = await (await fetch('/api/explain',{method:'POST',body:window.lastScan||'{}'})).json();
  document.getElementById('out').innerHTML +=
    '\\n\\nИИ ГОВОРИТ:\\n' + esc(d.text || d.error || 'тишина');
}
async function guess(){
  const p = document.getElementById('img').value;
  document.getElementById('out').innerText = 'Думаю...';
  const d = await (await fetch('/api/prompt',{method:'POST',body:JSON.stringify({path:p})})).json();
  document.getElementById('out').innerHTML =
    'ИСТОЧНИК: ' + esc(d.source) + '\\n\\n' + esc(d.prompt);
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, data: bytes, ctype: str) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, TimeoutError):
            pass

    def _json(self, obj):
        self._send(json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/api/downloads":
            return self._json(DOWNLOADS)
        self._send(HTML.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        try:
            if self.path == "/api/scan":
                res = scan()
                res["ram_tips"] = RAM_TIPS
                return self._json(res)
            if self.path == "/api/organize":
                return self._json({"moved": organize()})
            if self.path == "/api/explain":
                return self._json({"text": ask_llm(json.dumps(body, ensure_ascii=False))})
            if self.path == "/api/prompt":
                path = body.get("path", "")
                meta = read_metadata(path)
                if meta:
                    return self._json({"source": "metadata", "prompt": meta})
                try:
                    return self._json({"source": "vl", "prompt": ask_vl(path)})
                except Exception as e:
                    return self._json({"source": "none",
                                       "prompt": f"метаданных нет; VL-сервер на 8081 не отвечает: {e}"})
            if self.path == "/api/craft":
                return self._json(craft_prompt(body.get("base", ""),
                                               body.get("mode", "template"),
                                               body.get("genre", "portrait")))
            if self.path == "/api/civitai/manual":
                name = body.get("name", "")
                words = [w.strip() for w in re.split(r"[,\n;]+", body.get("words", "")) if w.strip()]
                if name:
                    CIV_CACHE[name] = {"words": words, "src": "manual"}
                    civ_save_cache()
                return self._json({"words": words})
            if self.path == "/api/build":
                if not LAST_ROLES:
                    scan()
                return self._json(build_one(body.get("pipeline", "sd15"), body.get("ckpt", ""),
                                            body.get("loras", []), body.get("mode", "img"), LAST_ROLES,
                                            body.get("prompt", ""), body.get("auto", True),
                                            body.get("embs", [])))
            if self.path == "/api/civitai":
                return self._json({"report": civ_enrich()})
            if self.path == "/api/queue":
                with QUEUE_LOCK:
                    jobs = [{k: j[k] for k in ("id", "name", "cmd", "status")} for j in QUEUE]
                return self._json({"jobs": jobs})
            if self.path == "/api/queue/add":
                return self._json({"id": add_job(body.get("name", ""), body.get("cmd", ""))})
            if self.path == "/api/queue/stop":
                return self._json({"status": stop_current()})
            if self.path == "/api/queue/clear":
                return self._json({"status": clear_done()})
            if self.path == "/api/queue/remove":
                return self._json({"status": remove_job(int(body.get("id", 0)))})
            if self.path == "/api/queue/log":
                return self._json({"log": read_tail(int(body.get("id", 0)))})
        except Exception as e:
            return self._json({"error": str(e)})
        self._json({"error": "unknown"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    load_queue()
    civ_load_cache()
    wc_load_yaml()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f"Concierge на http://{HOST}:{PORT}, воркер очереди запущен")
    print(f"Все картинки/видео → {OUTPUT_DIR}")
    QuietServer((HOST, PORT), Handler).serve_forever()