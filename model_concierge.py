"""Model Concierge v3.35: ПРЕСЕТЫ + ПРОФИЛИ + ETA. Профили генерации
(default/фотореализм/аниме/комикс) со своими шагами/CFG/негативом (Z-Image
профиль не трогает турбо-режим); библиотека пресетов с категориями, тегами и
историей изменений; экспорт/импорт файлом; ETA-индикатор по статистике прошлых
генераций (gen_stats.json). Плюс всё из v3.34 и вшитая правка ae.safetensors
-> vae_zimage."""

from __future__ import annotations

import base64
import difflib
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
VERSION = "v3.35"
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
PRESETS_FILE = SD_CCPP_ROOT / "presets.json"
STATS_FILE = SD_CCPP_ROOT / "gen_stats.json"
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

GENPROFILES = {
    "default": {"ru": "Дефолт", "steps": None, "cfg": None, "neg": None},
    "photo": {"ru": "Фотореализм", "steps": 25, "cfg": 5.0,
              "neg": "blurry, low quality, deformed, painting, illustration, anime, cartoon, 3d render, plastic skin, oversaturated, bad anatomy"},
    "anime": {"ru": "Аниме", "steps": 20, "cfg": 7.0,
              "neg": "blurry, low quality, deformed, realistic, photo, 3d render, plastic skin"},
    "comic": {"ru": "Комикс", "steps": 20, "cfg": 7.0,
              "neg": "blurry, low quality, deformed, realistic, photo, 3d render, smooth skin"},
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
    "animatediff_mm": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
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
SCAN_EXT = (".safetensors", ".gguf", ".pth", ".pt", ".ckpt")
DOWNLOADS: dict[str, dict] = {}

# ==================== ОЧЕРЕДЬ ====================
QUEUE: list[dict] = []
QUEUE_LOCK = threading.Lock()
CURRENT_PROC: subprocess.Popen | None = None
JOB_COUNTER = 0
LAST_ROLES: dict = {}
LAST_FILES: list[dict] = []
LORA_TRIGGERS: dict[str, list[str]] = {}
LORA_COMPAT: dict[str, str] = {}
CIV_CACHE_FILE = SD_CCPP_ROOT / "civitai_cache.json"
CIV_CACHE: dict[str, dict] = {}
CIV_TOKEN_FILE = SD_CCPP_ROOT / "civitai_token.txt"
CIV_TOKEN = CIV_TOKEN_FILE.read_text(encoding="utf-8").strip() if CIV_TOKEN_FILE.exists() else ""
WC_YAML: dict[str, list[tuple[float, str]]] = {}
WC_YAML_SRC: dict[str, str] = {}
WC_ERRORS: list[str] = []
GEN_STATS: list[dict] = []
# =================================================================


def _out(name: str) -> str:
    """Путь в output\\ (для -o и -i аргументов sd-cli)."""
    return str(OUTPUT_DIR / name)


def parse_cmd_meta(cmd: str) -> dict:
    """Достаёт параметры sd-cli из команды (sidecar-JSON, «повторить»)."""
    def one(pat: str) -> str | None:
        m = re.search(pat, cmd)
        return m.group(1) if m else None
    return {
        "prompt": one(r'-p "([^"]*)"'),
        "negative": one(r'-n "([^"]*)"'),
        "model": one(r'-m "([^"]*)"'),
        "diffusion": one(r'--diffusion-model "([^"]*)"'),
        "out": one(r'-o "([^"]*)"'),
        "steps": int(one(r'--steps (\d+)') or 0) or None,
        "cfg": one(r'--cfg-scale ([\d.]+)'),
        "width": one(r'-W (\d+)'),
        "height": one(r'-H (\d+)'),
        "loras": re.findall(r"<lora:([^:>]+):", cmd),
    }


def write_sidecar(job: dict) -> None:
    """Пишет имя.png.json рядом с выходом: параметры + сид из лога."""
    meta = dict(job.get("meta") or parse_cmd_meta(job["cmd"]))
    out = meta.get("out")
    if not out:
        return
    p = Path(out)
    log_path = LOG_DIR / f"job_{job['id']:03d}.log"
    if log_path.exists():
        seeds = re.findall(r"seed (\d+)", log_path.read_text(encoding="utf-8", errors="ignore"))
        if seeds:
            meta["seed"] = int(seeds[-1])
    meta["job_id"] = job["id"]
    meta["task"] = job["name"]
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["concierge"] = VERSION
    try:
        p.with_name(p.name + ".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


# ==================== ПРЕСЕТЫ И СТАТИСТИКА ====================
def presets_load() -> list:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8")).get("presets", [])
        except Exception:
            return []
    return []


def presets_save(items: list) -> None:
    PRESETS_FILE.write_text(json.dumps({"presets": items}, ensure_ascii=False, indent=1),
                            encoding="utf-8")


def stats_load() -> None:
    global GEN_STATS
    if STATS_FILE.exists():
        try:
            GEN_STATS = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            GEN_STATS = []


def stats_append(entry: dict) -> None:
    """Добавляет замер длительности, держит последние 200 записей."""
    GEN_STATS.append(entry)
    del GEN_STATS[:-200]
    try:
        STATS_FILE.write_text(json.dumps(GEN_STATS, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def eta_estimate(pipeline: str, mode: str, steps: int) -> dict:
    """Среднее время прошлых генераций того же конвейера/режима, с поправкой на шаги."""
    rel = [e for e in GEN_STATS if e.get("pipeline") == pipeline and e.get("mode") == mode]
    if not rel:
        return {"eta": None, "n": 0}
    est = []
    for e in rel:
        es = e.get("steps") or steps or 20
        est.append(e["secs"] * (steps / es if steps and es else 1))
    return {"eta": int(sum(est) / len(est)), "n": len(rel)}
# =================================================================


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


def _good_tag(w: str) -> bool:
    """Тег похож на настоящий триггер: нижняя латиница, без служебных символов."""
    w = w.strip()
    if len(w) < 3 or len(w) > 40:
        return False
    if not re.match(r"^[a-z0-9]", w):
        return False
    if re.search(r"[\[\]{}():;\"'<>=*#]", w):
        return False
    return w.lower() not in STOP_TAGS and w.lower() not in {
        "none", "null", "true", "false", "n/a", "",
        "more", "info", "repeats", "description", "usage", "notes",
    }


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
    seen: set[str] = set()
    res: list[str] = []
    for w in out:
        key = w.lower()
        if key not in seen and _good_tag(w):
            seen.add(key)
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
    """Разбор варианта: веса N:: (0:: выкл), K$$-альтернативы, «{a|b}» целиком —
    список равновесных вариантов; прочее — один вариант."""
    raw = raw.strip()
    boxed = raw.startswith("{") and raw.endswith("}")
    if boxed:
        raw = raw[1:-1].strip()
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
    if boxed and "|" in raw:
        return [(1.0, p.strip()) for p in raw.split("|") if p.strip()]
    return [(1.0, raw)] if raw else []


def wc_load_yaml() -> None:
    """Терпимый парсер wildcard-YAML: вложенность по отступам, ключи любым
    алфавитом и с «/», списки в кавычках и без, folded-скаляры «>-» читаются
    целиком, последовательности на одном отступе с ключом приписываются к
    нему; опции накапливаются. WC_YAML_SRC помнит папку-источник каждого пути."""
    WC_YAML.clear()
    WC_YAML_SRC.clear()
    if not WILDCARD_DIR.exists():
        return
    for p in sorted(WILDCARD_DIR.rglob("*.yaml")):
        lines = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        stack: list[tuple[int, str]] = []
        i = 0
        while i < len(lines):
            raw = lines[i]
            s = raw.strip()
            i += 1
            if not s or s.startswith("#") or s == "---":
                continue
            indent = len(raw) - len(raw.lstrip())
            if s.startswith("- "):
                while stack and stack[-1][0] > indent:
                    stack.pop()
                if not stack:
                    continue
                content = s[2:].strip()
                if re.match(r"^[>|][+-]?\d*$", content):
                    body: list[str] = []
                    while i < len(lines):
                        nxt = lines[i]
                        if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                            break
                        body.append(nxt.strip())
                        i += 1
                    content = " ".join(b for b in body if b)
                elif content.startswith('"') and content.count('"') == 1:
                    while i < len(lines):
                        nxt = lines[i].rstrip()
                        i += 1
                        content += "\n" + nxt
                        if nxt.endswith('"'):
                            break
                if content.startswith('"') and content.rstrip().endswith('"'):
                    content = content.strip()[1:-1]
                opts = _wc_parse_block(content)
                if opts:
                    path_key = "/".join(n for _, n in stack)
                    WC_YAML.setdefault(path_key, []).extend(opts)
                    rel = p.parent.relative_to(WILDCARD_DIR).as_posix()
                    WC_YAML_SRC[path_key] = rel if rel != "." else "(корень)"
                continue
            if s.endswith(":"):
                name = s[:-1].strip().strip('"').strip()
                if name:
                    while stack and stack[-1][0] >= indent:
                        stack.pop()
                    stack.append((indent, name))
                continue


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


def wc_browser() -> dict:
    """Список всех wildcards для морды: имя, источник, папка-группа, примеры.
    yaml группируется по РЕАЛЬНОЙ папке файла на диске (WC_YAML_SRC)."""
    items = []
    for stem, p in sorted(wc_map().items()):
        samples: list[str] = []
        count = 0
        try:
            with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    count += 1
                    if len(samples) < 3:
                        samples.append(s[:80])
        except Exception:
            continue
        rel = p.relative_to(WILDCARD_DIR)
        items.append({"name": stem, "kind": "txt",
                      "rel": rel.as_posix(),
                      "group": rel.parent.as_posix() if rel.parent.as_posix() != "." else "(корень)",
                      "count": count, "samples": samples})
    for path, opts in sorted(WC_YAML.items()):
        parts = path.split("/")
        items.append({"name": path, "kind": "yaml", "rel": path,
                      "group": WC_YAML_SRC.get(path, parts[0]),
                      "count": len(opts),
                      "samples": [t[:80] for _, t in opts[:3]]})
    return {"items": items}
# =================================================================


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


# ==================== МОДУЛЬ СОВМЕСТИМОСТИ ====================
def compat_report() -> dict:
    """Для каждой модели: ✅ поддерживаемые лоры, ⚠️ неопределённая семья, ❌ чужие."""
    if not LAST_FILES:
        scan()
    loras: list[dict] = []
    for f in LAST_FILES:
        r = f["role"]
        if r == "lora_sd":
            loras.append({"name": f["name"], "fam": LORA_COMPAT.get(f["name"], "")})
        elif r == "lora_qwen_image":
            loras.append({"name": f["name"], "fam": "qwen"})
        elif r == "lora_z_image":
            loras.append({"name": f["name"], "fam": "zimage"})
    models: list[dict] = []
    for f in LAST_FILES:
        if f["role"] == "full_sd15":
            models.append({"name": f["name"], "fam": "sd15"})
        elif f["role"] == "full_sdxl":
            models.append({"name": f["name"], "fam": "sdxl"})
        elif f["role"] == "dit_qwen_image":
            models.append({"name": f["name"], "fam": "qwen"})
        elif f["role"] == "dit_z_image":
            models.append({"name": f["name"], "fam": "zimage"})
    out = []
    for m in models:
        ok, warn, bad = [], [], []
        for l in loras:
            if m["fam"] in ("sd15", "sdxl"):
                if l["fam"] == m["fam"]:
                    ok.append(l["name"])
                elif l["fam"] == "":
                    warn.append(l["name"])
                else:
                    bad.append(l["name"])
            else:
                if l["fam"] == m["fam"]:
                    ok.append(l["name"])
                else:
                    bad.append(l["name"])
        out.append({"model": m["name"], "fam": m["fam"],
                    "ok": ok, "warn": warn, "bad": bad})
    return {"models": out}
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
        if "double_blocks" in s or "single_blocks" in s:
            return "lora_flux"
        if "input_blocks" in s or "output_blocks" in s or "middle_block_out" in s:
            return "lora_sd"
        if "transformer_blocks" in s and "double_blocks" not in s:
            return "lora_qwen_image"
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
    if low.startswith("ae."):
        return "vae_zimage"
    if any(t in s for t in ("lora_up", "lora_down", "lora_A", "lora_B", "lokr")):
        if low.startswith("zit") or "zit-" in low or "z_image" in low or "zimage" in low or "krea" in low:
            return "lora_z_image"
        if "double_blocks" in s or "single_blocks" in s:
            return "lora_flux"
        if "input_blocks" in s or "output_blocks" in s or "middle_block_out" in s:
            return "lora_sd"
        if "transformer_blocks" in s and "double_blocks" not in s:
            return "lora_qwen_image"
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
    if "animatediff" in low or "motion" in low or low.startswith("mm_") or "mm_sd" in low:
        return "animatediff_mm"
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
              custom_prompt: str = "", auto: bool = True, emb_rels: list | None = None,
              prof: dict | None = None) -> dict:
    """Собирает ОДНУ команду; ВСЕ выходные и входные файлы — в OUTPUT_DIR.
    Neg-эмбеддинги (имя содержит 'neg') едут в -n, позитивные — в промпт.
    prof (профиль генерации) подменяет шаги/CFG/негатив; Z-Image профиль не
    трогает турбо-шаги/CFG — только негатив."""
    prof = prof or {}
    steps_o = prof.get("steps")
    cfg_o = prof.get("cfg")
    neg_o = prof.get("neg")
    if pipeline == "zimage":
        steps_o = None
        cfg_o = None
    emb_rels = emb_rels or []
    neg_text = neg_o if neg_o else NEG.get(pipeline, NEG["sd15"])

    def sc(d_steps: int, d_cfg: float | None) -> str:
        st = steps_o if steps_o is not None else d_steps
        cf = cfg_o if cfg_o is not None else d_cfg
        return f"--steps {st}" + (f" --cfg-scale {cf}" if cf is not None else "")

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
            stem = Path(r).stem
            if "neg" in stem.lower():
                neg_text += f" {stem}"
                added.append(f"embedding {stem} -> в НЕГАТИВ")
            else:
                pl += f" {stem}"
                added.append(f"embedding {stem} -> в позитив")
    neg = f'-n "{neg_text}"'
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
                       f'-p "{pl}" {neg} {sc(20, 4.0)} -W 512 -H 512 -o "{img_path}"'}

    if pipeline == "zimage":
        dit = str(MODELS / ckpt_rel) if ckpt_rel else str(expected_path("dit_z_image"))
        if "krea" in stem.lower():
            return {"name": f"KREA2 · Картинка · {stem}{ltag} · 8 шагов", "added": added,
                    "cmd": f'{SD_CLI} --diffusion-model "{dit}"{la} -p "{pl}" {neg} '
                           f'{sc(8, 1.0)} -W 1024 -H 1024 -o "{img_path}"'}
        llm = str(MODELS / roles["llm_zimage"]["rel"]) if "llm_zimage" in roles else str(expected_path("llm_zimage"))
        vae = str(MODELS / roles["vae_zimage"]["rel"]) if "vae_zimage" in roles else str(expected_path("vae_zimage"))
        return {"name": f"Z-IMAGE · Картинка · {stem}{ltag} · 8 шагов", "added": added,
                "cmd": f'{SD_CLI} --diffusion-model "{dit}" --llm "{llm}" --vae "{vae}"{la} '
                       f'-p "{pl}" {neg} {sc(8, 1.0)} -W 1024 -H 1024 -o "{img_path}"'}

    m = f'-m "{MODELS / ckpt_rel}"'
    if pipeline == "sdxl":
        if mode == "hires":
            hr_name = f"{stem}{ltag}_hires.png"
            return {"name": f"SDXL · hires · {stem}{ltag}", "added": added,
                    "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} {sc(20, None)} -W 1024 -H 1024 '
                           f'--hires --hires-width 1920 --hires-height 1080 --hires-steps 10 -o "{_out(hr_name)}"'}
        return {"name": f"SDXL · Картинка · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} {sc(20, None)} -W 1024 -H 1024 -o "{img_path}"'}

    if mode == "hires":
        hr_name = f"{stem}{ltag}_hires.png"
        return {"name": f"SD1.5 · hires · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} {sc(20, None)} -W 512 -H 512 '
                       f'--hires --hires-width 1366 --hires-height 768 --hires-steps 10 -o "{_out(hr_name)}"'}
    if mode in ("txt2vid", "img2vid") and "animatediff_mm" in roles:
        mm = f'--motion-module "{MODELS / roles["animatediff_mm"]["rel"]}"'
        if mode == "txt2vid":
            tv_name = f"{stem}{ltag}_t2v.webm"
            return {"name": f"SD1.5 · ТЕКСТ→ВИДЕО · {stem}{ltag}", "added": added,
                    "cmd": f'{SD_CLI} {m}{ea} -M vid_gen {mm} --video-frames 32 --fps 8 -p "{pl}" {neg} '
                           f'{sc(12, None)} -W 512 -H 512 -o "{_out(tv_name)}"'}
        iv_name = f"{stem}{ltag}_i2v.webm"
        return {"name": f"SD1.5 · КАРТИНКА→ВИДЕО · {stem}{ltag}", "added": added,
                "cmd": f'{SD_CLI} {m}{ea} -M vid_gen {mm} --video-frames 32 --fps 8 -i "{img_path}" --strength 0.5 '
                       f'-p "{pl}" {neg} {sc(12, None)} -W 512 -H 512 -o "{_out(iv_name)}"'}
    return {"name": f"SD1.5 · Картинка · {stem}{ltag}", "added": added,
            "cmd": f'{SD_CLI} {m}{la}{ea} -p "{pl}" {neg} {sc(20, None)} -W 512 -H 512 -o "{img_path}"'}


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
  - ВСЕ КАРТИНКИ И ВИДЕО СОХРАНЯЮТСЯ В: output\\ (рядом — .json с параметрами)"""


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
        data = [{k: j.get(k) for k in ("id", "name", "cmd", "status", "meta")} for j in QUEUE]
    QUEUE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def add_job(name: str, cmd: str, meta: dict | None = None) -> int:
    global JOB_COUNTER
    with QUEUE_LOCK:
        JOB_COUNTER += 1
        QUEUE.append({"id": JOB_COUNTER, "name": name or f"job {JOB_COUNTER}",
                      "cmd": cmd, "status": "waiting", "meta": meta or {}})
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
        t0 = time.time()
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
            dur = time.time() - t0
            if job["status"] == "done":
                write_sidecar(job)
                m = parse_cmd_meta(job["cmd"])
                stats_append({
                    "pipeline": (job.get("meta") or {}).get("pipeline", "sd15"),
                    "mode": "video" if "vid_gen" in job["cmd"] else ("hires" if "--hires" in job["cmd"] else "img"),
                    "steps": m.get("steps") or 20,
                    "secs": round(dur, 1),
                })
            save_queue()


def gallery_items() -> dict:
    """Список выходов (новые первые) + sidecar-метаданные, до 120 штук."""
    items = []
    if OUTPUT_DIR.exists():
        files = [p for p in OUTPUT_DIR.iterdir()
                 if p.is_file() and p.suffix in (".png", ".webm", ".jpg")]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:120]:
            meta = None
            jp = p.with_name(p.name + ".json")
            if jp.exists():
                try:
                    meta = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    meta = None
            items.append({"name": p.name, "mtime": int(p.stat().st_mtime),
                          "size": p.stat().st_size, "meta": meta})
    return {"items": items}


def prompt_diff(a: str, b: str) -> dict:
    """Diff двух промптов по тегам: '=' равно, '-' было в A, '+' добавлено в B."""
    ta = [t.strip() for t in a.split(",") if t.strip()]
    tb = [t.strip() for t in b.split(",") if t.strip()]
    ops = []
    sm = difflib.SequenceMatcher(a=ta, b=tb)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            ops.append({"op": "=", "text": ", ".join(ta[i1:i2])})
        else:
            if op in ("replace", "delete"):
                ops.append({"op": "-", "text": ", ".join(ta[i1:i2])})
            if op in ("replace", "insert"):
                ops.append({"op": "+", "text": ", ".join(tb[j1:j2])})
    return {"ops": ops}


def scan() -> dict:
    global LAST_ROLES, LAST_FILES, LORA_COMPAT
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
    LORA_COMPAT = compat_opt
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
    if "animatediff_mm" not in roles:
        missing.append({"role": "animatediff_mm", "role_ru": ROLE_RU["animatediff_mm"],
                        "url": SOURCES["animatediff_mm"],
                        "put_to": str(expected_path("animatediff_mm"))})
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
summary{cursor:pointer}
.miss{color:#f66}.ok{color:#6f6}.hdr{color:#6cf}.run{color:#fc6}</style></head><body>
<h2>Model Concierge v3.35</h2>
<button onclick='scan()'>Сканировать папку models</button>
<button onclick='organize()'>Раскидать по папкам</button>
<button onclick='showCompat()'>🧩 Совместимость</button>
<button onclick='showGallery()'>🖼 Галерея</button>
<button onclick='showWc()'>🎭 Wildcards</button>
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
<select id='mode' onchange='updateEta()'></select>
<select id='prof' onchange='updateEta()'>
  <option value='default'>профиль: дефолт</option>
  <option value='photo'>профиль: фотореализм</option>
  <option value='anime'>профиль: аниме</option>
  <option value='comic'>профиль: комикс</option>
</select>
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
<button onclick='buildCmd()'>Собрать команду</button> <span id='eta' class='run'></span>
<br>
<span class='hdr'>📚 ПРЕСЕТЫ:</span>
<input id='psname' size='18' placeholder='имя пресета'>
<input id='pscat' size='10' placeholder='категория'>
<input id='pstags' size='16' placeholder='теги, через запятую'>
<select id='pslist'></select>
<button onclick='psApply()'>📂 Применить</button>
<button onclick='psSave()'>💾 Обновить</button>
<button onclick='psSaveAs()'>💾+ Новый</button>
<button onclick='psDelete()'>🗑</button>
<button onclick='psExport()'>⬇ Экспорт</button>
<input type='file' id='psfile' accept='.json' onchange='psImport(this)' style='width:110px'>
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
<div id='gal'></div>
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
  const ta = document.createElement('textarea');
  ta.value = path;
  document.body.appendChild(ta); ta.select();
  try{ document.execCommand('copy'); }catch(e){}
  document.body.removeChild(ta);
  document.getElementById('out').innerHTML =
    '<span class="ok">Скопировано в буфер:</span> ' + esc(path) +
    '\\nОткрой Проводник (Win+E) и вставь (Ctrl+V) в адресную строку.';
}
async function showCompat(){
  const out = document.getElementById('out');
  out.innerText = 'Считаю совместимость...';
  const d = await (await fetch('/api/compat',{method:'POST'})).json();
  const ru = {sd15:'SD 1.5', sdxl:'SDXL', qwen:'QWEN-IMAGE', zimage:'Z-IMAGE'};
  let t = '<span class="hdr">СОВМЕСТИМОСТЬ МОДЕЛЕЙ И ЛОР:</span>\\n';
  (d.models||[]).forEach(m => {
    t += '\\n' + esc(m.model) + '  [' + (ru[m.fam]||m.fam) + ']\\n';
    t += '  <span class="ok">✅ поддерживаются (' + m.ok.length + '):</span>\\n';
    m.ok.forEach(n => t += '     ' + esc(n) + '\\n');
    if((m.warn||[]).length){
      t += '  <span class="run">⚠️ семья не определена (' + m.warn.length + '):</span>\\n';
      m.warn.forEach(n => t += '     ' + esc(n) + '\\n');
    }
    t += '  <span class="miss">❌ не поддерживаются: ' + m.bad.length + '</span>\\n';
  });
  out.innerHTML = t;
}
async function showGallery(){
  const d = await (await fetch('/api/gallery',{method:'POST'})).json();
  window.gal = d.items||[]; window.cmpA = null;
  document.getElementById('gal').innerHTML =
    '<span class="hdr">ГАЛЕРЕЯ output\\:</span> ' +
    '<input id="galf" size="30" placeholder="фильтр: модель / слово из промпта" oninput="paintGrid()">' +
    '<select id="gals" onchange="paintGrid()"><option value="new">сначала новые</option>' +
    '<option value="old">сначала старые</option><option value="model">по модели</option></select>' +
    '<div id="galgrid" style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px"></div>';
  paintGrid();
}
function paintGrid(){
  const f = ((document.getElementById('galf').value)||'').toLowerCase();
  const sort = document.getElementById('gals').value;
  let items = window.gal||[];
  if(f) items = items.filter(it=>{ const m=it.meta||{};
    return (it.name+' '+(m.model||'')+' '+(m.diffusion||'')+' '+(m.prompt||'')+' '+
            (m.loras||[]).join(' ')).toLowerCase().includes(f); });
  if(sort=='old') items = items.slice().reverse();
  if(sort=='model') items = items.slice().sort((a,b)=>
    String((a.meta||{}).model||'').localeCompare(String((b.meta||{}).model||'')));
  window.galView = items;
  let t = '';
  items.forEach((it,idx)=>{
    const m = it.meta||{};
    const tip = esc((m.model||m.diffusion||it.name)+' | seed '+(m.seed!=null?m.seed:'?')+
                    ' | cfg '+(m.cfg||'?')+' | steps '+(m.steps||'?'));
    const media = it.name.endsWith('.webm')
      ? '<video src="/api/img?name='+encodeURIComponent(it.name)+'" width="170" muted loop onmouseover="this.play()" onmouseout="this.pause()"></video>'
      : '<img src="/api/img?name='+encodeURIComponent(it.name)+'" width="170" loading="lazy">';
    t += '<div style="width:170px;cursor:pointer" title="'+tip+'" onclick="galDetail('+idx+')">'+media+
         '<div style="font-size:11px;color:#888">'+esc(it.name.slice(0,28))+'</div></div>';
  });
  document.getElementById('galgrid').innerHTML = t || '<span class="miss">пусто (фильтр ничего не нашёл)</span>';
}
function galDetail(i){
  const it = (window.galView||[])[i]; if(!it) return;
  const m = it.meta||{};
  document.getElementById('gal').innerHTML =
    '<button onclick="showGallery()">← в галерею</button> ' +
    '<button onclick="galRepeat('+i+')">🔁 повторить параметры</button> ' +
    '<button onclick="galDescribe('+i+')">👁 описать</button> ' +
    '<button onclick="window.cmpA='+i+'">⚖ задать A</button> ' +
    '<button onclick="galDiff('+i+')">⚖ diff A↔эта</button> ' +
    '<span class="hdr">'+esc(it.name)+'</span>' +
    '<div id="galx" style="margin:8px 0"></div>' +
    '<pre style="white-space:pre-wrap">'+esc(JSON.stringify(m,null,1))+'</pre>' +
    (it.name.endsWith('.webm')
      ? '<video src="/api/img?name='+encodeURIComponent(it.name)+'" width="512" controls></video>'
      : '<img src="/api/img?name='+encodeURIComponent(it.name)+'" style="max-width:512px">');
}
function galRepeat(i){
  const m = ((window.galView||[])[i]||{}).meta||{};
  const pipe = m.pipeline;
  if(pipe && ['sd15','sdxl','qwen','zimage'].includes(pipe))
    document.getElementById('pipe').value = pipe;
  fillCons();
  const o = window.opts||{};
  const list = pipe=='sdxl' ? (o.sdxl||[]) : pipe=='qwen' ? (o.qwenDit||[]) :
               pipe=='zimage' ? (o.zimageDit||[]) : (o.sd15||[]);
  const src = m.model||m.diffusion||'';
  const hit = list.find(r=> src.includes(r.split(/[\\\\/]/).pop()));
  if(hit) document.getElementById('ckpt').value = hit;
  if(m.prof) document.getElementById('prof').value = m.prof;
  document.getElementById('pr').value = (m.prompt||'').replace(/\\s*<lora:[^>]*>/g,'').trim();
  document.querySelectorAll('.lora').forEach(cb=>{
    cb.checked = (m.loras||[]).includes(cb.value.replace(/\\.[^.]+$/,''));
  });
  document.getElementById('out').innerHTML =
    '<span class="ok">Параметры возвращены в конструктор.</span> Проверь лоры и жми «Собрать команду».';
}
async function galDescribe(i){
  const it = (window.galView||[])[i]; if(!it) return;
  document.getElementById('galx').innerText = 'Описываю...';
  const d = await (await fetch('/api/describe',{method:'POST',
    body:JSON.stringify({name:it.name})})).json();
  document.getElementById('galx').innerHTML =
    '<span class="hdr">ИСТОЧНИК: '+esc(d.source)+'</span><br>'+esc(d.prompt);
}
async function galDiff(i){
  const a = (window.galView||[])[window.cmpA]; const b = (window.galView||[])[i];
  if(window.cmpA==null || !a || !b){ alert('Сначала «⚖ задать A» на другой картинке.'); return; }
  const d = await (await fetch('/api/diff',{method:'POST',
    body:JSON.stringify({a:((a.meta||{}).prompt)||'', b:((b.meta||{}).prompt)||''})})).json();
  let t = '<div style="line-height:1.6">';
  (d.ops||[]).forEach(o=>{
    const col = o.op=='+' ? '#6f6' : o.op=='-' ? '#f66' : '#888';
    t += '<span style="color:'+col+'">'+esc(o.text)+'</span>';
  });
  t += '</div>';
  document.getElementById('galx').innerHTML =
    '<span class="hdr">DIFF ПРОМПТОВ (красный — было в A, зелёный — добавилось в B):</span>'+t;
}
async function showWc(){
  const d = await (await fetch('/api/wildcards',{method:'POST'})).json();
  window.wc = d.items||[];
  document.getElementById('gal').innerHTML =
    '<span class="hdr">🎭 WILDCARDS (всего '+(window.wc.length)+'):</span> ' +
    '<input id="wcf" size="30" placeholder="поиск по имени или папке" oninput="paintWc()">' +
    '<div style="color:#888;font-size:12px">клик по 📁 раскрывает папку; «вставить» добавляет __имя__ в промпт; «🎲» показывает случайный вариант</div>' +
    '<div id="wcgrid" style="margin-top:10px"></div>';
  paintWc();
}
function paintWc(){
  const f = ((document.getElementById('wcf').value)||'').toLowerCase();
  const items = (window.wc||[]).filter(it=>
    it.name.toLowerCase().includes(f) || (it.rel||'').toLowerCase().includes(f) ||
    (it.group||'').toLowerCase().includes(f));
  window.wcView = items;
  const groups = {};
  items.forEach((it,idx)=>{ (groups[it.group||'(прочее)'] = groups[it.group||'(прочее)']||[]).push(idx); });
  let t = '';
  Object.keys(groups).sort().forEach(g=>{
    t += '<details '+(f?'open':'')+' style="margin:4px 0">' +
         '<summary style="color:#6cf">📁 '+esc(g)+' ('+groups[g].length+')</summary>';
    groups[g].forEach(idx=>{
      const it = items[idx];
      t += '<div style="margin:4px 0 4px 16px">' +
        '<button onclick="wcInsert('+idx+')">вставить</button> ' +
        '<button onclick="wcRoll('+idx+',this)">🎲</button> ' +
        '<span class="'+(it.kind=='yaml'?'hdr':'ok')+'">'+esc(it.name.split('/').pop())+'</span> ' +
        '<span style="color:#888">('+it.count+')</span> ' +
        '<span class="wcsm" style="color:#666">'+esc((it.samples||[]).join(' | '))+'</span></div>';
    });
    t += '</details>';
  });
  document.getElementById('wcgrid').innerHTML = t || '<span class="miss">не найдено</span>';
}
function wcInsert(idx){
  const it = (window.wcView||[])[idx]; if(!it) return;
  const pr = document.getElementById('pr');
  pr.value = (pr.value.trim() ? pr.value.trim()+', ' : '') + '__' + it.name + '__';
  document.getElementById('out').innerHTML =
    '<span class="ok">Добавлено в промпт:</span> __' + esc(it.name) + '__';
}
async function wcRoll(idx, btn){
  const it = (window.wcView||[])[idx]; if(!it) return;
  const d = await (await fetch('/api/wcpick',{method:'POST',body:JSON.stringify({name:it.name})})).json();
  const row = btn.parentElement.querySelector('.wcsm');
  if(row) row.innerHTML = '<span class="run">🎲 '+esc(d.pick||'—')+'</span>';
}
async function psLoad(){
  const d = await (await fetch('/api/presets',{method:'POST'})).json();
  window.ps = d.presets||[];
  document.getElementById('pslist').innerHTML =
    (window.ps||[]).map(p=>'<option value='+p.id+'>'+esc((p.category?p.category+': ':'')+(p.name||('пресет '+p.id)))+'</option>').join('') ||
    '<option value=0>(пусто)</option>';
}
function currentState(){
  return {
    name: document.getElementById('psname').value || 'пресет',
    category: document.getElementById('pscat').value,
    tags: document.getElementById('pstags').value.split(',').map(s=>s.trim()).filter(Boolean),
    pipeline: document.getElementById('pipe').value,
    ckpt: document.getElementById('ckpt').value,
    mode: document.getElementById('mode').value,
    prof: document.getElementById('prof').value,
    prompt: document.getElementById('pr').value,
    loras: [...document.querySelectorAll('.lora:checked')].map(c=>c.value),
    embs: [...document.querySelectorAll('.emb:checked')].map(c=>c.value),
    auto: document.getElementById('autox').checked,
  };
}
async function psSaveAs(){
  const d = await (await fetch('/api/preset/save',{method:'POST',
    body:JSON.stringify({id:null, state:currentState()})})).json();
  await psLoad();
  document.getElementById('pslist').value = d.id;
  document.getElementById('out').innerHTML = '<span class="ok">Пресет создан (id '+d.id+').</span>';
}
async function psSave(){
  const sel = parseInt(document.getElementById('pslist').value);
  if(!sel){ psSaveAs(); return; }
  await fetch('/api/preset/save',{method:'POST',
    body:JSON.stringify({id:sel, state:currentState()})});
  await psLoad();
  document.getElementById('pslist').value = sel;
  document.getElementById('out').innerHTML = '<span class="ok">Пресет обновлён (история записана).</span>';
}
async function psDelete(){
  const sel = parseInt(document.getElementById('pslist').value);
  if(!sel) return;
  await fetch('/api/preset/delete',{method:'POST',body:JSON.stringify({id:sel})});
  await psLoad();
  document.getElementById('out').innerHTML = '<span class="ok">Пресет удалён.</span>';
}
function psApply(){
  const id = parseInt(document.getElementById('pslist').value);
  const p = (window.ps||[]).find(x=>x.id===id); if(!p) return;
  if(p.pipeline) document.getElementById('pipe').value = p.pipeline;
  fillCons();
  const ck = document.getElementById('ckpt');
  if(p.ckpt && [...ck.options].some(o=>o.value===p.ckpt)) ck.value = p.ckpt;
  const md = document.getElementById('mode');
  if(p.mode && [...md.options].some(o=>o.value===p.mode)) md.value = p.mode;
  if(p.prof) document.getElementById('prof').value = p.prof;
  document.getElementById('pr').value = p.prompt||'';
  document.getElementById('psname').value = p.name||'';
  document.getElementById('pscat').value = p.category||'';
  document.getElementById('pstags').value = (p.tags||[]).join(', ');
  document.getElementById('autox').checked = p.auto!==false;
  document.querySelectorAll('.lora').forEach(cb=>cb.checked=(p.loras||[]).includes(cb.value));
  document.querySelectorAll('.emb').forEach(cb=>cb.checked=(p.embs||[]).includes(cb.value));
  updateEta();
  document.getElementById('out').innerHTML = '<span class="ok">Пресет «'+esc(p.name)+'» применён.</span>';
}
async function psExport(){
  const d = await (await fetch('/api/presets',{method:'POST'})).json();
  const blob = new Blob([JSON.stringify(d,null,1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'concierge_presets.json';
  a.click();
}
function psImport(inp){
  const f = inp.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = async () => {
    try{
      const data = JSON.parse(r.result);
      const d = await (await fetch('/api/preset/import',{method:'POST',
        body:JSON.stringify({presets: data.presets||[]})})).json();
      await psLoad();
      document.getElementById('out').innerHTML =
        '<span class="ok">Импортировано пресетов: '+(d.presets||[]).length+'.</span>';
    }catch(e){
      document.getElementById('out').innerHTML = '<span class="miss">Ошибка импорта: '+esc(String(e))+'</span>';
    }
  };
  r.readAsText(f);
  inp.value = '';
}
async function updateEta(){
  const pipe = document.getElementById('pipe').value;
  const mode = document.getElementById('mode').value;
  const prof = document.getElementById('prof').value;
  const steps = pipe==='zimage' ? 8 : ({photo:25, anime:20, comic:20}[prof] || 20);
  const d = await (await fetch('/api/eta',{method:'POST',
    body:JSON.stringify({pipeline:pipe, mode:mode, steps:steps})})).json();
  document.getElementById('eta').innerHTML =
    d.eta ? '⏱ ~'+Math.max(1, Math.round(d.eta/60))+' мин (по '+d.n+' прошлым)' : '';
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
    embs.map(e => {
      const n = e.split(/[\\\\/]/).pop();
      return '<label><input type=checkbox class=emb value="'+esc(e)+'">'+(/neg/i.test(n)?'🛡':'')+' '+esc(n)+'</label>';
    }).join('');
  updateEta();
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
                auto:document.getElementById('autox').checked,
                prof:document.getElementById('prof').value};
  const d = await (await fetch('/api/build',{method:'POST',body:JSON.stringify(body)})).json();
  window.built = d;
  let t = esc(d.name) + ' <button onclick="addBuilt()">в очередь</button>\\n' + esc(d.cmd);
  if((d.added||[]).length)
    t += '\\n<span class="hdr">АВТО-ДОБАВЛЕНО В ПРОМПТ:</span>\\n' +
         d.added.map(a=>'  + '+esc(a)).join('\\n');
  document.getElementById('out').innerHTML = t;
  updateEta();
}
async function addBuilt(){
  await fetch('/api/queue/add',{method:'POST',body:JSON.stringify(
    {name:window.built.name, cmd:window.built.cmd, meta:window.built.meta||{}})});
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
  psLoad();
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
        if self.path.startswith("/api/img?"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = Path(q.get("name", [""])[0]).name
            p = OUTPUT_DIR / name
            if p.exists() and p.is_file():
                ctype = {".png": "image/png", ".jpg": "image/jpeg",
                         ".webm": "video/webm"}.get(p.suffix, "application/octet-stream")
                try:
                    self._send(p.read_bytes(), ctype)
                except Exception:
                    pass
                return
            self._send(b"not found", "text/plain")
            return
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
            if self.path == "/api/compat":
                return self._json(compat_report())
            if self.path == "/api/wildcards":
                return self._json(wc_browser())
            if self.path == "/api/wcpick":
                name = body.get("name", "")
                pick = wc_pick(name)
                if pick is None:
                    p = wc_map().get(name)
                    if p is not None:
                        lines = [l.strip() for l in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
                                 if l.strip() and not l.strip().startswith("#")]
                        pick = random.choice(lines) if lines else None
                return self._json({"pick": pick})
            if self.path == "/api/gallery":
                return self._json(gallery_items())
            if self.path == "/api/diff":
                return self._json(prompt_diff(body.get("a", ""), body.get("b", "")))
            if self.path == "/api/describe":
                p = OUTPUT_DIR / Path(body.get("name", "")).name
                meta = read_metadata(str(p))
                if meta:
                    return self._json({"source": "metadata", "prompt": meta})
                try:
                    return self._json({"source": "vl", "prompt": ask_vl(str(p))})
                except Exception as e:
                    return self._json({"source": "none", "prompt": f"VL недоступен: {e}"})
            if self.path == "/api/presets":
                return self._json({"presets": presets_load()})
            if self.path == "/api/preset/save":
                items = presets_load()
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                st = body.get("state", {})
                pid = body.get("id")
                target = next((x for x in items if x.get("id") == pid), None) if pid else None
                if target is not None:
                    target.update(st)
                    target["modified"] = now
                    target.setdefault("history", []).append({"ts": now, "action": "updated"})
                    target["history"] = target["history"][-10:]
                else:
                    pid = max((x.get("id", 0) for x in items), default=0) + 1
                    target = {"id": pid, "created": now,
                              "history": [{"ts": now, "action": "created"}], **st}
                    items.append(target)
                presets_save(items)
                return self._json({"id": pid, "presets": items})
            if self.path == "/api/preset/delete":
                items = [x for x in presets_load() if x.get("id") != body.get("id")]
                presets_save(items)
                return self._json({"presets": items})
            if self.path == "/api/preset/import":
                old = presets_load()
                new = body.get("presets", [])
                nxt = max((x.get("id", 0) for x in old), default=0)
                old_ids = {x.get("id") for x in old}
                for p in new:
                    if p.get("id") in old_ids:
                        nxt += 1
                        p["id"] = nxt
                    old_ids.add(p.get("id"))
                merged = old + new
                presets_save(merged)
                return self._json({"presets": merged})
            if self.path == "/api/eta":
                return self._json(eta_estimate(body.get("pipeline", "sd15"),
                                               body.get("mode", "img"),
                                               int(body.get("steps") or 0)))
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
                prof = GENPROFILES.get(body.get("prof", "default"), {})
                res = build_one(body.get("pipeline", "sd15"), body.get("ckpt", ""),
                                body.get("loras", []), body.get("mode", "img"), LAST_ROLES,
                                body.get("prompt", ""), body.get("auto", True),
                                body.get("embs", []), prof=prof)
                meta = parse_cmd_meta(res["cmd"])
                meta["pipeline"] = body.get("pipeline", "sd15")
                meta["added"] = res.get("added", [])
                meta["prof"] = body.get("prof", "default")
                res["meta"] = meta
                return self._json(res)
            if self.path == "/api/civitai":
                return self._json({"report": civ_enrich()})
            if self.path == "/api/queue":
                with QUEUE_LOCK:
                    jobs = [{k: j.get(k) for k in ("id", "name", "cmd", "status")} for j in QUEUE]
                return self._json({"jobs": jobs})
            if self.path == "/api/queue/add":
                return self._json({"id": add_job(body.get("name", ""), body.get("cmd", ""),
                                                 body.get("meta"))})
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
    stats_load()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f"Concierge {VERSION} на http://{HOST}:{PORT}, воркер очереди запущен")
    print(f"Все картинки/видео → {OUTPUT_DIR} (рядом .json с параметрами)")
    QuietServer((HOST, PORT), Handler).serve_forever()