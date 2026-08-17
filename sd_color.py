"""Цветной фильтр для sd-cli: красит важные строки лога прямо в консоли."""

from __future__ import annotations

import os
import re
import subprocess
import sys

os.system("")  # включает ANSI-цвета в консоли Windows

RED, GREEN, YELLOW, CYAN, RESET = "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"


def paint(line: str) -> str:
    """Раскрашивает одну строку лога по правилам."""
    if "LoRA tensors have been applied" in line:
        m = re.search(r"\(\s*(\d+)\s*/\s*(\d+)", line)
        if m and m.group(1) != "0":
            return GREEN + line + RESET      # лора села — зелёным
        return RED + line + RESET            # 0 / M — красным, глаз сразу цепляется
    if "[ERROR]" in line or "failed" in line.lower():
        return RED + line + RESET
    if "[WARN]" in line:
        return YELLOW + line + RESET
    if "Version:" in line or "generate image" in line or "sampling using" in line:
        return CYAN + line + RESET
    return line


def main() -> None:
    cmd = [r"D:\AI_Servers\sd-cpp\sd-cli.exe", *sys.argv[1:]]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as p:
        buf = b""
        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:                      # целые строки — красим
                line, buf = buf.split(b"\n", 1)
                sys.stdout.write(paint(line.decode("utf-8", "replace")) + "\n")
            if b"\r" in buf:                         # прогресс-бар — пропускаем живьём
                sys.stdout.write(buf.decode("utf-8", "replace"))
                sys.stdout.flush()
                buf = b""
        if buf:
            sys.stdout.write(paint(buf.decode("utf-8", "replace")))
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()