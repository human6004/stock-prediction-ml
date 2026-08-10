"""Chup anh demo trang /chat that bang Chrome DevTools Protocol.

Yeu cau: Flask dang chay tai 127.0.0.1:5000 va .env co du 3 bien LLM.
Anh luu vao docs/slides/img/. Khong ve gia, khong dung DOM bom san.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cdp import WS, http_json

BASE = Path(__file__).resolve().parents[2]
IMG = BASE / "docs" / "slides" / "img"
IMG.mkdir(parents=True, exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222
PROFILE = Path(__file__).resolve().parent / "_chrome_profile"

QUESTIONS = [
    "Model dang dung la gi va ket qua tren TEST the nao?",
    "FPT co tin hieu UP khong?",
    "Chi so P/E va tin tuc moi nhat cua FPT the nao?",
]


def start_chrome():
    args = [
        CHROME,
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={PROFILE}",
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-scrollbars",
        "--window-size=1280,1600",
        "about:blank",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            http_json(f"http://127.0.0.1:{PORT}/json/version")
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("chrome debug port khong len")


def page_ws():
    targets = http_json(f"http://127.0.0.1:{PORT}/json")
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("khong co page target")
    return WS(pages[0]["webSocketDebuggerUrl"])


def ask(ws, text):
    """Go cau hoi vao form that va cho transcript co them cap hoi-dap."""
    before = ws.evaluate("document.querySelectorAll('#chat-transcript li').length")
    ws.evaluate(
        "(() => {"
        "const box = document.getElementById('chat-message');"
        f"box.value = {json.dumps(text)};"
        "box.dispatchEvent(new Event('input', {bubbles: true}));"
        "document.getElementById('chat-form')"
        ".dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));"
        "return true; })()"
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        count = ws.evaluate("document.querySelectorAll('#chat-transcript li').length")
        busy = ws.evaluate(
            "document.getElementById('chat-form').getAttribute('aria-busy')"
        )
        if count >= before + 2 and busy != "true":
            time.sleep(1.2)
            return True
        time.sleep(1.0)
    return False


def shoot(ws, name, full_page=True):
    metrics = ws.evaluate(
        "JSON.stringify({w: document.documentElement.scrollWidth,"
        " h: document.documentElement.scrollHeight})"
    )
    size = json.loads(metrics)
    params = {"format": "png", "captureBeyondViewport": True}
    if full_page:
        params["clip"] = {
            "x": 0,
            "y": 0,
            "width": min(size["w"], 1400),
            "height": min(size["h"], 3000),
            "scale": 1,
        }
    res = ws.call("Page.captureScreenshot", params, timeout=60)
    import base64

    out = IMG / name
    out.write_bytes(base64.b64decode(res["data"]))
    return out, size




def ensure_flask():
    """Bao dam Flask dang phuc vu 127.0.0.1:5000; neu chua thi tu spawn trong process nay."""
    import subprocess
    import urllib.request

    def alive():
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/chat", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    if alive():
        return None
    exe = str(BASE / ".venv" / "Scripts" / "python.exe")
    proc = subprocess.Popen(
        [exe, "app.py"],
        cwd=str(BASE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if alive():
            return proc
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Flask khong khoi dong duoc tai 127.0.0.1:5000.")

def launch_chrome():
    """Khoi dong Chrome headless kem remote debugging trong chinh process nay."""
    import shutil, subprocess, tempfile, urllib.error
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    exe = next((c for c in candidates if os.path.exists(c)), None)
    if exe is None:
        raise RuntimeError("Khong tim thay Chrome/Edge tren may.")
    profile = tempfile.mkdtemp(prefix="cdp_profile_")
    args = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--hide-scrollbars",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile}",
        "--window-size=1280,900",
        "about:blank",
    ]
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(60):
        try:
            http_json("http://127.0.0.1:9222/json/version")
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Chrome khong mo debug port 9222.")

def main():
    flask_proc = ensure_flask()
    chrome = launch_chrome()
    try:
        ws = page_ws()
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1280, "height": 1600, "deviceScaleFactor": 2, "mobile": False},
        )
        ws.call("Page.navigate", {"url": "http://127.0.0.1:5000/chat"})
        time.sleep(3)
        print("page title:", ws.evaluate("document.title"))

        for i, q in enumerate(QUESTIONS, start=1):
            ok = ask(ws, q)
            print(f"Q{i} answered={ok}: {q}")
            if not ok:
                print("  -> timeout, dung lai")
                break

        text = ws.evaluate(
            "Array.from(document.querySelectorAll('#chat-transcript li'))"
            ".map(li => li.innerText).join('\\n---\\n')"
        )
        Path(r"docs\slides\_chat_transcript.txt").write_text(text or "", encoding="utf-8")
        print("transcript chars:", len(text or ""))

        out, size = shoot(ws, "chatbot_demo.png")
        print("saved", out, out.stat().st_size, "page size", size)
    finally:
        try:
            chrome.terminate()
        except Exception:
            pass
        if flask_proc is not None:
            try:
                flask_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
