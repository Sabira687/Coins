"""Tenge Scan server: serves the phone web app and the model API.

    python app/server.py            # HTTPS on the local network, prints a QR code for the phone

Environment: WEIGHTS, INDEX (default: the final model in results_a2/, else the development model of fold 1).
"""
import os, socket, subprocess, sys, threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coinlib.config import CLASSES, SHORT
from coinlib.models import DEVICE
from coinlib.predictor import CoinPredictor
from coinlib.gridscale import EXPECTED_CELLS

STATIC = Path(__file__).parent / "static"
FINAL = (ROOT / "results_a2/final_dinov2.pt", ROOT / "results_a2/index.npz")
DEV = (ROOT / "results_paper/weights/E2__dinov2_s__P2__f1__s42.pt", ROOT / "results_a2_dev/index.npz")
if os.environ.get("WEIGHTS"):
    W_PATH, I_PATH, MODE = Path(os.environ["WEIGHTS"]), Path(os.environ["INDEX"]), "custom"
elif all(p.exists() for p in FINAL):
    (W_PATH, I_PATH), MODE = FINAL, "final"
else:
    (W_PATH, I_PATH), MODE = DEV, "dev"
UNCERTAIN_BELOW = float(os.environ.get("UNCERTAIN_BELOW", 0.55))

app = FastAPI(title="Tenge Scan")
lock = threading.Lock()
predictor = None


@app.on_event("startup")
def load():
    global predictor
    print(f"loading model ({MODE}): {W_PATH.name} on {DEVICE}", flush=True)
    predictor = CoinPredictor(W_PATH, I_PATH, uncertain_below=UNCERTAIN_BELOW)
    print("model ready", flush=True)


def decode(body: bytes):
    return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)


@app.get("/api/info")
def info():
    return dict(device=DEVICE, mode=MODE, model="DINOv2 ViT-S/14", n_index=int(len(predictor.emb)),
                classes=SHORT, uncertain_below=UNCERTAIN_BELOW, expected_cells=EXPECTED_CELLS)


@app.post("/api/detect")
async def detect(request: Request):
    img = decode(await request.body())
    if img is None: return JSONResponse({"found": False, "error": "bad image"}, 400)
    d = predictor.detect(img)
    return dict(found=d is not None, **(d or {}))


@app.post("/api/classify")
async def classify(request: Request, cx: float, cy: float, r: float):
    img = decode(await request.body())
    if img is None: return JSONResponse({"error": "bad image"}, 400)
    with lock:
        return predictor.classify(img, (cx, cy, r))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
for d in ("dataset", "dataset_p2"):          # thumbnails of the nearest neighbours; nothing else of the repo is served
    app.mount(f"/files/{d}", StaticFiles(directory=ROOT / d), name=d)


# ---------- launcher: HTTPS (the phone camera needs a secure context) + QR code
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip):
    d = Path(__file__).parent / "certs"; d.mkdir(exist_ok=True)
    key, crt, stamp = d / "key.pem", d / "cert.pem", d / "ip.txt"
    if not (key.exists() and crt.exists() and stamp.exists() and stamp.read_text() == ip):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-keyout", str(key), "-out", str(crt),
                        "-subj", "/CN=tenge-scan", "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
                       check=True, capture_output=True)
        stamp.write_text(ip)
    return key, crt


if __name__ == "__main__":
    import uvicorn
    if os.environ.get("NOSSL"):          # local testing on the computer itself (localhost counts as a secure context)
        uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning"); sys.exit()
    port = int(os.environ.get("PORT", 8443)); ip = lan_ip(); key, crt = ensure_cert(ip)
    url = f"https://{ip}:{port}"
    try:
        import qrcode
        q = qrcode.QRCode(border=1); q.add_data(url); q.print_ascii(invert=True)
    except Exception:
        pass
    print(f"\nOpen on the phone (same Wi-Fi): {url}\n(the browser warns about the self-signed certificate: Advanced → continue)\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_keyfile=str(key), ssl_certfile=str(crt), log_level="warning")
