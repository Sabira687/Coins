# Tenge Scan: phone app (Assignment 2)

A web app that opens in the phone browser: live camera → coin detection → denomination and confidence → explanations
(nearest dataset images, attention map, embedding map, latency and device). The model runs on the laptop (DINOv2 ViT-S/14);
the phone only streams camera frames. **No image is stored**: frames are processed in memory and discarded.

## Run
```bash
source .venv/bin/activate
pip install -r app/requirements.txt          # once
python app/server.py
```
The terminal prints a QR code and an address like `https://192.168.x.x:8443`.

1. Laptop and phone must be in the same network (Wi-Fi, or the phone's hotspot with the laptop connected to it).
2. Scan the QR code. The browser warns about the self-signed certificate: *Advanced → continue*. The camera only works over https, which is why the certificate exists.
3. Allow the camera, tap START SCAN, point the phone at a coin (plain background works best). If the coin is not found, tap on it.
4. Optional: "Add to Home Screen" gives a full-screen app.

Test on the computer without a camera: `NOSSL=1 python app/server.py`, open `http://localhost:8000/?demo=1` (sample photos from `app/static/samples/`).

## Model files
The server uses `results_a2/final_dinov2.pt` and `results_a2/index.npz` (made by `a2_final_training.ipynb`, see `a2/RUN_FINAL_TRAINING.md`).
If they do not exist it falls back to the development model (weights of fold 1 of the paper experiments, index in `results_a2_dev/`), and the System block says "development (fold 1)".
Build another index: `python app/build_index.py --weights ... --out ... --p2-coins c3,c4 [--no-p1]`.
Settings: `UNCERTAIN_BELOW` (default 0.55: below this confidence the app answers "NOT SURE"), `PORT`, `WEIGHTS` + `INDEX`.

## Files
| Path | Content |
|---|---|
| `server.py` | FastAPI: `/api/detect`, `/api/classify`, `/api/info`, static files, HTTPS launcher |
| `static/index.html`, `static/app.js` | canvas front end (HUD, camera, animations, sound) |
| `static/samples/` | sample frames for the demo mode |
| `build_index.py` | embedding index for the nearest-neighbour gallery |
| `mockup/` | the approved design mock-up (static, no model) |
| `../coinlib/predictor.py` | detection, classification, kNN, attention, 2D map |

## How the explanations are made
- **Detection:** Hough circle, then ellipse fit (`extract_frames_p2.py`), the same code that built the dataset.
- **Nearest coins:** cosine similarity of L2-normalised DINOv2 embeddings against the training images; kNN vote with k = 5.
- **Attention:** attention of the CLS token in the last block, mean over heads, 16×16 patches, 3×3 median filter against isolated background tokens. It shows where the model looks, not a proof of why it decides.
- **Embedding map:** 2D LDA projection of the reference embeddings (class-separating), the query is marked.
- **NOT SURE:** shown when the top probability is below `UNCERTAIN_BELOW`.
