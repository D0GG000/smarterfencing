# SmarterFencing (local package)

Production snapshot of **smarterfencing.ai** app code at version **262**.  
Same Flask `/demo` pipeline, results UI, coaching sections, blog, and Docker deploy scripts — ready to run locally and push to GitHub.

---

## What’s included

| Area | Contents |
|------|----------|
| **Web app** | `app.py`, templates, static assets, auth/queue/community/blog routes |
| **Pipeline** | Touch + attack classifiers, ViTPose / MotionBERT / arm-attempt ONNX, ODTrack |
| **Coaching UI** | Sectioned LLM analysis: **Your style** + **Top 3 drills** (with short *why*) |
| **Weights** | Touch v346, attack proximity models, ViTPose-H, RTMDet, MotionBERT, YOLO/RTMPose ONNX, ODTrack |
| **Deploy** | `Dockerfile*` + `DOCKER.md` (RunPod / Docker Hub flow) |

Runtime uploads/DB stay out of git (`local_workspace/` is gitignored).

---

## App sections (product)

1. **Home** — brand hero, product tour, analyzer / build log / about  
2. **Demo** — upload bout video, select fencers + lights, run pipeline  
3. **Result** — touch timeline, overlays, edits, share  
4. **Coaching** — separate **Your style** block and **Practice / Top 3 drills** cards  
5. **Archetypes** — style gallery  
6. **Blog / community / profile** — content and account surfaces  

---

## Repo layout

```text
.
├── app.py / demo.py / …     # Flask + pipeline
├── static/                  # theme, coaching UI, tour, brand
├── templates/               # pages (index, demo, result, …)
├── content/blog/            # markdown posts
├── checkpoints/             # RTMDet, MotionBERT, ONNX (+ ViTPose-H via download)
├── models/odtrack/          # ODTrack weights
├── vendor/odtrack/          # ODTrack source (vendored)
├── mmpose_configs/          # ViTPose config
├── scripts/                 # Docker build helpers
├── run_local_webapp.py      # local entrypoint
├── start_local_webapp.sh    # Linux/macOS / WSL
├── start_local_webapp.ps1   # Windows
├── requirements-web.txt
├── requirements-ml.txt
└── DOCKER.md
```

---

## Quick start (local)

### 1. Environment

Use a CUDA 11.8 + Python 3.8 env matching production (see `requirements-ml.txt` and `Dockerfile.base`), then:

```bash
pip install -r requirements-web.txt
```

### 2. Optional coaching LLM

- **Ollama (default):** install Ollama, `ollama pull llama3.2:3b`, leave defaults in `run_local_webapp.py`  
- **OpenAI:** set `OPENAI_API_KEY` / `OPENAI_MODEL` and clear `OPENAI_BASE_URL`

### 3. Run

```bash
# Linux / macOS / WSL
chmod +x start_local_webapp.sh
./start_local_webapp.sh

# Windows PowerShell
.\start_local_webapp.ps1
```

Open **http://127.0.0.1:5000/demo** (local mode auto-logs in; no Google gate).

---

## Weights

Most checkpoints are in the repo via Git LFS. **ViTPose-H (~2.4 GB)** is over GitHub’s 2 GB LFS file limit, so it stays local / downloadable:

```powershell
.\scripts\download_weights.ps1
```

```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

If you already copied it from the deploy tree, it lives at `checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth`.

---

## Docker / production deploy

See **DOCKER.md**. Typical app-only rebuild:

```bash
BUILD_BASE=0 BUILD_WEIGHTS=0 ./scripts/build-docker.sh
```

---

## Version

See `version.py` → `__version__ = "262"` (matches the latest deployed app image).
