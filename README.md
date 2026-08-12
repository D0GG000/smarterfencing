# SmarterFencing

**Computer vision for fencing bout analysis** — prepared for the
[MIT Sloan Sports Analytics Conference](https://www.sloansportsconference.com/).

Upload a bout video. SmarterFencing detects scoreboard lights, tracks both
fencers, estimates pose, classifies touch location and attack type, then
returns an interactive timeline with style / coaching sections.

Live product: [smarterfencing.ai](https://smarterfencing.ai)

---

## What this repo includes

| Piece | Role |
|-------|------|
| Flask web app | Home, `/demo` upload UI, results, blog, optional coaching LLM |
| Vision pipeline | Light detection → fencer/scoreboard tracking → ViTPose → 3D lift → touch & attack models |
| Bundled weights | Touch v346, attack, RTMDet, MotionBERT, YOLO/RTMPose ONNX, ODTrack (~0.7 GB via Git LFS) |
| Homepage tour | Bundled demo bout (`static/demo/`) so the product walkthrough works offline |

ViTPose-H (~2.4 GB) exceeds GitHub’s per-file LFS limit and is downloaded separately.

---

## Quick start

### Requirements

- NVIDIA GPU + CUDA **11.8**
- Python **3.8** (conda recommended)
- [Git LFS](https://git-lfs.com)

### 1. Clone

```bash
git lfs install
git clone https://github.com/D0GG000/smarterfencing.git
cd smarterfencing
git lfs pull
```

### 2. Environment

```bash
conda create -y -n mmpose-env python=3.8
conda activate mmpose-env
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-ml.txt
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet mmpretrain mmpose
pip install -r requirements-web.txt
```

### 3. Download ViTPose-H

```powershell
.\scripts\download_weights.ps1
```

```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

### 4. Run locally

```powershell
.\start_local_webapp.ps1
```

```bash
./start_local_webapp.sh
```

Open **http://127.0.0.1:5000/** for the homepage tour, or **/demo** to analyze a bout.
Local mode skips Google login and stores uploads under `local_workspace/`.

Optional coaching LLM: install [Ollama](https://ollama.com) and `ollama pull llama3.2:3b`
(defaults in `run_local_webapp.py`), or set an OpenAI-compatible API key.

---

## Pipeline (high level)

1. **Lights** — scoreboard light regions → touch events  
2. **Tracking** — fencers + score lights (ODTrack / legacy fallback)  
3. **Pose** — ViTPose-H (COCO-17) on pre-touch windows  
4. **3D** — MotionBERT lift  
5. **Classification** — touch location + attack type models  
6. **Analysis UI** — timeline, overlays, editable macros, sectioned coaching (“Your style” / “Top 3 drills”)

---

## Repo layout

```text
app.py / demo.py          Flask + pipeline
static/ templates/        UI (incl. sectioned coaching)
static/demo/              Bundled homepage tour assets
checkpoints/              Detector / pose ONNX & MotionBERT
models/odtrack/           ODTrack weights
vendor/odtrack/           ODTrack source
mmpose_configs/           ViTPose configs
run_local_webapp.py       Local entrypoint
scripts/download_weights* ViTPose-H fetch
```

---

## Conference note

Prepared for **MIT Sloan Sports Analytics Conference** submissions and demos.
App version: see `version.py` (`262`).

---

## Optional: Docker deploy

Production Docker / GPU host notes are in `DOCKER.md` (optional; not required for local Sloan demos).

---

## License

See `LICENSE`.
