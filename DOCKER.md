# Docker build (three images)

Split so frequent app deploys stay small and fast.

| Image | Contents | When to rebuild |
|-------|----------|-----------------|
| `aliceds/fencing-base:latest` | CUDA, conda, PyTorch, OpenMMLab, RTMDet + MotionBERT checkpoints | PyTorch / mmcv / mmdet / mmpose version bumps |
| `aliceds/fencing-weights:latest` | COCO-17 ViTPose-H `.pth`, touch v3.46, attack `best_attack_3d_proximity.pth`, ODTrack | ViTPose / touch / attack / ODTrack checkpoint change |
| `aliceds/fencing-mmpose:latest` | Flask app, templates, cloudflared, ViTPose weights copied in | App code, templates, startup scripts (most builds) |

RunPod runs **`aliceds/fencing-mmpose:latest`** only.

## First time (full build)

From `app/`:

```bash
chmod +x scripts/prepare-weights-context.sh scripts/build-docker.sh
./scripts/build-docker.sh
```

Build order: **base → weights → app** (all pushed by default).

## App-only rebuild (typical deploy)

```bash
BUILD_BASE=0 BUILD_WEIGHTS=0 ./scripts/build-docker.sh
```

Requires `fencing-base` and `fencing-weights` already on the registry.

## Rebuild only what changed

```bash
# ViTPose-H COCO-17 weights refresh, same PyTorch stack
BUILD_BASE=0 ./scripts/build-docker.sh

# PyTorch / OpenMMLab upgrade, same app + weights tags
BUILD_WEIGHTS=0 BUILD_BASE=1 ./scripts/build-docker.sh
```

## Manual steps

```bash
# 1) ML base (~6–8 GB, rare)
docker buildx build --platform linux/amd64 \
  -f Dockerfile.base \
  -t aliceds/fencing-base:latest \
  --push .

# 2) COCO-17 ViTPose-H + touch v3.46 weights (~2–3 GB, rare)
./scripts/prepare-weights-context.sh
docker buildx build --platform linux/amd64 \
  -f Dockerfile.weights \
  -t aliceds/fencing-weights:latest \
  --push \
  weights-context

# 3) App (small layer on top)
docker buildx build --platform linux/amd64 \
  -f Dockerfile \
  --build-arg FENCING_BASE_IMAGE=aliceds/fencing-base:latest \
  --build-arg FENCING_WEIGHTS_IMAGE=aliceds/fencing-weights:latest \
  -t aliceds/fencing-mmpose:latest \
  --push .
```

## Environment overrides

- `BASE_IMAGE` — ML base tag (default `aliceds/fencing-base:latest`)
- `WEIGHTS_IMAGE` — ViTPose weights tag (default `aliceds/fencing-weights:latest`)
- `APP_IMAGE` — app tag (default `aliceds/fencing-mmpose:latest`)
- `BUILD_BASE=0` — skip base build
- `BUILD_WEIGHTS=0` — skip weights build
- `PUSH=0` — `--load` locally instead of `--push`
- `VITPOSE_H_CHECKPOINT` — explicit local COCO-17 ViTPose-H `.pth` for `prepare-weights-context.sh` (else uses `app/checkpoints/` or downloads OpenMMLab ViTPose-H)
- `ODTRACK_CHECKPOINT` — explicit ODTrack `.pth.tar` for `prepare-weights-context.sh` (else uses `app/models/odtrack/ODTrack_ep0300.pth.tar` if present)
- `TRACKER_BACKEND` — `odtrack` (default) or `legacy` for score-light tracking in the pipeline

## Local coaching LLM (Ollama on RunPod)

The app image installs **Ollama** and `start.sh` starts it by default:

- `ENABLE_OLLAMA=1` (default)
- `OPENAI_BASE_URL=http://127.0.0.1:11434/v1`
- `OPENAI_MODEL=llama3.2:3b`
- `OPENAI_API_KEY=ollama`
- Models stored on the network volume: `OLLAMA_MODELS=/workspace/ollama`

First boot after deploy downloads the model into `/workspace/ollama` (can take a few minutes). Later restarts reuse it.

To use **cloud OpenAI** instead on RunPod:

```text
ENABLE_OLLAMA=0
OPENAI_BASE_URL=
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Score-light tracking (ODTrack)

Local one-time setup (vendor clone + ~370MB weights):

```bash
cd app
python setup_odtrack.py
```

In Docker, the ODTrack vendor is cloned during the app image build; weights are copied from `fencing-weights`. If weights are missing at runtime, the pipeline falls back to the legacy ORB tracker.

## Pinned tags (recommended for production)

```text
aliceds/fencing-base:pytorch2.1-cu118-mmcv2.1
aliceds/fencing-weights:epoch65
aliceds/fencing-mmpose:2026-06-14
```

## Disk space

- Full build + push: **~15–25 GB** free on the builder
- Weights context bundles **one** ViTPose-H checkpoint (~2.4 GB)
- If build fails with "no space left on device": `docker builder prune -af`

## Runtime override

Mount a checkpoint and set `VITPOSE_H_CHECKPOINT=/path/to/model.pth` instead of using baked weights.

## R2 public URLs (required for video + email links)

Set `R2_PUBLIC_URL` to the **exact** public base from the Cloudflare R2 portal, for example:

```text
R2_PUBLIC_URL=https://pub-d2648056ec514bdea3d1935baa03c098.r2.dev
```

Videos and results JSON must use the same host. The app does **not** guess `pub-{R2_ACCOUNT_ID}.r2.dev` — that hash is not your account ID.
