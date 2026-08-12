# Docker deploy (optional)

Local Sloan demos should use `run_local_webapp.py` / `start_local_webapp.*`.
This file is for rebuilding GPU images on a host such as RunPod.

Split so frequent app deploys stay small:

| Image | Contents | When to rebuild |
|-------|----------|-----------------|
| `YOUR_DOCKERHUB/fencing-base:latest` | CUDA, conda, PyTorch, OpenMMLab, RTMDet + MotionBERT | PyTorch / mmcv / mmdet / mmpose bumps |
| `YOUR_DOCKERHUB/fencing-weights:latest` | ViTPose-H, touch, attack, ODTrack | Checkpoint changes |
| `YOUR_DOCKERHUB/fencing-mmpose:latest` | Flask app + templates on top of the above | App code changes |

Override tags via env:

```bash
export BASE_IMAGE=YOUR_DOCKERHUB/fencing-base:latest
export WEIGHTS_IMAGE=YOUR_DOCKERHUB/fencing-weights:latest
export APP_IMAGE=YOUR_DOCKERHUB/fencing-mmpose:latest
```

## App-only rebuild

```bash
BUILD_BASE=0 BUILD_WEIGHTS=0 ./scripts/build-docker.sh
```

## Full first build

```bash
chmod +x scripts/prepare-weights-context.sh scripts/build-docker.sh
./scripts/build-docker.sh
```

## Runtime env (production)

Set secrets in the host environment — never commit them:

- `SECRET_KEY`
- `ADMIN_TOKEN` (required to enable `/api/admin/*`; empty disables admin)
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- `R2_PUBLIC_URL` — Cloudflare R2 public base, e.g. `https://pub-xxxxxxxx.r2.dev`
- `CLOUDFLARE_TUNNEL_TOKEN` — only if using `start.sh` tunnel mode
- Optional: `OPENAI_API_KEY` / `OPENAI_MODEL`, or local Ollama defaults in the Dockerfile

## Coaching LLM

Image defaults to local Ollama. For cloud OpenAI:

```text
ENABLE_OLLAMA=0
OPENAI_BASE_URL=
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

## ODTrack

```bash
python setup_odtrack.py
```

In Docker, vendor is cloned during the app image build; weights come from the weights image.
