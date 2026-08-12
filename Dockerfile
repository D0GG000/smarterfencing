# App image (thin): Flask app + tunnel on top of fencing-base + fencing-weights.
ARG FENCING_BASE_IMAGE=aliceds/fencing-base:latest
ARG FENCING_WEIGHTS_IMAGE=aliceds/fencing-weights:latest

FROM ${FENCING_WEIGHTS_IMAGE} AS fencing_weights
FROM ${FENCING_BASE_IMAGE}

ENV PORT=5000 \
    PORT_HEALTH=5000 \
    FLASK_APP=app.py \
    DATABASE_URL=sqlite:////workspace/blog/blog.db \
    UPLOAD_DIR=/workspace/uploads \
    OUTPUT_2D=/workspace/unlabeled \
    OUTPUT_3D=/workspace/3d_outputs \
    WORKSPACE_TMP=/workspace/tmp \
    WORKSPACE_BLOG_DIR=/workspace/blog \
    MODEL_PATH=/app/best_touch_v346_coco17_bs10_multivid_val.pth \
    ATTACK_MODEL_PATH=/app/best_attack_3d_proximity.pth \
    ARM_ATTEMPT_BACKEND=onnx \
    ENABLE_OLLAMA=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/workspace/ollama \
    OPENAI_BASE_URL=http://127.0.0.1:11434/v1 \
    OPENAI_MODEL=llama3.2:3b

WORKDIR /app

# ---- Flask / web deps (light; rebuild with app code) ----
# onnxruntime-gpu 1.16.3 matches CUDA 11.8 in fencing-base (arm-attempt ONNX stack).
RUN pip install --no-cache-dir \
        flask flask-cors gunicorn \
        flask-sqlalchemy \
        authlib requests \
        markdown2 pyyaml python-dateutil \
        boto3 \
        "onnxruntime-gpu==1.16.3"

RUN curl -fsSL -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared \
    && chmod +x /usr/local/bin/cloudflared \
    && mkdir -p /var/log

# Local coaching LLM (OpenAI-compatible). Models live on /workspace at runtime.
RUN curl -fsSL https://ollama.com/install.sh | sh \
    && command -v ollama >/dev/null

RUN mkdir -p /workspace/blog /workspace/uploads /workspace/unlabeled \
    /workspace/3d_outputs /workspace/tmp /workspace/ollama \
    /app/checkpoints

COPY . /app
COPY static/ /app/static/
COPY templates/ /app/templates/
RUN if [ ! -f /app/vendor/odtrack/experiments/odtrack/baseline.yaml ]; then \
      mkdir -p /app/vendor && git clone --depth 1 https://github.com/GXNU-ZhongLab/ODTrack.git /app/vendor/odtrack; \
    fi \
 && python -c "import setup_odtrack as s; s._patch_vendor(); s.write_local_py()"
COPY --from=fencing_weights /weights/checkpoints/ /app/checkpoints/
COPY --from=fencing_weights /weights/touch/ /app/
COPY --from=fencing_weights /weights/attack/ /app/
COPY --from=fencing_weights /weights/odtrack/ /app/models/odtrack/

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5000

CMD ["/app/start.sh"]
