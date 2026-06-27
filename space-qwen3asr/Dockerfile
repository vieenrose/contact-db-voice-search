# Qwen3-ASR-0.6B-Agent voice attendant — transformers, CPU, in-process (no llama.cpp).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# qwen-asr carries the qwen3_asr modeling code (vanilla transformers lacks it). --no-deps skips its
# PyAV/video dependency; then neutralize the av-pulling inference import so the model loads bare.
RUN pip install --no-cache-dir --no-deps qwen-asr qwen-omni-utils && \
    f=$(find / -path '*/qwen_asr/__init__.py' 2>/dev/null | head -1) && \
    sed -i 's/^from \.inference/# from .inference/' "$f" && echo "neutralized $f"

COPY . /app
ENV TORCH_THREADS=2 HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
