# SAM 3 on Jetson Orin Nano (8GB), random weights (pending HF access approval)

Env: Python 3.10 venv, torch 2.11.0 (CUDA 12.6) from pypi.jetson-ai-lab.io/jp6/cu126,
facebookresearch/sam3 installed with `pip install -e .`, plus:
  pip install einops pycocotools psutil
  pip install --no-deps triton --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
Pass bpe_path explicitly (editable install breaks the default lookup).
Must run under torch.autocast("cuda", dtype=torch.bfloat16).

Results (random weights, img_001, prompt "jersey number"):
- 841M params, 3,410 MB GPU (fp32 weights), build 17s
- 1.22-1.36s per image after warm-up (4.47s first run), peak GPU ~4.0 GB
- NvMap error 12 retries seen: too tight to co-run with llama-server at -ngl 99
- Full model.to(bf16): 1,818 MB but breaks the decoder (fp32 section) and
  destroys complex RoPE buffers. Needs a selective (vision trunk only) conversion,
  verifiable only with real weights.
