"""SAM 3 on Jetson: memory + speed with RANDOM weights (outputs meaningless). No HF download."""
import time, torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def mem_avail_mb():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) // 1024

print(f"MemAvailable before: {mem_avail_mb()} MB")
t0 = time.time()
model = build_sam3_image_model(load_from_HF=False, checkpoint_path=None, bpe_path="/mnt/nvme/project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
torch.cuda.synchronize()
n_params = sum(p.numel() for p in model.parameters())
print(f"Built in {time.time() - t0:.1f}s | params {n_params / 1e6:.0f}M | "
      f"GPU allocated {torch.cuda.memory_allocated() / 2**20:.0f} MB | MemAvailable {mem_avail_mb()} MB")

processor = Sam3Processor(model)
img = Image.open("/mnt/nvme/project/zero-training-edge-recognition/data/images/img_001.jpg").convert("RGB")

for label, ctx in [("bf16 autocast", torch.autocast("cuda", dtype=torch.bfloat16))]:
    with torch.no_grad(), ctx:
        for i in range(3):                      # first run includes warm-up
            torch.cuda.synchronize(); t = time.time()
            state = processor.set_image(img)
            out = processor.set_text_prompt(state=state, prompt="jersey number")
            torch.cuda.synchronize()
            print(f"[{label}] run {i + 1}: {time.time() - t:.2f}s | peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MB "
                  f"| MemAvailable {mem_avail_mb()} MB | boxes {len(out['boxes'])}")
