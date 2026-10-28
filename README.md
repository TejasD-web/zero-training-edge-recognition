# Zero-Training Fine-Grained Visual Recognition on Edge

Composing foundation models across edge and cloud tiers to read fine-grained visual details, such as jersey numbers, without training a single weight. Each model does the step it is best at, and a training-free confidence signal decides when the edge asks the cloud for help.

**Meta Global AI Developer Hackathon** | Oct 26 – Nov 4, 2026 | Push the Frontier

Jersey numbers are the benchmark, not the product. The pattern applies to any small detail in a wide frame: serial numbers, labels, signage, part markings.

## Architecture

1. **Edge read (Jetson):** Qwen2.5-VL 3B (4-bit, llama.cpp) reads the full frame.
2. **Edge cross-check:** a second view of the same frame (a segmentation or grounding crop) is read by the same VLM.
3. **Confidence routing:** if the views agree, the edge answer is returned. If they disagree, the frame escalates.
4. **Cloud tier (Meta Model API):** SAM 3.1 localizes the detail by concept ("jersey number"), and the crop is read by the edge VLM or Muse Spark.

## Hardware

- Jetson Orin Nano Super 8 GB (JetPack 6.2.1 / L4T 36.4.7), 25 W power envelope
- SAM 2.1 hiera-tiny (PyTorch) and Qwen2.5-VL-3B Q4_K_M + Q8 mmproj (llama.cpp server)

## Findings so far

Benchmark: 17 hand-labeled images with easy, medium and hard tiers, scored by exact match (any listed number counts for multi-player frames). Every image is read at the same 256-token image budget, so conditions differ only in what is in the frame. At n=17, a 1–2 image difference (6–12 points) is within noise.

| Condition | Accuracy | Easy | Medium | Hard |
|---|---|---|---|---|
| **Full image** | **71% (12/17)** | 6/7 | 2/5 | 4/5 |
| Qwen-grounded box crop | 41% (7/17) | 3/7 | 2/5 | 2/4 |
| Qwen box → SAM 2.1 refined crop | 29% (5/17) | 2/7 | 2/5 | 1/4 |
| Center-point SAM 2.1 crop | 29% (5/17) | 1/7 | 3/5 | 1/5 |

**1. A focused prompt matters more than cropping.** An earlier harness with a multi-field extraction prompt scored 0% on jersey numbers. A single-question prompt on the full frame scores 71% with the same model and weights.

**2. On the edge model, the weak step is localization, not reading.** Cropping only helps when the crop contains the number. Qwen 3B's grounding boxes were checked by hand: one landed exactly on the number (read correctly), and the others landed on a sponsor patch, the crowd, or a different player. When the box is right, the crop reads correctly. That is why the cloud tier's job is concept-level localization (SAM 3.1), not just "a bigger model".

**3. View agreement is a training-free confidence signal.** When the full-frame answer matches at least one crop-based answer, it is correct 6 of 7 times (86%). When it disagrees with all of them, it is correct 6 of 10 times. Agreement decides what stays on the edge and what escalates.

**4. Views are complementary.** At least one condition is correct on 13 of 17 frames, so better routing has room to beat the best single view.

## Edge engineering notes (Jetson Orin Nano)

- **GPU memory is a 960 MB contiguous pool (CMA), not 8 GB.** `cudaMalloc` on this L4T release draws from CMA, which cannot be raised past ~960 MB on this board. Larger values reserve nothing (CmaTotal=0).
- **CMA fragments.** The VLM's 433 MB vision-encoder buffer fails to allocate after SAM has run on the GPU, even with ~700 MB nominally free.
- **CMA leaks on process exit (L4T 36.4.7).** Each llama-server exit leaks hundreds of MB, so models must stay resident in long-lived processes and benchmarks start from a fresh boot.
- **What works:** start the VLM first on clean CMA, keep 10 LLM layers on GPU (`-ngl 10`), set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, run SAM in a single batch process, and warm up and retry SAM's first allocation. All 99 layers on GPU starves the vision buffer.
- **Grounding needs resolution:** Qwen-VL grounding needs about 1000 image tokens (`--image-max-tokens 1024`, `-ub 1024` so the image is processed in one batch). Reading is done at 256 tokens.
- Latency: ~2.4 s per VLM read, ~8–9 s per grounding call, ~0.4 s per SAM 2.1 segmentation on GPU (~15 s on CPU).

## Reproduce

```bash
source activate.sh
sudo reboot   # fresh CMA before each run
python benchmark/run_benchmark.py --baseline --grounding
```

Results are written to `results/benchmark_<timestamp>.json`. Images are not distributed with this repo.
