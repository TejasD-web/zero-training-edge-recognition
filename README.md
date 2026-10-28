# Zero-Training Fine-Grained Visual Recognition on Edge

Composing Meta's foundation models across edge and cloud tiers to read fine-grained visual details that defeat full-frame inference — without training a single weight.

**Meta Global AI Developer Hackathon** | Oct 26 – Nov 4, 2026 | Push the Frontier

## Architecture

1. Full frame → SAM 2.1 (edge, local) → player segmentation
2. Crop per detection → edge VLM (Qwen2.5-VL 4-bit, llama.cpp)
3. Low confidence → SAM 3.1 + Muse Spark (Meta Model API, cloud tier)

## Hardware Constraint

- Jetson Orin Nano Super 8GB unified memory
- 25W power envelope

## Key Insight

A 4-bit VLM scores 0% on jersey numbers in full frames — the detail is too small after vision encoder resize. Segmentation-based cropping fixes this without retraining, and the pattern generalizes to any fine-grained detail in a wide frame.
