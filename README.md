# Edge-Cloud Jersey Number Recognition

SAM 2.1 segmentation + VLM inference pipeline for jersey number reading on Jetson Orin Nano (8GB).

Meta Global AI Developer Hackathon — Oct 26 – Nov 4, 2026

## Architecture
1. Full frame → SAM 2.1 → player segmentation
2. Crop per player → edge VLM (Qwen2.5-VL 4-bit, llama.cpp)
3. Low confidence → route to Meta Model API

## Hardware
- Jetson Orin Nano Super 8GB
- 25W power budget
