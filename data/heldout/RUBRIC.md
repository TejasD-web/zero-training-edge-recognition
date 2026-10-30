# Held-out set rules (written before collection)

Purpose: test the router policy chosen on the 17-image dev set. Run ONCE; no tuning afterwards.

Target mix: 20 images, ~4 easy / 8 medium / 8 hard. Images not in the dev set, different photos of different players.

Difficulty:
- easy: one main player, number clearly visible, roughly facing the camera, no blur
- medium: 2-3 players, OR partial occlusion, OR number at an angle
- hard: motion blur, heavy occlusion, 4+ players, rotated/partly hidden number, or unusual view (overhead, mid-air)

Labeling:
- jersey_number = every number a person can read with confidence; comma-separated if several players
- prediction counts as correct if it matches any listed number (same as dev set)
- labels and difficulty assigned by looking at the image only, before any model run

Pre-registered router policy (chosen on dev set, results/logprob_probe_20261030_031140.json):
- Primary: edge full-image read; answer on-device if joint digit probability >= 0.9, else escalate to Muse Spark (full image).
- Secondary (reported for comparison): agreement cascade from router/route.py (Qwen box agreement -> SAM 3.1 crop agreement -> Muse).
