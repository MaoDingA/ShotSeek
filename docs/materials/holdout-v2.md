# Holdout v2 material

## Purpose

`holdout-v2` is an independent generalization check created after the
Development v1 morphology normalization was frozen. Its results are not used
to modify the v1 retrieval or verification rules.

## Source

- Film: *Caminandes 2: Gran Dillama*
- Publisher: Blender Foundation / Blender Open Movies
- Official download:
  <https://download.blender.org/demo/movies/caminandes_gran_dillama.mp4.zip>
- Official Blender Open Movies channel:
  <https://video.blender.org/c/blender_open_movies/videos>
- Material SHA-256:
  `468e6743c674689a728726bbe4bb4b2a65bd8702a89f021af26a8bb4d450eebd`
- Duration: `146.041667 s` source; `146.000 s` normalized video timeline
- Source media: H.264, 1920x1080, 24 fps CFR, AAC stereo

The local media itself is ignored by Git. The hash and official source URL make
the benchmark material reproducible without redistributing a binary in this
repository.

## Runtime evidence

- Runtime mode: `live`
- Final state: `READY`
- StepFun visual artifact: `LIVE`, `step-3.7-flash`
- StepFun ASR artifact: `LIVE`, `stepaudio-2.5-asr`
- Chunk duration: `60 s`
- Visual workers: `3`
- Scenes: `18`
- End-to-end elapsed time: `144.77 s`
- Realtime factor: `0.992`
- Retry count: `0`
- Registry integrity: `ok`

The first attempt exposed a production issue: AAC padding extended the MP4
container duration by 48 ms beyond the video stream. The strict CFR probe
incorrectly compared the frame-derived duration with the whole container.
Commit `0486188` changed the contract to validate the video stream duration
while keeping the frame-count consistency gate. A regression test preserves
this behavior.

## Annotation protocol

1. Run the frozen production pipeline to `READY`.
2. Human-review the generated scene summaries and a time-sampled contact sheet.
3. Write positive, ordinal, temporal and hard-negative queries.
4. Freeze and hash the JSONL before the first benchmark execution.
5. Run the frozen rule planner and verifier without adapting them to v2.

This is a single-reviewer benchmark, not a statistically powered or
double-annotated scientific dataset. It is evidence of reproducible product
behavior on previously unseen material, not a claim of universal accuracy.
