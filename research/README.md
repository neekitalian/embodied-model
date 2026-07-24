# Research Portal assets (Round 2 perceptual study)

The Research Portal (`/research`, source `research.html`) reads `stimuli.json` and serves videos from
these folders. Everything here is static, so it works locally (`npm run dev`) and on Vercel.

## Layout

```
research/
  stimuli.json          manifest: 48 stimuli (4 actions x 3 genres x 4 allocation levels) + metadata
  sources/              the 4 source action videos (clean names) + rendered original skeletons
  stimuli/              the 48 translated stimulus videos (+ their motion JSON)
```

## Generating the stimuli (on your Mac)

Put the 4 source videos in `~/Desktop/Videos/` (waving.mp4, shaking hands.mp4, walk_riku_cam1.mp4,
sorting.mp4), then:

```
cd pipeline
pip install mediapipe opencv-python matplotlib
brew install ffmpeg          # for mp4 output; without it, renders fall back to .gif
python make_stimuli.py --videos ~/Desktop/Videos
```

This copies the sources under clean names, extracts skeletons, grafts every genre x allocation level at a
constant allocation (0.20 / 0.40 / 0.60 / 0.80), renders each to mp4, and rewrites `stimuli.json`.

Then commit `research/sources`, `research/stimuli`, and `research/stimuli.json` so Vercel serves them.

## Data

Participant responses autosave to the browser's localStorage and are exported as CSV from the Admin tab
(raw = one row per participant x stimulus; summary = grouped by action x genre x allocation level).
There is no server; nothing is uploaded.
