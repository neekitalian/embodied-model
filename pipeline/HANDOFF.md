# Handoff prompt — run the Unnoticed Dance local pipeline

Paste the block below into a Claude Code / Cowork session **running on the Mac**, started inside
`~/embodied-model/pipeline`. That session can see the files, the webcam, the AIST videos, and Unity —
things the cloud session cannot. Everything it needs is already in this repo and tested in isolation.

---

You are working on my "Unnoticed Dance" installation, in `~/embodied-model/pipeline` on my Mac
(macOS, zsh, conda `base`, Python 3.12). Deps are installed: mediapipe, opencv-python, numpy, python-osc.
**First: `git pull` in `~/embodied-model` — it has recent fixes.**

## Goal
Get the local end-to-end running: build genre **reference** clips from AIST videos, capture a **visitor**
clip, then run `run_local.py` to produce an identity-preserving **genre-styled** motion, preview it, and
(if Unity is set up) stream it to a VRM avatar over VMC. Then help me tune it.

## What already exists here (all unit-tested; don't rewrite — reuse)
- `video_to_reference.py` — dance video → HumanML3D-22 reference JSON (MediaPipe pose).
- `genre_style.py` — identity-preserving per-zone genre transfer (content=visitor, style=reference).
  Two `ADAPT` points to later plug in my real `semantic_spectrum` encoder; a documented proxy runs now.
- `run_local.py` — one command: visitor + genre reference → transfer → save/preview/stream (VMC 39539).
- `view_clip.py` — 3D skeleton viewer. `stage12_capture_lift.py` — webcam → HumanML3D-22.
- `stage7_vmc_sender.py` — joints → VMC → Unity. `hml_skeleton.py` — skeleton + rotation math.
- Tests: `python hml_skeleton.py`, `test_lift.py`, `test_genre_style.py`, `test_run_local.py` (all pass).

## Hard-won facts about my data / environment
- AIST **videos** are in `~/Downloads/lite/` and `~/Downloads/dance_genre_estimation/` (flat, no subfolders).
  They are non-commercial → research/gallery track only; do not commit them to the repo.
- Genres present include **`gJB`** (Ballet Jazz → *ballet*), **`gMH`** (Middle Hip-hop → *hip-hop*), plus
  `gBR`, `gHO`, `gKR`. **`gJS` (Street Jazz → jazz) may be absent — check.**
- MediaPipe reads **one** body, so use **single-dancer, camera `c01`** clips (avoid `sGR` group clips with
  `d04_d05_d06`). Known-good single-dancer clips: `gJB_sFM_c01_d07_mJB3_ch04.mp4`,
  `gMH_sBM_c01_d24_mMH3_ch07.mp4`.
- zsh gotcha: **do not put `# comments` on the same line as a command** (they become args). One command per line.

## Steps
1. `git pull`; run the tests above to confirm the toolchain works.
2. Check which genres you actually have:
   `ls ~/Downloads/lite ~/Downloads/dance_genre_estimation | grep -oE '^g[A-Z]{2}' | sort -u`
3. Build references from single-dancer c01 clips (name the outputs by genre so `--genre` finds them):
   `python video_to_reference.py --video ~/Downloads/lite/gJB_sFM_c01_d07_mJB3_ch04.mp4 --out refs/ballet.json --seconds 6`
   `python video_to_reference.py --video ~/Downloads/lite/gMH_sBM_c01_d24_mMH3_ch07.mp4 --out refs/hip-hop.json --seconds 6`
   (If `gJS` exists, make `refs/jazz.json` too; otherwise pick a third available genre.)
4. **Preview each reference** to confirm the pose is clean (bad camera angle → junk skeleton):
   `python view_clip.py refs/ballet.json`
5. Capture a visitor clip: `python stage12_capture_lift.py --seconds 6 --out visitor_clip.npy`
   (or export `visitor_clip.json` from the /portal page). Preview it with `view_clip.py`.
6. Run the transfer and preview:
   `python run_local.py --visitor visitor_clip.npy --genre ballet --refs-dir refs/ --alpha 0.5 --out styled.npy`
   `python view_clip.py styled.npy`
7. Tune `--alpha` (0 = pure me, 1 = full genre). Report what each genre looks like.
8. Only if I ask / Unity is ready: `... --stream --loop` and follow `UNITY_SETUP.md`. Expect to tune VMC
   coordinate handedness live (see the CAVEAT in `stage7_vmc_sender.py`).

## Likely issues to watch for and fix
- MediaPipe world-landmark axis orientation vs my y-up/hip-centered convention (check `view_clip` looks upright).
- AIST reference vs visitor at different fps — `genre_style.rhythm_align` handles beat alignment; if timing looks
  off, check the foot-contact phase on noisy references.
- A reference with a non-frontal camera gives a poor skeleton — swap to a `c01` single-dancer clip.
- The transfer is a geometric proxy until my `semantic_spectrum` encoder is wired at the two `ADAPT` points in
  `genre_style.py`; don't expect deep vocabulary yet — expect rhythm + posture + expansiveness shifts.

## Done when
`refs/` has clean per-genre references, `run_local.py` produces a styled clip that is recognizably me but
carries the genre, and `view_clip.py` confirms it. Streaming to Unity is a bonus.
