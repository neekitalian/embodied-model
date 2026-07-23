# Pipeline

Stage 1–7 scaffolds for the installation. Each piece was unit-tested in isolation; the
end-to-end run happens locally (needs a webcam, your `semantic_spectrum`, and AIST++).

## Files
| File | Stage | Status |
|---|---|---|
| `stage12_capture_lift.py` | 1+2 · webcam → MediaPipe → HumanML3D 22 | built, `test_lift.py` 12/12 |
| `hml_skeleton.py` | shared skeleton + joints↔rotation math | built, FK round-trip 0.03% |
| `view_clip.py` | 3D stick-figure viewer for any `(T,22,3)` clip | built |
| `stage7_vmc_sender.py` | 7 · joints → VMC → Unity | built, OSC loopback ok |
| `run_mvp.py` | end-to-end orchestrator (calls your `semantic_spectrum`) | scaffold, 3 ADAPT points |
| `genre_style.py` | identity-preserving per-zone genre style transfer (rhythm-align + zone-α) | proxy runs, `test_genre_style.py` 7/7; 2 ADAPT points for the real encoder |
| `video_to_reference.py` | dance video (AIST DB) -> HumanML3D-22 reference JSON (MediaPipe) | reuses stage12 |
| `aist_to_reference.py` | AIST++ SMPL motion -> HumanML3D-22 reference JSON | scaffold, 1 ADAPT (SMPL forward) |
| `run_local.py` | end-to-end: visitor + genre reference -> transfer -> VMC->Unity | `test_run_local.py` 6/6 |

| `UNITY_SETUP.md` | Stage 3 / afternoon Unity receiver setup | notes |

## Install
```
pip install numpy mediapipe opencv-python python-osc matplotlib
```

## Run order (local)
1. `python stage12_capture_lift.py --seconds 6 --out visitor_clip.npy`
2. `python view_clip.py visitor_clip.npy`            # confirm capture is clean
3. wire the 3 ADAPT points in `run_mvp.py` to your real `semantic_spectrum` signatures
4. `python run_mvp.py --visitor visitor_clip.npy --library target_library/`
5. `python stage7_vmc_sender.py blended_for_unity.npy --fps 30 --loop`  # + Unity per UNITY_SETUP.md

## Tests (no camera/GPU needed)
```
python hml_skeleton.py   # FK round-trip
python test_lift.py      # capture+lift logic
python test_tools.py     # viewer render + VMC OSC loopback
```


## Make genre reference clips (for genre_style.py)
AIST DB gives **videos** (non-commercial → gallery/research track only). Turn them into references:
```
# from a dance video (Jazz=gJS, Ballet=gJB, Hip-hop=gLH/gMH):
python video_to_reference.py --video gJB_sBM_c01_d05_mJB0_ch01.mp4 --out ballet_ref.json --seconds 6
# or batch a folder:
python video_to_reference.py --glob "~/Downloads/aist/gLH_*.mp4" --out-dir refs/
# then transfer the visitor's identity into that genre:
python genre_style.py --visitor visitor_clip.json --reference ballet_ref.json --alpha 0.5 --out styled.npy
```

## One-command local run (capture -> genre -> Unity)
```
# style + preview:
python run_local.py --visitor visitor_clip.json --genre ballet --refs-dir refs/ --alpha 0.5 --out styled.npy
python view_clip.py styled.npy
# style + stream live to Unity (EVMC4U on 39539):
python run_local.py --visitor visitor_clip.json --genre hip-hop --refs-dir refs/ --stream --loop
```
(Or just upload a genre video in the /portal page and download its JSON — same result, no code.)

## Data contract
`(T, 22, 3)` float32 · HumanML3D 22-joint order · y-up · pelvis(0) carries global position.
