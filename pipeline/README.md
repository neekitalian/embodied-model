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

## Data contract
`(T, 22, 3)` float32 · HumanML3D 22-joint order · y-up · pelvis(0) carries global position.
