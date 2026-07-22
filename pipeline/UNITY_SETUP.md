# Stage 3 / afternoon — Unity VMC receiver setup

Goal: a VRoid VRM avatar in Unity moving from `stage7_vmc_sender.py`. All packages MIT, run on macOS.

## 1. Project
- Unity **2022.3 LTS**, **URP** template (Toon Shader wants URP/HDRP).

## 2. Packages (install in this order)
1. **UniVRM** — https://github.com/vrm-c/UniVRM/releases → import the `.unitypackage` (VRM 0.x or VRM 1.0; match your VRoid export).
2. **uOSC** — Package Manager → Add from git URL: `https://github.com/hecomi/uOSC.git#upm`
3. **EVMC4U** (EasyVirtualMotionCaptureForUnity) — https://github.com/gpsnmeajp/EasyVirtualMotionCaptureForUnity → import `.unitypackage`.

## 3. Scene
1. Drag your **VRoid `.vrm`** into the scene (UniVRM imports it as a Humanoid prefab).
2. Add an empty GameObject → attach **`ExternalReceiver`** (EVMC4U). Set its **target** to the VRM avatar.
3. Confirm the uOSC **Server** listens on **UDP 39539** (EVMC4U wires this; verify the port).
4. Press Play, then run:  `python stage7_vmc_sender.py blended_for_unity.npy --fps 30 --loop`

## 4. The gotchas that will actually bite (from the pipeline graph)
- **VRoid ships in A-pose.** In the model's **Avatar → Configure**, enforce **T-pose** (Pose ▸ Enforce T-Pose) or the arms sit offset. This is the #1 first-run problem.
- **Foot-skating.** Enable **Foot IK** / an IK pass — Mecanim warns feet drift when proportions differ.
- **Coordinate handedness** (the sender's big caveat). If, on first run:
  - limbs are **mirrored** (left↔right) → in `stage7_vmc_sender.py` flip **x** instead of z (`flip_pos`: negate `p[0]`; `flip_quat`: `(w, x, -y, -z)`),
  - whole body faces **backwards** → also negate root z,
  - avatar is **upside-down / lying down** → swap which axis is up (our data is y-up; confirm your capture kept y-up).
  Tune these live — I couldn't verify them without the avatar.
- **Root height.** Our joints are hip-centered with the pelvis carrying world motion; if the avatar floats or sinks, offset the Hips y so feet touch the floor.

## 5. Sanity path
Before wiring the full blend, stream a **raw captured clip** (`visitor_clip.npy`) so you're debugging *retargeting* alone, not retargeting + blending at once. Use `view_clip.py` first to confirm the clip itself is clean.

## Data contract (all stages agree on this)
`(T, 22, 3)` float32 · HumanML3D 22-joint order · **y-up** · pelvis(0) carries global position · other joints relative.
Joint order & Unity HumanBodyBones map live in `hml_skeleton.py` (`JOINT_NAMES`, `UNITY_BONE`).
