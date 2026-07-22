"""
Stage 7 — stream a (T,22,3) HumanML3D joint clip to Unity over the VMC Protocol.

Python "Performer" -> UDP 127.0.0.1:39539 -> EVMC4U ExternalReceiver ("Marionette") -> VRM avatar.
Bone rotations come from hml_skeleton (swing-only), converted to Unity's left-handed frame.

Deps:  pip install python-osc numpy
Usage: python stage7_vmc_sender.py blended_for_unity.npy --fps 30 --loop

!!! COORDINATE CAVEAT: right-handed y-up (our data) -> Unity left-handed y-up requires a handedness
    flip. The flip below (negate z on position; (w,x,y,z)->(w,-x,-y,z) on rotation) is the standard
    z-flip, but the EXACT convention depends on how your capture axes ended up. I cannot verify this
    against a live avatar from here — expect to tune `flip_pos` / `flip_quat` once you see the avatar
    move (limbs mirrored -> flip x instead; whole body backwards -> also negate root z).
"""
import argparse
import time
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from hml_skeleton import (JOINT_NAMES, UNITY_BONE, rest_offsets,
                          joints_to_global_quats, global_to_local)


def flip_pos(p):
    return np.array([p[0], p[1], -p[2]])          # z-flip: RH y-up -> Unity LH y-up

def flip_quat(q):                                  # q = [w,x,y,z]
    return np.array([q[0], -q[1], -q[2], q[3]])


def precompute(seq):
    """joints -> per-frame (unity local positions, unity local quats) for all 22 bones."""
    off = rest_offsets(seq)
    gq = joints_to_global_quats(seq, off)
    lq = global_to_local(gq)                        # local rotations, parent-relative
    T = seq.shape[0]
    # local bone position = the (fixed) rest offset, flipped to Unity; root uses per-frame world pos
    upos = np.zeros((T, 22, 3)); uquat = np.zeros((T, 22, 4))
    for t in range(T):
        for j in range(22):
            uquat[t, j] = flip_quat(lq[t, j])
            upos[t, j] = flip_pos(seq[t, 0]) if j == 0 else flip_pos(off[j])
    return upos, uquat


def send_frame(client, upos, uquat):
    # root
    p, q = upos[0], uquat[0]                         # joint 0 = pelvis = Hips = root
    client.send_message("/VMC/Ext/Root/Pos",
                        ["root", float(p[0]), float(p[1]), float(p[2]),
                         float(q[1]), float(q[2]), float(q[3]), float(q[0])])   # qx,qy,qz,qw
    # bones
    for j, name in enumerate(JOINT_NAMES):
        bone = UNITY_BONE[name]
        p, q = upos[j], uquat[j]
        client.send_message("/VMC/Ext/Bone/Pos",
                            [bone, float(p[0]), float(p[1]), float(p[2]),
                             float(q[1]), float(q[2]), float(q[3]), float(q[0])])
    client.send_message("/VMC/Ext/OK", [1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()

    seq = np.load(a.clip).astype(np.float32)
    assert seq.shape[1:] == (22, 3), f"expected (T,22,3), got {seq.shape}"
    upos, uquat = precompute(seq)
    client = SimpleUDPClient(a.host, a.port)
    dt, T = 1.0 / a.fps, len(seq)
    print(f"[vmc] streaming {T} frames @ {a.fps:.0f}fps -> {a.host}:{a.port}  (Ctrl-C to stop)")
    try:
        while True:
            for t in range(T):
                send_frame(client, upos[t], uquat[t])
                time.sleep(dt)
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    print("[vmc] done")


if __name__ == "__main__":
    main()
