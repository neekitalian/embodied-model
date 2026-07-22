"""
Shared HumanML3D-22 skeleton + joints<->rotations math.  numpy only.

Used by view_clip.py (bones for drawing) and stage7_vmc_sender.py (bone rotations for VMC).
Rotations are quaternions [w,x,y,z] throughout — NO Euler anywhere, to avoid rotation-order bugs.
The joints->rotations path is SWING-ONLY (no bone twist): each bone is oriented by a shortest-arc
"look-at child" rotation. Good for an MVP avatar; loses forearm/upper-arm twist. FK round-trip
below quantifies the (small) position error this introduces.
"""
import numpy as np

JOINT_NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
               "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
               "neck", "left_collar", "right_collar", "head", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# SMPL / HumanML3D kinematic tree
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# bone segments (for line drawing) = every (parent, child)
BONES = [(PARENTS[j], j) for j in range(1, 22)]

# the child that defines each joint's forward axis (for look-at orientation); leaves -> None
PRIMARY_CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12,
                 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21,
                 10: None, 11: None, 15: None, 20: None, 21: None}

# HumanML3D joint -> Unity HumanBodyBones name (for VMC)
UNITY_BONE = {
    "pelvis": "Hips", "left_hip": "LeftUpperLeg", "right_hip": "RightUpperLeg",
    "spine1": "Spine", "left_knee": "LeftLowerLeg", "right_knee": "RightLowerLeg",
    "spine2": "Chest", "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "spine3": "UpperChest", "left_foot": "LeftToes", "right_foot": "RightToes",
    "neck": "Neck", "left_collar": "LeftShoulder", "right_collar": "RightShoulder",
    "head": "Head", "left_shoulder": "LeftUpperArm", "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm", "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}

# ---------------- quaternion utils ([w,x,y,z]) ----------------
def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0, 0, 0])

def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_rotate(q, v):
    qv = np.array([0., v[0], v[1], v[2]])
    return q_mul(q_mul(q, qv), q_conj(q))[1:]

def q_from_two_vectors(a, b):
    """Shortest-arc rotation taking unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 1 - 1e-8:
        return np.array([1., 0, 0, 0])
    if d < -1 + 1e-8:                       # antiparallel: 180° about any perpendicular axis
        axis = np.cross(a, [1., 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0., 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0., *axis])
    axis = np.cross(a, b)
    return q_normalize(np.array([1 + d, *axis]))

# ---------------- skeleton math ----------------
def rest_offsets(seq):
    """
    Per-bone rest offset (parent->child), fixed across the clip.
    Length = mean bone LENGTH (magnitude); direction = mean direction.
    NB: averaging the raw vectors would shrink the length (directions cancel), rebuilding bones
    too short in FK. Magnitude is what FK preserves (swing rotation keeps |offset|), so we must
    carry the true length; the rest direction only sets the reference the swing rotates from.
    """
    seq = np.asarray(seq)
    off = np.zeros((22, 3), dtype=np.float64)
    for j in range(1, 22):
        vecs = seq[:, j, :] - seq[:, PARENTS[j], :]        # (T,3)
        mean_len = float(np.mean(np.linalg.norm(vecs, axis=1)))
        mean_dir = np.mean(vecs, axis=0)
        n = np.linalg.norm(mean_dir)
        off[j] = (mean_dir / n * mean_len) if n > 1e-9 else np.array([0., mean_len, 0.])
    return off

def joints_to_global_quats(seq, offsets):
    """Per-joint GLOBAL orientation from a look-at-primary-child alignment (swing only)."""
    T = seq.shape[0]
    gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            c = PRIMARY_CHILD[j]
            if c is None:
                gq[t, j] = gq[t, PARENTS[j]]        # leaf inherits parent orientation
                continue
            rest_dir = offsets[c]
            cur_dir = seq[t, c] - seq[t, j]
            if np.linalg.norm(cur_dir) < 1e-8:
                gq[t, j] = gq[t, PARENTS[j]] if PARENTS[j] >= 0 else np.array([1., 0, 0, 0])
                continue
            gq[t, j] = q_from_two_vectors(rest_dir, cur_dir)
    return gq

def global_to_local(gq):
    """Convert global orientations to parent-relative (local) — what BVH/VMC want."""
    T = gq.shape[0]
    lq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            p = PARENTS[j]
            lq[t, j] = gq[t, j] if p < 0 else q_mul(q_conj(gq[t, p]), gq[t, j])
    return lq

def fk(gq, offsets, root_pos):
    """Forward kinematics from GLOBAL quats -> joint positions (for round-trip validation)."""
    T = gq.shape[0]
    pos = np.zeros((T, 22, 3))
    for t in range(T):
        pos[t, 0] = root_pos[t]
        for j in range(1, 22):
            p = PARENTS[j]
            pos[t, j] = pos[t, p] + q_rotate(gq[t, p], offsets[j])
    return pos


T_POSE = {  # realistic parent->child offsets, y-up, meters (a human standing, arms out)
    1: (0.09, 0, 0), 2: (-0.09, 0, 0), 3: (0, 0.10, 0),
    4: (0, -0.40, 0), 5: (0, -0.40, 0), 6: (0, 0.12, 0),
    7: (0, -0.40, 0), 8: (0, -0.40, 0), 9: (0, 0.12, 0),
    10: (0, -0.05, 0.12), 11: (0, -0.05, 0.12), 12: (0, 0.10, 0),
    13: (0.05, 0.05, 0), 14: (-0.05, 0.05, 0), 15: (0, 0.15, 0),
    16: (0.12, 0, 0), 17: (-0.12, 0, 0), 18: (0.26, 0, 0), 19: (-0.26, 0, 0),
    20: (0.25, 0, 0), 21: (-0.25, 0, 0),
}

if __name__ == "__main__":
    # ---- FK round-trip self-test on a REALISTIC humanoid doing moderate swing-only motion ----
    T = 60
    off = np.zeros((22, 3))
    for j, v in T_POSE.items():
        off[j] = v

    truth_gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
    for t in range(T):
        for j in range(22):
            c = PRIMARY_CHILD[j]
            if c is None:
                truth_gq[t, j] = truth_gq[t, PARENTS[j]]
                continue
            # gentle target wobble around the rest direction -> pure swing, no degeneracies
            base = np.array(off[c], float)
            d = base + 0.20 * np.linalg.norm(base) * np.array(
                [np.sin(0.2*t + j), np.cos(0.15*t + j), np.sin(0.1*t + 2*j)])
            truth_gq[t, j] = q_from_two_vectors(base, d)
    root = np.stack([np.array([0.002*t, 0.01*np.sin(0.1*t), 0]) for t in range(T)])
    seq = fk(truth_gq, off, root)

    off2 = rest_offsets(seq)
    gq = joints_to_global_quats(seq, off2)
    recon = fk(gq, off2, seq[:, 0, :])
    span = np.mean(np.linalg.norm(seq - seq.mean(axis=1, keepdims=True), axis=-1))
    err = np.linalg.norm(recon - seq, axis=-1).mean()
    print(f"[round-trip] mean joint error = {err:.5f} m  ({100*err/span:.2f}% of body span)")
    print(f"[shapes] global {gq.shape} -> local {global_to_local(gq).shape}")
    print("PASS ✓" if err < 0.01 else "CHECK plumbing")
    print("Note: real capture adds bone twist + slight length wobble -> expect a low-single-digit %"
          " residual. That's the swing-only approximation, fine for an MVP avatar.")
