"""
Genre style transfer - the REAL version of what the browser previews.

Content = the visitor's motion (identity).  Style = a genre REFERENCE clip (AIST++ / Mixamo).
Output = the visitor's movement re-expressed in the genre, with identity preserved.

This is the local-pipeline counterpart to portal.html's heuristic editions. Unlike the browser
(which can only reshape the visitor's own trajectory), this injects genre vocabulary + rhythm
from actual reference motion - the only way to truly capture a genre.

Design (mirrors Motion Puzzle / your zone-alpha):
  1. Rhythm-align the visitor to the genre reference's beat (foot-contact phase -> time-warp).
  2. Per ANATOMICAL ZONE, blend the visitor's *content* with the reference's *style* by alpha.
     Style here = the reference's per-zone dynamics (local velocity texture / expansiveness),
     transplanted onto the visitor's pose via feature-space interpolation.
  3. Reconstruct joints, 1e-filter, hand off to VMC/Unity.

!!! Two ADAPT points marked below plug into your real semantic_spectrum encoder. Without it,
    this file runs a transparent geometric proxy (documented), so the pipeline is end-to-end today
    and upgrades in place when the encoder lands.

Deps: numpy.  Reuses pipeline/hml_skeleton.py for the zone map + skeleton.
"""
import argparse, json, os
import numpy as np
from hml_skeleton import JOINT_NAMES, IDX, PARENTS

# 8 anatomical zones on the HumanML3D-22 skeleton (the "zones" your semantic_spectrum decomposes).
ZONES = {
    "root":       [IDX["pelvis"], IDX["spine1"]],
    "spine":      [IDX["spine2"], IDX["spine3"], IDX["neck"], IDX["head"]],
    "left_arm":   [IDX["left_collar"], IDX["left_shoulder"], IDX["left_elbow"], IDX["left_wrist"]],
    "right_arm":  [IDX["right_collar"], IDX["right_shoulder"], IDX["right_elbow"], IDX["right_wrist"]],
    "left_leg":   [IDX["left_hip"], IDX["left_knee"], IDX["left_ankle"], IDX["left_foot"]],
    "right_leg":  [IDX["right_hip"], IDX["right_knee"], IDX["right_ankle"], IDX["right_foot"]],
}

# per-zone identity<->style weight. Lower alpha = more identity (protect it); higher = more genre.
# Legs/arms carry most genre character; keep the root near identity so the body stays "yours".
DEFAULT_ALPHA = {"root":0.10, "spine":0.30, "left_arm":0.55, "right_arm":0.55, "left_leg":0.50, "right_leg":0.50}


def load_clip(path):
    if path.endswith(".json"):
        d = json.load(open(path)); return np.asarray(d["joints"], dtype=np.float32)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    raise ValueError(f"unsupported clip: {path}")


def foot_contact_phase(clip):
    """Rough beat phase from vertical foot-speed minima (steps land on beats). Returns per-frame phase 0..1."""
    lf, rf = clip[:, IDX["left_foot"], :], clip[:, IDX["right_foot"], :]
    speed = np.linalg.norm(np.diff(lf, axis=0), axis=1) + np.linalg.norm(np.diff(rf, axis=0), axis=1)
    speed = np.concatenate([speed[:1], speed])
    # contacts = local minima of foot speed
    contacts = [t for t in range(1, len(speed)-1) if speed[t] <= speed[t-1] and speed[t] < speed[t+1]]
    if len(contacts) < 2:
        return np.linspace(0, 1, len(clip), endpoint=False) % 1.0
    phase = np.zeros(len(clip))
    for a, b in zip(contacts[:-1], contacts[1:]):
        phase[a:b] = np.linspace(0, 1, b-a, endpoint=False)
    return phase


def rhythm_align(visitor, reference):
    """Time-warp the visitor onto the reference's beat phase (DTW-lite by phase matching)."""
    vph, rph = foot_contact_phase(visitor), foot_contact_phase(reference)
    T = len(visitor)
    # for each visitor frame, keep its own pose but resample so its phase tracks the reference's phase timeline
    out_idx = np.clip(np.round(np.interp(
        np.linspace(rph[0], rph[0]+ (rph[-1]-rph[0]), T),  # target phase timeline from reference
        np.linspace(vph[0], vph[-1], T), np.arange(T))).astype(int), 0, T-1)
    return visitor[out_idx]


def zone_style_features(clip, joints):
    """ADAPT #1: replace with your semantic_spectrum per-zone encoder.
    Proxy = per-zone local dynamics: mean speed, expansiveness (spread about zone centroid), vertical bias."""
    z = clip[:, joints, :]
    vel = np.linalg.norm(np.diff(z, axis=0), axis=-1).mean() if len(z) > 1 else 0.0
    centroid = z.mean(axis=1, keepdims=True)
    spread = np.linalg.norm(z - centroid, axis=-1).mean()
    vbias = float(np.mean(z[..., 1] - centroid[..., 1]))
    return np.array([vel, spread, vbias], dtype=np.float32)


def transfer(visitor, reference, alpha=None):
    """Identity-preserving per-zone style transfer. visitor content + reference style -> styled clip.
    The overall style strength (mean alpha) scales the WHOLE effect - rhythm, zone reshaping, and
    smoothing - so alpha=0 returns the visitor untouched (identity), and alpha=1 is full genre."""
    alpha = {**DEFAULT_ALPHA, **(alpha or {})}
    # keep the FULL visitor length: the reference only supplies style statistics + a beat phase,
    # so a 30s capture styled by a 6s reference stays 30s (mirrors the portal.html fix)
    visitor = np.asarray(visitor, dtype=np.float32)
    T = len(visitor)
    strength = float(np.mean([alpha.get(z, 0.4) for z in ZONES]))

    # rhythm: warp visitor to the reference beat, blended in by overall strength (0 -> no warp)
    aligned = rhythm_align(visitor, reference)
    out = strength * aligned + (1.0 - strength) * visitor

    for zone, joints in ZONES.items():
        a = alpha.get(zone, 0.4)
        vf = zone_style_features(visitor, joints)
        rf = zone_style_features(reference, joints)
        # ADAPT #2: with the real encoder this is a decode of a blended latent. Proxy transplants the
        # reference zone's dynamics (expansiveness / vertical-bias) onto the visitor's motion by alpha.
        v = out[:, joints, :]
        cen = v.mean(axis=1, keepdims=True)
        dev = v - cen
        spread_ratio = (rf[1] + 1e-6) / (vf[1] + 1e-6)
        s = (1 - a) + a * np.clip(spread_ratio, 0.5, 2.0)          # blend toward reference expansiveness
        dev = dev * s
        dev[..., 1] += a * (rf[2] - vf[2]) * 0.5                    # nudge vertical bias toward the genre
        out[:, joints, :] = cen + dev

    sm = one_euro_smooth(out)
    return strength * sm + (1.0 - strength) * out                  # smoothing also scales with strength


def one_euro_smooth(clip, alpha=0.4):
    """Light temporal smoothing (stand-in for the 1e-filter) to kill zone-boundary jitter."""
    out = clip.copy()
    for t in range(1, len(clip)):
        out[t] = alpha * clip[t] + (1 - alpha) * out[t-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visitor", required=True, help="visitor clip (.json from portal, or .npy)")
    ap.add_argument("--reference", required=True, help="genre reference clip (.npy/.json), HumanML3D-22")
    ap.add_argument("--alpha", type=float, default=None, help="uniform identity<->style weight override (0..1)")
    ap.add_argument("--out", default="styled.npy")
    a = ap.parse_args()
    visitor, reference = load_clip(a.visitor), load_clip(a.reference)
    alpha = {z: a.alpha for z in ZONES} if a.alpha is not None else None
    styled = transfer(visitor, reference, alpha)
    np.save(a.out, styled.astype(np.float32))
    print(f"[genre_style] visitor {visitor.shape} + reference {reference.shape} -> {styled.shape} -> {a.out}")
    print("Note: proxy transfer (real vocabulary needs your semantic_spectrum encoder - see ADAPT #1/#2).")


if __name__ == "__main__":
    main()
