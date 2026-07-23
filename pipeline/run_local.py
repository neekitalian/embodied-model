"""
End-to-end LOCAL runner: visitor clip -> genre transfer -> (optional) VMC stream to Unity.

One command ties the whole local path together:
  portal/video capture (visitor_clip.json)  +  genre reference (refs/<genre>.json)
    -> genre_style.transfer (identity-preserving)  -> styled.npy  -> VMC -> VRM avatar

Deps: numpy, python-osc. Reuses genre_style.py + stage7_vmc_sender.py + hml_skeleton.py.

Examples:
  # produce a styled clip (Ballet), identity<->style at 0.5:
  python run_local.py --visitor visitor_clip.json --genre ballet --refs-dir refs/ --alpha 0.5 --out styled.npy
  # ...and stream it live to Unity (EVMC4U on UDP 39539):
  python run_local.py --visitor visitor_clip.json --reference refs/ballet.json --stream --loop
"""
import argparse, glob, os
import numpy as np
import genre_style
from stage7_vmc_sender import precompute, send_frame


def resolve_reference(reference, genre, refs_dir):
    if reference:
        return reference
    if not genre:
        raise SystemExit("give --reference PATH or --genre NAME (+ --refs-dir)")
    # match refs/<genre>*.json, or an AIST code (hip-hop covers Middle gMH + LA gLH)
    codes = {"jazz":["gJS"], "ballet":["gJB"], "hip-hop":["gMH","gLH"], "hiphop":["gMH","gLH"],
             "house":["gHO"], "break":["gBR"], "krump":["gKR"], "pop":["gPO"], "lock":["gLO"], "waack":["gWA"]}
    pats = [f"*{genre.lower()}*.json"] + [f"{c}*.json" for c in codes.get(genre.lower(), [])]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(refs_dir, p)))
        if hits:
            return hits[0]
    raise SystemExit(f"no reference for genre '{genre}' in {refs_dir} (looked for {pats})")


def run(visitor_path, reference_path, alpha, enhance=False):
    visitor = genre_style.load_clip(visitor_path)
    reference = genre_style.load_clip(reference_path)
    if enhance:
        import genre_motifs
        styled, sim = genre_motifs.enhance(visitor, reference, base_alpha=alpha if alpha is not None else 0.45)
        print(f"[enhance] motif similarity to this genre: {sim:.3f}")
    else:
        a = {z: alpha for z in genre_style.ZONES} if alpha is not None else None
        styled = genre_style.transfer(visitor, reference, a)
    return visitor, reference, np.asarray(styled, dtype=np.float32)


def stream(styled, host, port, fps, loop):
    import time
    from pythonosc.udp_client import SimpleUDPClient
    upos, uquat = precompute(styled)
    client = SimpleUDPClient(host, port)
    dt, T = 1.0 / fps, len(styled)
    print(f"[run_local] streaming {T} frames @ {fps:.0f}fps -> {host}:{port} (Ctrl-C to stop)")
    try:
        while True:
            for t in range(T):
                send_frame(client, upos[t], uquat[t]); time.sleep(dt)
            if not loop:
                break
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visitor", required=True, help="visitor clip (.json from portal / video_to_reference, or .npy)")
    ap.add_argument("--reference", help="genre reference clip path")
    ap.add_argument("--genre", help="genre name (resolved against --refs-dir) instead of --reference")
    ap.add_argument("--refs-dir", default="refs", help="folder of genre reference clips")
    ap.add_argument("--alpha", type=float, default=0.5, help="identity<->style weight 0..1 (0=you, 1=full genre)")
    ap.add_argument("--enhance", action="store_true", help="motif-gated: boost genre where you resemble its signature moves")
    ap.add_argument("--report", action="store_true", help="print which genre your motion most resembles (allocation)")
    ap.add_argument("--out", default="styled.npy")
    ap.add_argument("--stream", action="store_true", help="stream the styled clip to Unity over VMC")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    a.visitor = os.path.expanduser(a.visitor)
    a.reference = os.path.expanduser(a.reference) if a.reference else a.reference
    a.refs_dir, a.out = os.path.expanduser(a.refs_dir), os.path.expanduser(a.out)

    if a.report:
        import glob, genre_motifs
        refs = {os.path.splitext(os.path.basename(p))[0]: genre_style.load_clip(p)
                for p in sorted(glob.glob(os.path.join(a.refs_dir, "*.json")) + glob.glob(os.path.join(a.refs_dir, "*.npy")))}
        vis = genre_style.load_clip(a.visitor)
        print("[allocation] the genre your body most resembles:")
        for g, s in sorted(genre_motifs.allocate(vis, refs).items(), key=lambda kv: -kv[1]):
            print(f"  {g:12s} similarity {s:.3f}")
        if not a.genre and not a.reference:
            return

    ref = resolve_reference(a.reference, a.genre, a.refs_dir)
    visitor, reference, styled = run(a.visitor, ref, a.alpha, enhance=a.enhance)
    np.save(a.out, styled)
    print(f"[run_local] visitor {visitor.shape} + reference {reference.shape} ({os.path.basename(ref)}) "
          f"-> styled {styled.shape} @ alpha={a.alpha} -> {a.out}")
    if a.stream:
        stream(styled, a.host, a.port, a.fps, a.loop)
    else:
        print(f"Preview it:  python view_clip.py {a.out}   |   Stream it: add --stream")


if __name__ == "__main__":
    main()
