"""Test view_clip (render) and stage7_vmc_sender (OSC loopback) with a synthetic clip."""
import subprocess, sys, time, threading, os
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE

# --- build a synthetic (T,22,3) clip and save it ---
T = 40
off = np.zeros((22, 3))
for j, v in T_POSE.items(): off[j] = v
gq = np.tile(np.array([1., 0, 0, 0]), (T, 22, 1))
for t in range(T):
    for j in range(22):
        c = PRIMARY_CHILD[j]
        if c is None: gq[t, j] = gq[t, PARENTS[j]]; continue
        base = np.array(off[c], float)
        d = base + 0.25 * np.linalg.norm(base) * np.array([np.sin(0.3*t+j), np.cos(0.2*t+j), 0.2])
        gq[t, j] = q_from_two_vectors(base, d)
root = np.stack([np.array([0.003*t, 0, 0]) for t in range(T)])
seq = fk(gq, off, root).astype(np.float32)
np.save("synthetic_clip.npy", seq)
print(f"[setup] wrote synthetic_clip.npy {seq.shape}")

# --- TEST 1: view_clip renders a gif ---
r = subprocess.run([sys.executable, "view_clip.py", "synthetic_clip.npy", "--save", "preview.gif"],
                   capture_output=True, text=True)
ok1 = os.path.exists("preview.gif") and os.path.getsize("preview.gif") > 1000
print(f"[view_clip] {'PASS' if ok1 else 'FAIL'} - {r.stdout.strip()} {r.stderr.strip()[-200:]}")

# --- TEST 2: VMC sender emits well-formed OSC to a loopback server ---
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

got = {"root": 0, "bone": 0, "ok": 0, "bones": set(), "argcount_ok": True}
def on_root(addr, *args):
    got["root"] += 1
    if len(args) != 8: got["argcount_ok"] = False
def on_bone(addr, *args):
    got["bone"] += 1; got["bones"].add(args[0] if args else None)
    if len(args) != 8: got["argcount_ok"] = False
def on_ok(addr, *args): got["ok"] += 1

disp = Dispatcher()
disp.map("/VMC/Ext/Root/Pos", on_root)
disp.map("/VMC/Ext/Bone/Pos", on_bone)
disp.map("/VMC/Ext/OK", on_ok)
srv = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 39539), disp)
threading.Thread(target=srv.serve_forever, daemon=True).start()

r2 = subprocess.run([sys.executable, "stage7_vmc_sender.py", "synthetic_clip.npy", "--fps", "120"],
                    capture_output=True, text=True)
time.sleep(0.3); srv.shutdown()

exp_bones = 22
ok2 = (got["root"] == T and got["bone"] == T*22 and got["ok"] == T
       and len(got["bones"]) == exp_bones and got["argcount_ok"])
print(f"[vmc] root={got['root']} bone={got['bone']} ok={got['ok']} "
      f"distinct_bones={len(got['bones'])} argcounts_ok={got['argcount_ok']}")
print(f"[vmc] {'PASS' if ok2 else 'FAIL'} - {r2.stdout.strip()} {r2.stderr.strip()[-200:]}")

print("\nRESULT:", "ALL PASS ✓" if (ok1 and ok2) else "SOME FAILED ✗")
