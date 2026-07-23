"""End-to-end test for run_local: visitor+reference -> transfer -> VMC precompute -> OSC loopback."""
import json, os, threading, time, sys, subprocess
import numpy as np
from hml_skeleton import fk, q_from_two_vectors, PRIMARY_CHILD, PARENTS, T_POSE
import run_local, genre_style

def synth(seed, motion):
    rng = np.random.default_rng(seed); off = np.zeros((22,3))
    for j,val in T_POSE.items(): off[j]=val
    T=90; gq=np.tile(np.array([1.,0,0,0]),(T,22,1))
    for t in range(T):
        for j in range(22):
            c=PRIMARY_CHILD[j]
            if c is None: gq[t,j]=gq[t,PARENTS[j]]; continue
            base=np.array(off[c],float)
            d=base+motion*np.linalg.norm(base)*np.array([np.sin(0.2*t+j),np.cos(0.15*t+j),np.sin(0.1*t+2*j)])
            gq[t,j]=q_from_two_vectors(base,d)
    root=np.stack([np.array([0.002*t,0.01*np.sin(0.1*t),0]) for t in range(T)])
    return fk(gq,off,root).astype(np.float32)

def ck(n,c): print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
ok=True

# write synthetic visitor + a genre reference JSON in the portal format
vis=synth(0,0.20); ref=synth(7,0.55)
json.dump({"format":"HumanML3D-22","fps":30,"frames":len(vis),"joints":vis.tolist()}, open("/tmp/vis.json","w"))
os.makedirs("/tmp/refs",exist_ok=True)
json.dump({"format":"HumanML3D-22","fps":30,"frames":len(ref),"joints":ref.tolist()}, open("/tmp/refs/gJB_ballet.json","w"))

# resolve by genre name
rp = run_local.resolve_reference(None, "ballet", "/tmp/refs")
ok &= ck("genre 'ballet' resolves to gJB reference", os.path.basename(rp)=="gJB_ballet.json")

# run transfer chain
_,_,styled = run_local.run("/tmp/vis.json", rp, 0.5)
ok &= ck("styled shape (T,22,3)", styled.shape==vis.shape)
ok &= ck("styled finite", np.all(np.isfinite(styled)))

# VMC precompute produces per-frame bone pos+quat for all 22 bones
from stage7_vmc_sender import precompute
upos,uquat = precompute(styled)
ok &= ck("precompute shapes", upos.shape==(len(styled),22,3) and uquat.shape==(len(styled),22,4))
ok &= ck("quats finite & ~unit", np.all(np.isfinite(uquat)) and abs(np.linalg.norm(uquat[0,1])-1)<0.1)

# OSC loopback: stream a couple frames, count messages
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
got={"root":0,"bone":0,"ok":0}
disp=Dispatcher()
disp.map("/VMC/Ext/Root/Pos", lambda a,*x: got.__setitem__("root",got["root"]+1))
disp.map("/VMC/Ext/Bone/Pos", lambda a,*x: got.__setitem__("bone",got["bone"]+1))
disp.map("/VMC/Ext/OK", lambda a,*x: got.__setitem__("ok",got["ok"]+1))
srv=osc_server.ThreadingOSCUDPServer(("127.0.0.1",39547),disp)
threading.Thread(target=srv.serve_forever,daemon=True).start()
run_local.stream(styled[:3], "127.0.0.1", 39547, 120, False)
time.sleep(0.3); srv.shutdown()
ok &= ck(f"OSC loopback: root={got['root']} bone={got['bone']} ok={got['ok']}",
         got["root"]==3 and got["bone"]==3*22 and got["ok"]==3)

print("\nRESULT:", "ALL PASS ✓" if ok else "SOME FAILED ✗")
