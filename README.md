# embodied-model — Unnoticed Dance

Home for the **real-time identity-preservation motion installation** (Tokyo gallery + humanoid-robotics SDK prototype). A visitor moves in front of a webcam and is re-expressed as a 3D dance avatar with their movement *identity* preserved.

## Layout
```
index.html          Project dashboard (pipeline status + runbook)   →  served at /
dance-graph.html    Interactive tool knowledge graph                →  served at /dance-graph
vercel.json         Static routing (cleanUrls)
pipeline/           Python pipeline scaffolds + tests + Unity notes
```

## Web (Vercel)
Static, self-contained, no build step. Import this repo at **vercel.com/new** → Deploy.
Dashboard at `/`, tool graph at `/dance-graph`. Every push auto-redeploys.
Local preview: `npm start` → http://localhost:3000

## Pipeline
`pipeline/` holds the Stage 1–7 scaffolds (capture → lift → blend → retarget → VMC→Unity),
all unit-tested in isolation. See `pipeline/README.md` for run order. The blend itself
(`semantic_spectrum`) + datasets run on the local Mac / momask-codes checkout.

## Two license tracks
Gallery/paper track may use AMASS/HumanML3D/MoMask/AIST++ (non-commercial). The SDK product
track must stay on Mixamo + MotionAGFormer (Apache-2.0). Keep them clean — see `/dance-graph`.
