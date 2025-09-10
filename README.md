# CoMotion-Pipeline

Pipeline for processing CoMotion JSON outputs (SMPL 24-joint skeletons).  
This code estimates standing height, wrist span, and walking displacement with calibration by height (**S_h**) or tape distance (**S_d**).  
It also generates CSV summaries and figures for use in reports or dissertations.

---

##  Features
- Visualises **24-joint SMPL skeletons** (3D plots).
- Computes:
  - Standing height (raw, S_h, S_d).
  - Wrist span (raw, S_h, S_d).
  - Walking displacement (with pause detection).
- Exports:
  - Summary CSV (`skeleton_measurements.csv`).
  - Time-series CSV (height + wrist).
  - Plots: ground path, wrist curve, height bar, skeleton frames.
- Works on JSON outputs from **CoMotion**.

---

##  Installation

Clone this repository and Install dependencies:

```bash
git clone https://github.com/VaishnaviP18/CoMotion-Pipeline.git
cd CoMotion-Pipeline
pip install -r requirements.txt
