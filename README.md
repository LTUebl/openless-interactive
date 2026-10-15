# OpenLESS Interactive: Automated Jump-Landing Biomechanics & ACL Injury Risk Evaluation

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat&logo=github)](https://ltuebl.github.io/openless-interactive/)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.11.28.24318160-darkred)](https://doi.org/10.1101/2024.11.28.24318160)
[![Institution](https://img.shields.io/badge/Lawrence%20Technological%20University-EBL-navy)](https://www.ltu.edu)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, educational biomechanics web application for automated jump-landing assessment and ACL injury risk screening, developed by the **Experimental Biomechanics Laboratory (EBL)** in the **Department of Biomedical Engineering** at **Lawrence Technological University (LTU)**.

🌐 **Public Interactive Platform:** [https://ltuebl.github.io/openless-interactive/](https://ltuebl.github.io/openless-interactive/)  
📄 **Precomputed Clinical PDF Report:** [OpenLESS_Report_OpenCapData_71.pdf](OpenLESS_Report_OpenCapData_71.pdf)

---

## 📌 Scientific & Biomechanical Foundation

The **Landing Error Scoring System (LESS)** is a validated clinical screening tool used to identify aberrant jump-landing mechanics associated with non-contact anterior cruciate ligament (ACL) injury (Padua et al., 2009). Traditionally, LESS scoring requires trained clinicians to manually inspect high-speed 2D video across sagittal and frontal views, scoring 19 distinct kinematic and postural items.

**OpenLESS** (Turner et al., 2024) automates this assessment using markerless motion capture powered by **OpenCap** (Uhlrich et al., 2022, 2023). By processing synchronized multi-camera smartphone video through neural networks and OpenSim inverse kinematics, OpenLESS calculates continuous 3D joint kinematics and evaluates all 19 standardized LESS criteria without human observer bias or manual digitizing.

### Citation
> Turner, J. A., et al. (2024). *Automated Scoring of Jump-Landing Biomechanics using Markerless Motion Capture (OpenLESS).* **medRxiv**, 2024.11.28.24318160. [https://doi.org/10.1101/2024.11.28.24318160](https://doi.org/10.1101/2024.11.28.24318160).

---

## 🚀 Key Platform Features

1. **Synchronized Dual-Camera Video Playback**
   - High-speed 60 Hz synchronized video player with slow-motion (0.25x, 0.5x, 1.0x) and single-frame stepping (1/60s).
   - Toggle between **Cam 0** (sagittal/side-view perspective) and **Cam 1** (angled/frontal perspective).
   - Scrub bar precisely linked to the subject's stance phase.

2. **Automated 19-Criteria Scorecard**
   - Dynamic interactive scorecard categorizing criteria by movement phase:
     - *Initial Ground Contact (Sagittal):* Knee flexion, hip flexion, trunk flexion, ankle plantarflexion.
     - *Initial Ground Contact (Frontal/Transverse):* Foot landing timing asymmetry, foot position asymmetry, lateral trunk flexion, medial knee position, knee valgus / stance width, foot rotation.
     - *Maximum Stance Displacement:* Knee flexion displacement, hip flexion displacement, trunk displacement, medial knee collapse / dynamic valgus, lateral hip shift.
     - *Overall Composite:* Sagittal joint displacement (soft vs. stiff), overall movement impression.
   - Expandable drawers containing the **clinical & biomechanical rationale** for each item.
   - Filter views by movement phase or isolate errors.

3. **Interactive Plotly 3D Kinematics**
   - **Sagittal Plane:** Time-series curves for knee flexion, hip flexion, and forward trunk lean against clinical LESS threshold cutoffs (30° and 65° dashed reference lines).
   - **Frontal & Transverse Planes:** Continuous hip adduction (dynamic valgus), hip rotation, and lateral trunk tilt.
   - **Contact Event Detection:** 4th-order zero-phase Butterworth-filtered vertical trajectories ($Y$) of left/right toe and heel markers showing automated initial contact (IC) and toe-off (TO) event detection.

4. **Comparative Analysis Mode**
   - Side-by-side kinematic overlays and itemized error point distributions comparing **Stepoff-Leaning** (compliant, protective landing) vs. **Stepoff-Upright** (stiff, high-risk landing).

5. **In-Browser Client-Side Evaluator (Upload Custom Data)**
   - Students and researchers can drag-and-drop their own OpenCap `.trc` (marker coordinates) and `.mot` (OpenSim kinematics) files.
   - Runs client-side zero-phase Butterworth filtering and calculates all 19 LESS scores entirely in the browser—no server or Python installation required!

6. **Educational Student Lab Module**
   - Four embedded conceptual questions with interactive "Reveal Answer" accordions explaining:
     - Quadriceps tendon angle of insertion and anterior tibial shear forces.
     - Proactive trunk lean and external joint moment arm redistribution.
     - Multi-planar dynamic knee valgus collapse mechanisms.
     - Impulse-momentum theorem ($\int F \, dt = \Delta p$) and peak ground reaction force damping.

---

## 📊 Summary of Example Trials (`OpenCapData_71`)

| Parameter | Trial 1: Stepoff-Leaning | Trial 2: Stepoff-Upright | Clinical Significance |
| :--- | :---: | :---: | :--- |
| **Total LESS Score** | **1 / 19** (Excellent) | **14 / 19** (Poor) | $\le 4$ = Low Risk, $\ge 8$ = High ACL Risk |
| **ACL Injury Risk** | **Low Risk** | **High Risk** | High scores correlate with increased non-contact tears |
| **Knee Flexion @ IC** | $35.9^\circ$ (L) / $37.3^\circ$ (R) | $18.5^\circ$ (L) / $13.9^\circ$ (R) | Cutoff $\ge 30^\circ$. Extended knees load ACL via quad pull |
| **Hip Flexion @ IC** | $39.5^\circ$ (L) / $41.8^\circ$ (R) | $18.2^\circ$ (L) / $20.9^\circ$ (R) | Cutoff $\ge 30^\circ$. Gluteal shock absorption |
| **Trunk Flexion @ IC** | **$45.2^\circ$ forward lean** | **$-6.3^\circ$ upright/ext** | Cutoff $\ge 10^\circ$. Forward lean transfers load to hips |
| **Knee Displacement** | $45.5^\circ$ (L) / $44.1^\circ$ (R) | $23.6^\circ$ (L) / $19.4^\circ$ (R) | Cutoff $\ge 45^\circ$. Deep cushioning attenuates GRF |
| **Foot Strike Pattern** | Forefoot first (both feet) | Heel strike first (left foot) | Heel strikes bypass Achilles/calf spring compliance |
| **Dynamic Knee Valgus** | None ($0^\circ$ adduction) | **Severe** ($+3$ error points) | Primary predictor of non-contact ACL rupture |
| **Lateral Hip Shift** | $0.4^\circ$ difference | **$10.8^\circ$ asymmetric shift** | Asymmetric loading overwhelms unilateral knee |
| **Overall Impression** | **0 (Excellent)** | **2 (Poor / High Risk)** | Global multi-planar rating |

---

## 📁 Repository Directory Structure

```text
openless-interactive/
│
├── index.html                       # Standalone interactive single-page application (Tailwind + Plotly)
├── app.js                           # App logic, precomputed dataset, Plotly charts, client-side OpenLESS engine
├── precomputed_data.json            # Extracted time-series kinematics & scores for instant client loading
├── OpenLESS_Report_OpenCapData_71.pdf# 3-page publication-quality PDF report with data visualizations
├── README.md                        # Documentation and student lab guide
│
├── videos/                          # Synchronized 60 Hz MP4 camera feeds
│   ├── Stepoff-Leaning_sync.mp4     # Trial 1 Cam0 (Sagittal)
│   ├── Stepoff-Leaning_cam1_sync.mp4# Trial 1 Cam1 (Angled)
│   ├── Stepoff-Upright_sync.mp4     # Trial 2 Cam0 (Sagittal)
│   └── Stepoff-Upright_cam1_sync.mp4# Trial 2 Cam1 (Angled)
│
├── data/                            # OpenCap raw and processed files
│   ├── Stepoff-Leaning.mot          # OpenSim inverse kinematics joint angles
│   ├── Stepoff-Leaning.trc          # 3D marker coordinate trajectories
│   ├── Stepoff-Upright.mot          # OpenSim inverse kinematics joint angles
│   ├── Stepoff-Upright.trc          # 3D marker coordinate trajectories
│   ├── less_scores_final.csv        # Comprehensive 68-parameter scoring output
│   └── less_short_final.csv         # 21-column summary scorecard
│
└── python/                          # Python CLI scoring and PDF export scripts
    ├── run_openless_scoring.py      # Automated OpenLESS signal processing and scoring script
    └── export_openless_pdf.py       # High-resolution ReportLab PDF generation script
```

---

## 💻 Running the Python Pipeline Locally

To run the automated scoring and generate reports locally using Python:

```bash
# Clone the repository
git clone https://github.com/LTUebl/openless-interactive.git
cd openless-interactive

# Install dependencies
pip install numpy pandas scipy reportlab matplotlib

# Run automated scoring on OpenCap data
python python/run_openless_scoring.py

# Export publication-quality PDF report
python python/export_openless_pdf.py
```

---

## 🏫 Educational Use & Laboratory Instruction

This tool is designed for undergraduate and graduate courses in:
- **Biomechanics of Human Movement**
- **Orthopedic Biomechanics**
- **Sports Science & Injury Prevention**
- **Rehabilitation Engineering**

### Suggested Student Assignment:
1. Open the [live website](https://ltuebl.github.io/openless-interactive/).
2. Watch both landing trials in 0.5x slow-motion from Cam0 and Cam1.
3. Compare the sagittal plane joint displacement curves between *Stepoff-Leaning* and *Stepoff-Upright*.
4. Identify at least three specific criteria in the 19-item scorecard where the upright subject accumulated error points.
5. Answer the four embedded student lab questions and discuss how targeted feedback (e.g., "bend your knees", "lean forward at the waist", "land softly on the balls of your feet") can retrain jump-landing motor patterns to reduce ACL rupture incidence.

---

## 📄 License & Attribution

Developed by **Dr. Eric Meyer** and the **Experimental Biomechanics Laboratory (EBL)**, Lawrence Technological University.  
OpenLESS algorithm based on work by **Turner et al. (2024)**, Stanford University and collaborating institutions.  
Released under the MIT License.
