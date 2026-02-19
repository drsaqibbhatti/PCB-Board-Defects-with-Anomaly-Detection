# PCB Board Defects with Anomaly Detection (Heatmap Localization)

Anomaly detection pipeline for **PCB board inspection** that produces **heatmap-based defect localization** (e.g., short/bridge, missing/extra patterns, irregular regions) with **reduced labeling requirements**.

Project page: https://drsaqibbhatti.com/projects/pcb-anomaly.html

---

## Overview

### Problem
PCB inspection often suffers from:
- limited labeled defect examples
- many defect types that change over time
- need for **localization** (not only pass/fail)

### Solution
This project uses an **anomaly detection approach** to:
- learn “normal” PCB patterns
- detect deviations as anomalies
- output **heatmaps** to localize defect regions

---

## Key Features
- Heatmap anomaly detection for PCB boards
- Localization-focused output (overlay + ROI)
- Works with limited defect labels (normal-heavy training)
- Post-processing for stable decisions and fewer false alarms
- Designed for production-style AOI workflows

---

## Pipeline (High Level)
1. **Preprocessing**
   - resize / normalize
   - optional ROI alignment / masking
2. **Anomaly model inference**
   - generate anomaly score map / heatmap
3. **Post-processing**
   - smoothing / thresholding
   - connected components / region filtering
4. **Decision**
   - pass/fail + localized defect visualization

---

## Tech Stack
- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**

---

## Results
- Reliable PCB anomaly detection with strong localization quality
- Visual heatmaps for operator verification
- Robust decision logic via post-processing

> Note: Specific production metrics and client details are not included (confidential).

