# Third-Party Components, Dependencies and Licenses

This document discloses all third-party libraries, external dependencies, and AI development assistance used in the **NOP Vision Intelligence Challenge 2026 Candidate Solution**, conforming to `RULES.md` and `SUBMISSION.md`.

---

## 1. External Libraries & Frameworks

| Component | Version | Source / Provider | License | Commercial Use Status | Usage / Rationale |
|---|---|---|---|---|---|
| **OpenCV (`opencv-python`)** | 4.11.0.86 | OpenCV Foundation / GitHub | Apache 2.0 | Permitted | Video decoding, encoding, colorspace conversions (HSV), contours, morphology, and drawing utilities. |
| **NumPy** | 1.26.4 | NumPy Developers / PyPI | BSD 3-Clause | Permitted | Vectorized mathematical operations, matrix manipulations, and Kalman state calculations. |
| **SciPy** | 1.13.1 | SciPy Community / PyPI | BSD 3-Clause | Permitted | `scipy.optimize.linear_sum_assignment` for optimal bipartite Hungarian matching in multi-object tracking. |
| **Pandas** | 2.3.3 | Pandas Development Team | BSD 3-Clause | Permitted | Tabular data manipulation, CSV parsing, and evaluation data restructuring. |
| **jsonschema** | 4.25.1 | Julian Berman / PyPI | MIT License | Permitted | Schema validation of generated observation records against `nop_reference/evidence_contract.json`. |
| **pytest** | 8.4.2 | pytest-dev / PyPI | MIT License | Permitted | Automated test execution and test discovery for the unit test suite. |
| **Streamlit** | 1.50.0 | Snowflake Inc. / PyPI | Apache 2.0 | Permitted | Interactive web application for the Video Intelligence Review Surface and evidence audit dashboard. |

---

## 2. Pretrained Models & Weights

- **Core Pipeline**: Runs an **offline, self-contained mathematical perception engine** (`AdaptiveDetector` + `RobustKalmanTracker`) requiring zero external proprietary weights or cloud dependencies.
- **Deep Model Extension (`PretrainedDnnDetector`)**: Supports standard ONNX / Caffe / Darknet models (e.g., MobileNet-SSD, YOLO) via OpenCV DNN. When enabled, standard open-source weights (e.g. MobileNet-SSD under Apache 2.0) are utilized with local CPU execution.

---

## 3. Synthetic Datasets & Ground Truth

- **TSCI / GTS25 Public Benchmark Pack**: S01, S02, S03, S04 generated using `tools/generate_synthetic_dataset.py`.
- **Ground Truth**: `data/ground_truth/public_event_truth.csv`.

---

## 4. AI-Assisted Development Disclosure

Conforming to Section 14 of `RULES.md`:
- **AI Coding Assistant**: Google DeepMind Antigravity / Gemini 3.8 Flash.
- **Scope of AI Assistance**:
  - Code architecture refactoring and modularization.
  - Designing Kalman Filter covariance matrix parameters and occlusion coasting mechanisms.
  - Debugging multi-object contour separation during dense crossing scenarios.
  - Writing automated unit tests and test fixtures.
  - Generating documentation and reports.
- **Human/Engineering Responsibility**: All architectural decisions, mathematical state formulas, edge-case failure analyses, and validation metrics were verified and confirmed through empirical execution and benchmark evaluation.
