# Technical Report — AI-Powered Football Possession Detection System
**Competition:** More Than A Hack 2027 — Phase 1: Ball Possession
**Official Repository:** `metrica-sports/more-than-a-hack-2027`

## 1. Objective & Metric Optimization
Predicts frame-level ball possession (`H` = FC Barcelona, `A` = Athletic Club, `D` = Dead Ball, `X` = Ungraded/Abstain) across all 199,319 frames (25 FPS) to maximize `spellF1` (temporal IoU over 0.3..0.7) and `macroF1`.

## 2. Multi-Stage Architecture
1. **Data Loader & 10-Point Verification (`src/data_loader.py`, `src/data/audit.py`)**: Aligns ATD (`video_frame = frame_id - 1`), Smart Tagging (`round(seconds * 25)`), and Homography (`0`-indexed). Empirically verifies speed units (`m/s` vs `kph`), detects homography discontinuities, and scrubs single-frame teleportation glitches.
2. **Uncertainty-Aware Missing-Ball Recovery (`src/ball_recovery.py`)**: Handles the 40.7% missing-ball rate on graded frames using zero-phase Kalman smoothing on visible segments, cubic Hermite splines on short gaps (<=0.6s), damped velocity + player cluster focal blending on medium gaps (0.6-3.0s), and low-confidence focal fallback on long gaps (>3.0s).
3. **Identity-Invariant Team & Goalkeeper Features (`src/tracking_features.py`)**: Permutation-invariant across the 46 switching track slots; computes k-nearest distances, Gaussian pitch-control potentials, velocity alignment, pass-in-flight flags, <8-frame deflection suppression, per-half attack direction inference, and goalkeeper penalty-box possession logic.
4. **Smart Tagging Context & Mirrored Pair Deduplication (`src/smart_tagging_features.py`)**: Deduplicates mirrored offensive/defensive phase pairs, decomposes Set Pieces into pre-restart dead-ball vs live restart windows, and resolves `Team = N/A` throw-ins from touchline proximity.
5. **Multi-Signal Dead-Ball & Frame Probability Model (`src/dead_ball_detector.py`, `src/possession_model.py`)**: Combines out-of-bounds geometry, stationary ball duration, collective player slowdown, and pre-second-half kickoff dead-ball masking (`110808..113741` -> `D`).
6. **Temporal Segmentation Ensemble (`src/temporal_model.py`, `src/ensemble.py`)**: Combines Multi-Scale Probabilistic Segmentation, Context-Modulated HMM-Viterbi, and Semi-Markov Minimum-Duration Dynamic Programming per graded half.

## 3. Reproducibility & Execution
- Install: `pip install -r requirements.txt`
- Configure data: `export MTAH_DATA_DIR=/path/to/data`
- Run prediction: `python run.py --output outputs/mtah2027_team_submission.csv`
- Validate CSV: `python evaluation/validate_submission.py outputs/mtah2027_team_submission.csv`
- Evaluate: `python evaluation/run_evaluation.py`
