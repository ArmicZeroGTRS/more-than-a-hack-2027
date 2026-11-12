# Technical report — possession inference for BAR-ATH

Team submission for **More than a Hack 2027, Phase 1: Ball Possession**.
This document describes what the code in this repository actually does. Where a number is not
measured, it says so.

---

## 1. Problem, and what is actually optimised

One label per video frame (`H`, `A`, `D`, `X`), 199,319 rows. The ranking metric is `spellF1`: a
possession is treated as one object, predicted possessions are matched one-to-one to true ones by
temporal IoU (0.3–0.7), and the F1 of that match is averaged over the thresholds. `macroF1` is the
tie-breaker.

Three properties of that metric drive the whole design:

- **A possession is a maximal run of `H` or of `A`.** `D` runs are stoppages, not possessions, so a
  predicted `D` in the middle of one possession cuts it into two predicted spells — one of which can
  match at most, while the other becomes a false positive.
- **Fragmentation is expensive, a lag is cheap.** The metric's own table: a perfect submission one
  frame late scores 1.0000, one second late scores 0.6539. Erring towards "the same team still has
  it" is nearly free; erring towards "someone else has it now" is not.
- **`X` is not an escape hatch.** On the ranking metric every `X` is forward-filled with the previous
  label in the same half, so it means "repeat my last answer". `X` is therefore used only outside the
  two graded blocks, where nothing is scored.

So the system is built to produce **few, long, correctly-bounded possessions**, not to win a
frame-by-frame accuracy contest.

## 2. Assets used, and assets dropped

| asset | used | why |
|---|---|---|
| ATD tracking (tactical) | **yes** | the only signal needed: player and ball positions per frame |
| ATD metadata XML | **yes** | team membership of the 46 track slots |
| Smart Tagging | **yes, weakly** | one dead-ball prior over `SET PIECES` intervals, nothing else |
| Homography | **no** | the ATD coordinates are already on the pitch; a transform adds a failure mode and no information |
| tactical video | **no** | everything the model needs is in the tracking; decoding 7.6 GB of H.264 without a clear use is cost without evidence |
| broadcast video and its ATD | **no** | not synchronised to the tactical timeline, and a defensible mapping is a project of its own |
| public 5-minute ground truth | **not used for any decision** | see §8 |

The 46 slots are tracklets, not people (`ShirtNumber` is the slot index). Every feature is a
**team** aggregate — nearest player, player density around the ball, team centroid — so an identity
switch or a duplicate detection degrades one frame instead of corrupting a per-player history.

## 3. Architecture

```
ATD text ──► src/atd.py ──► src/evidence.py ──► src/model.py ──► CSV ──► official validator
             (parse,          (three scores        (Viterbi,
              cached)          per frame)           spell repair)
```

**`src/atd.py`** streams the 199,319-line export once into compact arrays: ball position, distance
from the ball to the nearest player of each team, per-team Gaussian player density around the ball,
a contact flag, team centroids and per-team player counts. Malformed lines are counted and reported,
never dropped quietly. Frame `frame_id = 1` lands on submission frame `0`. Positions are converted to
metres (105.3 × 68.0) before any distance is compared — normalised coordinates on a non-square pitch
silently stretch distances along the short axis. The file's own speed field is ignored: its unit is
undocumented, and every speed this system needs is measurable from positions at a known 25 fps.

**`src/evidence.py`** produces three scores per frame on one scale (≈0–3):

- *Home / Away* — a decaying function of the nearest player's distance to the ball, plus the team's
  player density around the ball, plus a contact bonus inside 2 m. Deliberately not "the nearest
  player wins": an opponent pressing from 1.5 m is nearer the ball than the player shielding it
  without being in possession.
- *Dead* — ball provably still for a second, or ball unseen for two seconds **while both team
  centroids are also still**, or the ball reported outside the pitch, plus the weak set-piece prior.

**`src/model.py`** runs a three-state Viterbi over those scores. A switch costs a fixed number of
nats, and an uncertain frame contributes the same log-probability to all three states, so the optimal
path holds its label until the evidence has consistently favoured another state for about half a
second. That is hysteresis as a model rather than two hand-set thresholds, and it is order-independent.
A repair pass then merges `H`/`A` runs shorter than 1.6 s into the longer neighbour, dissolves
stoppages under 1 s, and bridges a stoppage under 2 s that is flanked by the same team. Each half is a
separate sequence: no possession survives halftime, and every frame outside the graded blocks is `X`.

**`src/config.py`** holds every tunable in one frozen dataclass mirrored by `configs/final.json`.

## 4. The two hard cases

**The ball is invisible on 41 % of graded frames.** This is the single most important fact in the
dataset, and almost none of that absence is a stoppage. So absence gives *no* team evidence — the two
team scores are zeroed, which makes the frame neutral and lets the temporal model hold its state —
and becomes dead-ball evidence only when the players have stopped as well. A missing ball is treated
as an observation problem, never as a transition and never as a dead ball on its own.

**Dead ball has to be conservative.** Following §1: over-predicting `D` costs a spell, under-predicting
it costs a few IoU points on one spell. Hence the rules above are conjunctive (ball still *and* the
shape of the game still), and short stoppages between the same team are bridged rather than kept.
`D` is never inferred from a set-piece tag alone. The same logic covers a case the code never
hardcodes: the second half is graded from frame 110,808 but the observable kickoff is roughly 2,900
frames later, and two minutes of two teams standing still is exactly what the dead-ball rule detects.

## 5. Training data

**None.** No labelled data of any kind was used — not the public 5-minute window, not the private
full-match truth, not external event or possession data. The parameters are football priors with
their reasoning written next to them in `src/config.py` and `src/evidence.py`: a controlling player
is about a stride from the ball; a possession shorter than ~1.6 s is an artefact; a genuine stoppage
lasts longer than a couple of seconds. There is no supervised model, so there is nothing to
overfit and no train/test split to leak.

## 6. Evaluation methodology, and what is *not* claimed

- The **official scorer is imported, not reimplemented** (`run_pipeline.py` calls
  `possession_metrics.validate_submission`), and the pipeline exits non-zero if the file fails.
- The four published baselines were **reproduced from a clean clone** before any modelling code was
  written; the local board matches `README.md` exactly.
- `tests/test_pipeline.py` checks the code, not the score: frame alignment, missing values, neutral
  evidence on an absent ball, a short spike not causing a switch, sustained evidence switching once,
  short-run merges, real stoppages surviving, blocks never being crossed, and the 199,319-row file
  contract against the official validator. 15 checks, no data required, run in CI.
- **No claim is made here that this beats the baseline.** The only ground truth available locally is
  17 possessions on the public window, which `EVALUATION.md` shows is enough to rank two versions of
  the same system backwards. A measured comparison would need the private full-match truth, which
  nobody outside the organisers has.

## 7. Limitations

- **Turnovers during ball absence are invisible.** With no ball there is no local evidence, so the
  model holds its previous answer. A change of possession that happens entirely inside an occlusion
  is missed, and only the next visible contact corrects it.
- **Possession is inferred from geometry, not from the ball's history.** A player standing near a
  ball that is actually travelling past them still looks like a possessor for the frames it takes the
  density term to change.
- **Goalkeeper handling is generic.** Keepers are not identified; a keeper holding the ball is
  reached through the same evidence as anyone else, so a keeper standing in a crowded six-yard box
  is a weak spot.
- **Smart Tagging is barely used.** `Type`, `Side`, mirrored phase pairs and the `N/A` throw-ins were
  not modelled; the only consumer is a dead-ball prior over `SET PIECES` intervals.
- **Intervals are frames.** Every threshold is a frame count at 25 fps, so added time and long
  stoppages are handled by state, not by any clock.
- **Not attempted:** video decoding, own detection or tracking, broadcast synchronisation, a learned
  classifier, a dashboard. Each has to earn its place against `spellF1` first, and none of them is
  needed for a valid submission.

## 8. Reproducibility

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/mtah2027            # Windows: $env:MTAH_DATA_DIR = "D:\mtah2027"

python tests/test_pipeline.py                     # 15 self-checks, no data needed
python run_pipeline.py --config configs/final.json --out submissions/submission.csv
```

The run parses the tracking once (cached outside the repository, keyed on the source file's size and
mtime), scores every frame, writes the CSV, runs the official validator and prints PASS/FAIL plus the
label counts and spell count. Same inputs, same bytes every time: no randomness anywhere in the
pipeline, and the parameters are pinned in `configs/final.json`.

Repository hygiene is enforced rather than promised: `.gitignore`, a pre-commit hook
(`git config core.hooksPath .githooks`) and `scripts/check_repo_hygiene.py`, which fails on any
tracked match asset, any derived frame-level export, and any file over 5 MB. The same script takes
`--staged` for the hook and `--tracked` for a manual or CI check. `submissions/`, `reports/` and the
parse cache are never committed.

## 9. External code and data references

- Official repository, code and documentation: `metrica-sports/more-than-a-hack-2027` (MIT), used
  unmodified — `games.py`, `evaluation/`, `starter/`. The scorer and the validator were not touched.
- `numpy`, `pandas` only; versions pinned in `requirements.txt` as installed (numpy 2.5.4, pandas
  3.1.1, CPython 3.12.10).
- No pretrained model, no external dataset, no commercial API, no hosted inference.
