# Phase 0 — repository and data audit

Everything below was read from the repository itself (`games.py`, `evaluation/`, `starter/`, the
markdown documents) and, where marked *unverified*, still has to be confirmed against the gated
bundle. The competition data was not opened while this document was written, and no match-specific
constant has been inferred from it.

## 1. What already exists

| entry point | role |
|---|---|
| `games.py` | the only place paths, frame counts, offsets and the graded blocks are defined |
| `evaluation/possession_metrics.py` | the metric library. **The specification**: every number on the board is defined here |
| `evaluation/run_evaluation.py` | the scorer. Reproduces the published board exactly |
| `evaluation/validate_submission.py` | the file contract, as a standalone check |
| `starter/baseline.py` | the floor: nearest tracked player to the ball + centred majority vote |
| `starter/load_atd.py`, `load_smart_tagging.py`, `load_homography.py` | loaders with verified time bases |
| `run.py` | a previous minimal attempt: `baseline.py` plus two constants tied to specific frames of this match |

`run.py` is left in place but is superseded: two of its constants (the second-half kickoff frame and
club-name substring matching) are exactly the kind of match-specific constant `RULES.md` rules out,
and it cannot do better than the baseline it wraps. `TECHNICAL_REPORT.md` described a module tree
that does not exist in this repository; it has been rewritten to describe what actually runs.

## 2. Data path configuration

`games.DATA_DIR` = `$MTAH_DATA_DIR` if set, else `./data`. `asset_paths(source=...)` builds every
filename from `SOURCES`/`SUFFIXES`, so nothing in the pipeline spells a filename out. `require()`
raises once, listing every missing file. The evaluation needs no data at all — it reads CSVs — so a
fresh clone can be scored before the bundle exists.

## 3. Frame alignment

Three clocks, and they only agree by construction:

| source | index | to a submission row |
|---|---|---|
| submission | video frame 0..199318 | — |
| ATD `frame_id` | 1..199319 | `video_frame = frame_id - 1` |
| ATD raw `IdFrame` (inside the XML/TXT) | the tracker's own counter | `id_frame = video_frame + offsets[half]`, `1702057` / `1702058` |
| Smart Tagging | seconds from the first tactical frame | `round(seconds * 25)` |
| homography | 0-based `frame` | already a submission row |

The `frame_id` in the kloppy dataframe is a **positional** index, which is why it starts at 1 and
matches `pollevents`; the offsets in `games.py` map the tracker's *own* counter. Both statements in
`DATA.md` are true and neither replaces the other. `src/atd.py` uses `frame_id - 1` and is tested
on it (`tests/test_pipeline.py::test_parse_alignment_and_missing`).

## 4. Team mapping

`<Team>` order in the metadata decides home/away (FIFA/EPTS convention, and what `starter/baseline.py`
does); each `<Player>` carries `teamId`. Slots are tracklets, not people: 46 slots for 22 players,
`ShirtNumber` is the slot index. Every feature in `src/` is therefore a **team aggregate**, never a
per-slot one. The pipeline prints both club names at start-up so a reordered export is visible.

## 5. Scoring behaviour (read from the code, not the prose)

- Graded frames: `valid_runs` derives them from the ground truth, so the two halves are
  `(11185, 90652)` and `(110808, 190198)` — half-open, and never merged across halftime.
- `X` is forward-filled inside each half (back-filled at the head) before any temporal metric.
- A **possession is a maximal run of `H` or of `A`**; `D` runs are not possessed by anyone.
- `spellF1` = F1 of a one-to-one greedy IoU match, averaged over IoU 0.3/0.4/0.5/0.6/0.7.
- `macroF1` = mean per-class F1 over H, A, D; a predicted `X` on a graded frame is a plain miss.
- Ungraded frames are outside every run, so whatever is written there is ignored.

**Consequences used in the design.** A false `D` inside a possession splits one true spell into two
predicted ones, which costs a whole spell *and* adds a false positive; a missed `D` only makes one
predicted spell slightly too long. So the dead-ball rule is deliberately the conservative side of
that trade, and `src/model.py` bridges short stoppages flanked by the same team.

## 6. Baseline reproduction

```
python evaluation/run_evaluation.py        # scored against the public 5-minute window
```

Reproduced on a clean clone before any modelling work was written, on Python 3.12.10 with numpy
2.5.4 / pandas 3.1.1: the four shipped baselines score exactly the values printed in `README.md`
(spellF1 0.4242 / 0.3256 / 0.2868 / 0.0378, macroF1 0.5812 / 0.6207 / 0.6200 / 0.5972). The
published full-match floor is `baseline_smooth50` at spellF1 0.2433; it cannot be reproduced locally
because the full-match ground truth is private, so 0.2433 is treated as the number to beat, not as a
measurement available here.

## 7. Known uncertainties (still open, and how the code avoids depending on them)

1. **Speed units.** The XML labels speed in kph; magnitudes look like m/s. Unresolved by inspection
   and irrelevant: `src/` never reads the speed field, every speed is measured from positions at
   25 fps.
2. **Homography convention.** `DATA.md` states the matrix maps normalised pitch to pixels; a
   local audit reported pitch corners landing outside the image. It is not needed: the ATD
   coordinates are already on the pitch, so no transform enters the possession decision. The
   homography is only required by `run_pipeline.py` as an asset-presence check.
3. **Ball coordinates outside 0-1.** Treated as a separate dead-ball signal (`oob_margin`), never
   clipped, and never used to move a player.
4. **Goalkeeper identity.** The XML may flag keepers; this was not verified, so no feature depends on
   it. A keeper holding the ball is reached through the same density evidence as anyone else.
5. **Smart Tagging semantics.** Interval boundaries, `Type` coverage, mirrored phase pairs and the
   `N/A` team were not verified against the file. The parser is `starter/load_smart_tagging.py`
   (verified time base) and the only consumer is one weak dead-ball prior.
6. **Second-half kickoff.** Graded frames start at 110808 but the observable kickoff is ~113742, so
   roughly two minutes of graded time precede the restart. Nothing is hardcoded: the dead-ball
   evidence (ball still or unseen, both team centroids still for a second) covers it, which is why
   that rule exists at all.
7. **Per-frame grade of the graded block edges.** `valid_runs` is derived from the ground truth, so a
   block boundary can only be confirmed by scoring. `games.half_video_blocks` is used for prediction
   and is the published, label-free version.

## 8. Planned architecture (as built)

```
run_pipeline.py        orchestrate: parse (cached) -> evidence -> Viterbi -> repair -> CSV -> validator
src/atd.py             stream the ATD text into compact per-frame arrays
src/evidence.py        per-frame Home / Away / dead-ball scores
src/model.py           Viterbi over H-A-D, then spell repair
src/config.py          every tunable, one frozen dataclass, JSON-mirrored
configs/final.json     the parameters used for the submitted file
tests/test_pipeline.py synthetic self-checks (no data)
scripts/check_repo_hygiene.py + .githooks/pre-commit + CI   nothing private is committed
```

Not built, on purpose: video decoding, broadcast synchronisation, a learned classifier, and a
dashboard. Each is a later stage that has to earn its place against `spellF1` first.
