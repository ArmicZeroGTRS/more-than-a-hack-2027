"""Self-checks for the possession engine. No framework, no data: `python tests/test_pipeline.py`.

Everything here runs on synthetic arrays built in the file. That is the point -- these are
correctness checks for the code (does frame_id-1 line up, does a neutral frame hold the state, does a
6-frame flicker survive, does the CSV contract hold), not evidence about the competition match.
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import atd, config, evidence, model  # noqa: E402

PITCH = (105.3, 68.0)
PARAMS = config.Params()

META = """<?xml version="1.0"?>
<root>
  <Teams>
    <Team id="100"><Name>Home FC</Name></Team>
    <Team id="200"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player teamId="100" playerId="1" jerseyNo="1"/>
    <Player teamId="200" playerId="2" jerseyNo="2"/>
  </Players>
</root>
"""


def _write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def test_team_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        meta = _write(tmp, "meta.xml", META)
        assert list(atd.team_of_track(meta)) == [atd.HOME, atd.AWAY]
        assert atd.team_names(meta) == ("Home FC", "Away FC")


def test_parse_alignment_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        meta = _write(tmp, "meta.xml", META)
        raw = _write(tmp, "raw.txt", "\n".join([
            # frame_id 1 -> submission frame 0: home player on the ball, 1.05 m away
            "1:0.500,0.500,0.0;0.600,0.500,0.0:0.510,0.500,0.0",
            # frame_id 2 -> submission frame 1: ball not tracked
            "2:0.500,0.500,0.0;NaN,NaN,NaN:NaN,NaN,NaN",
            # a malformed line must be counted, never silently dropped
            "garbage",
            "3:NaN,NaN,NaN;0.700,0.500,0.0:0.700,0.500,0.0",
        ]) + "\n")

        tracking = atd.load(raw, 3, PITCH, atd.team_of_track(meta),
                            near_radius_m=2.0, density_sigma_m=3.0)

        assert tracking["malformed"] == 1
        assert tracking["out_of_range"] == 0
        assert tracking["n_slots"] == 2
        # frame_id 1 lands on index 0, not 1
        assert abs(tracking["ball"][0, 0] - 0.510) < 1e-6
        assert abs(tracking["dmin"][0, 0] - 1.053) < 1e-2
        assert abs(tracking["dmin"][0, 1] - (0.09 * 105.3)) < 1e-2
        assert tracking["contact"][0, 0] and not tracking["contact"][0, 1]
        assert tracking["density"][0, 0] > tracking["density"][0, 1] > 0
        # missing ball: no distance, no density, and the position stays NaN
        assert np.isnan(tracking["ball"][1, 0])
        assert np.isnan(tracking["dmin"][1, 0])
        assert tracking["density"][1].tolist() == [0.0, 0.0]
        # a missing player is not a player at (0, 0)
        assert tracking["count"][2].tolist() == [0, 1]
        assert abs(tracking["centroid"][2, 1, 0] - 0.700) < 1e-6


def _tracking(n, ball=(0.5, 0.5), home_dmin=0.4, away_dmin=6.0, home_density=2.0,
              away_density=0.2, ball_valid_from=0, centroid=None):
    track = {
        "ball": np.tile(np.asarray(ball, dtype=np.float32), (n, 1)),
        "dmin": np.stack([np.full(n, home_dmin, np.float32),
                          np.full(n, away_dmin, np.float32)], axis=1),
        "density": np.stack([np.full(n, home_density, np.float32),
                             np.full(n, away_density, np.float32)], axis=1),
        "contact": np.stack([np.full(n, home_dmin < 2.0), np.full(n, away_dmin < 2.0)], axis=1),
        "centroid": centroid if centroid is not None else np.tile(
            np.asarray([[0.35, 0.5], [0.65, 0.5]], dtype=np.float32), (n, 1, 1)),
    }
    if ball_valid_from > 0:
        track["ball"][:ball_valid_from] = np.nan
        track["dmin"][:ball_valid_from] = np.nan
        track["density"][:ball_valid_from] = 0.0
        track["contact"][:ball_valid_from] = False
    return track


def test_evidence_prefers_the_team_on_the_ball():
    scores = evidence.scores(_tracking(200), PARAMS, PITCH)
    assert (scores["home"] > scores["away"]).all()
    # a ball that never moves is also dead-ball evidence, which is not a contradiction
    assert (scores["dead"][-1] > 0)


def test_missing_ball_gives_no_team_evidence():
    track = _tracking(200, ball_valid_from=200)  # ball never tracked
    scores = evidence.scores(track, PARAMS, PITCH)
    assert (scores["home"] == 0).all() and (scores["away"] == 0).all()


def test_missing_ball_with_moving_teams_is_not_dead():
    track = _tracking(200, ball_valid_from=200)
    drift = np.linspace(0.35, 0.65, 200, dtype=np.float32)
    track["centroid"][:, 0, 0] = drift
    track["centroid"][:, 1, 0] = 1.0 - drift
    scores = evidence.scores(track, PARAMS, PITCH)
    assert not scores["team_still"][-1]
    assert scores["dead"][-1] == 0.0


def test_missing_ball_with_static_teams_is_dead():
    track = _tracking(200, ball_valid_from=200)
    # teams seen and standing still; centroids constant already, only the ball is missing
    track["centroid"][:] = (0.35, 0.5)
    scores = evidence.scores(track, PARAMS, PITCH)
    assert scores["dead"][-1] > 0


def test_static_ball_is_dead():
    scores = evidence.scores(_tracking(200), PARAMS, PITCH)
    assert scores["ball_still"][-1]
    assert scores["dead"][-1] >= PARAMS.w_dead_static_ball


def test_out_of_pitch_ball_is_dead_evidence():
    scores = evidence.scores(_tracking(200, ball=(1.08, 0.5)), PARAMS, PITCH)
    assert scores["out_of_pitch"][-1]
    assert scores["dead"][-1] >= PARAMS.w_dead_out_of_pitch


def _emissions(home_value, away_value, dead_value=0.0, n=1):
    """Log-probabilities for a block built from constant evidence scores."""
    logits = np.stack([np.full(n, home_value, np.float64),
                       np.full(n, away_value, np.float64),
                       np.full(n, dead_value, np.float64)], axis=1)
    return model.log_softmax(logits / PARAMS.temperature)


def test_hysteresis_ignores_a_short_spike():
    n = 400
    logits = np.zeros((n, 3))
    logits[:, 0] = 3.0  # home is controlling the ball all along
    logits[200:206, 0], logits[200:206, 1] = 0.0, 3.0  # six frames of away evidence
    path = model.viterbi(model.log_softmax(logits), PARAMS.switch_penalty,
                         PARAMS.dead_switch_penalty)
    assert len(model.runs_of(path)) == 1


def test_sustained_evidence_switches_once():
    n = 400
    logits = np.zeros((n, 3))
    logits[:200, 0] = 3.0
    logits[200:, 1] = 3.0
    path = model.viterbi(model.log_softmax(logits), PARAMS.switch_penalty,
                         PARAMS.dead_switch_penalty)
    runs = model.runs_of(path)
    assert [code for _, _, code in runs] == [model.HOME, model.AWAY]
    # the switch is allowed to lag the evidence by the time it takes to pay the penalty
    assert 200 <= runs[1][0] <= 210


def test_neutral_evidence_holds_the_current_state():
    path = model.viterbi(_emissions(0.0, 0.0, n=300), PARAMS.switch_penalty,
                         PARAMS.dead_switch_penalty)
    assert len(model.runs_of(path)) == 1


def test_repair_merges_short_runs_and_bridges_short_stoppages():
    codes = np.full(300, model.HOME, dtype=np.int8)
    codes[100:104] = model.AWAY          # 4-frame flicker
    codes[200:215] = model.DEAD          # 15-frame stoppage, same team either side
    model.repair(codes, [(0, len(codes))], PARAMS)
    assert list(model.runs_of(codes)) == [(0, 300, model.HOME)]


def test_repair_keeps_a_real_stoppage():
    codes = np.full(400, model.HOME, dtype=np.int8)
    codes[200:275] = model.DEAD  # 3 s
    model.repair(codes, [(0, len(codes))], PARAMS)
    assert [code for _, _, code in model.runs_of(codes)] == [model.HOME, model.DEAD,
                                                             model.HOME]


def test_predict_marks_ungraded_frames_and_never_crosses_a_block():
    n = 300
    track = _tracking(n, home_dmin=0.3, away_dmin=9.0)
    scores = evidence.scores(track, PARAMS, PITCH)
    scores["dead"][:] = 0.0
    blocks = [(50, 150), (200, 290)]
    labels, codes = model.predict(scores, blocks, n, PARAMS)
    assert list(labels[:50]) == [model.UNGRADED] * 50
    assert list(labels[150:200]) == [model.UNGRADED] * 50
    assert list(labels[290:]) == [model.UNGRADED] * 10
    assert set(labels[50:150]) == {"H"}
    assert set(labels[200:290]) == {"H"}
    assert len(model.spells(codes, blocks)) == 2


def test_submission_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "possession_metrics", os.path.join(ROOT, "evaluation", "possession_metrics.py"))
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.csv")
        with open(good, "w", newline="") as handle:
            handle.write("frame,label\n")
            handle.writelines("{},X\n".format(i) for i in range(199319))
        frame = metrics.read_label_csv(good)
        assert metrics.validate_submission(frame, 199319) is None

        short = os.path.join(tmp, "short.csv")
        with open(short, "w", newline="") as handle:
            handle.write("frame,label\n")
            handle.writelines("{},H\n".format(i) for i in range(199318))
        frame = metrics.read_label_csv(short)
        assert "row count" in metrics.validate_submission(frame, 199319)


if __name__ == "__main__":
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for name, test in tests:
        try:
            test()
        except AssertionError as error:
            failed += 1
            print("FAIL {} {}".format(name, error or ""))
        else:
            print("ok   {}".format(name))
    print("\n{} of {} passed".format(len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
