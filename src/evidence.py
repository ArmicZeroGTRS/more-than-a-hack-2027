"""Turn the parsed tracking into three per-frame scores: Home, Away, dead ball.

Every number here is evidence, not a decision. The scores are handed to the temporal model
(`src/model.py`), which is the only place that chooses a label. Keeping the split is what makes the
system debuggable: when a spell goes wrong you can ask "was the evidence wrong or was the transition
wrong?" and answer it.

The three physical claims the scores encode
------------------------------------------
1. **Possession is a local property of the ball.** The team whose players are nearest the ball, and
   who have the most players around it, is the team controlling it. A single nearest player is not
   enough -- an opponent pressing from 1.5 m away is closer to the ball than the player shielding it
   without being in possession -- so the score is a soft density, not an argmin.
2. **Missing ball is not dead ball.** The ball is invisible on 41% of graded frames and almost none
   of that is a stoppage. Absence therefore produces *neutral* Home/Away evidence, and only becomes
   dead-ball evidence when the players also stop moving.
3. **A stoppage is a shape, not a label.** Ball still (or gone) *and* both team centroids still for a
   second is a restart. A set-piece tag is used as a weak extra vote, never as the answer: the tags
   are automatic, cover only 59 of the match's stoppages, and a tagged phase can contain live play.
"""

import numpy as np

DENSITY_CLIP = 1.0  # density -> density/(1+density), so the term stays in [0, 1)


def _displacement(position, window, scale):
    """Metres moved over `window` frames, NaN where either end is unknown.

    Positions in this dataset are normalised to 0-1 on a pitch that is not square, so a raw
    coordinate difference is not a distance: the x axis has to be scaled by the length and the y
    axis by the width before they can be compared with a metre threshold.
    """
    out = np.full(len(position), np.nan, dtype=np.float32)
    if window < len(position):
        delta = position[window:] - position[:-window]
        out[window:] = np.hypot(delta[:, 0] * scale[0], delta[:, 1] * scale[1])
    return out


def _seen_throughout(valid, window):
    """True where the object was tracked on every one of the last `window` frames (inclusive)."""
    n = len(valid)
    cumulative = np.concatenate(([0], np.cumsum(valid, dtype=np.int32)))
    out = np.zeros(n, dtype=bool)
    if n >= window:
        out[window - 1:] = (cumulative[window:] - cumulative[:n - window + 1]) == window
    return out


def _forward_fill(values):
    """Last known position, held forward. NaN before the first observation."""
    out = values.copy()
    index = np.where(np.isfinite(values[:, 0]), np.arange(len(values)), -1)
    last = np.maximum.accumulate(index)
    seen = last >= 0
    out[seen] = values[last[seen]]
    return out


def _age(valid):
    """Frames since the last observation (0 on a frame that has one)."""
    index = np.where(valid, np.arange(len(valid)), -1)
    last = np.maximum.accumulate(index)
    age = np.arange(len(valid)) - last
    age[last < 0] = len(valid)
    return age


def set_piece_mask(events, n_frames):
    """Frames covered by any SET PIECES annotation, half-open [start, end)."""
    mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        if (event["code"] or "").strip().upper() != "SET PIECES":
            continue
        start = max(0, int(event["start_frame"]))
        stop = min(n_frames, int(event["end_frame"]))
        if stop > start:
            mask[start:stop] = True
    return mask


def scores(tracking, params, pitch, set_pieces=None):
    """Dict of per-frame arrays: `home`, `away`, `dead` scores plus the masks behind them.

    All three are on the same scale, roughly 0-3, so `src/model.py` can treat the sum as a single
    logit vector without per-term rescaling.
    """
    ball = tracking["ball"]
    n = len(ball)
    valid = np.isfinite(ball[:, 0])
    window = params.static_window_frames

    dmin = tracking["dmin"]
    density = tracking["density"]
    contact = tracking["contact"].astype(np.float32)
    density_term = density / (DENSITY_CLIP + density)

    proximity = np.where(np.isfinite(dmin),
                         np.exp(-np.nan_to_num(dmin, nan=0.0) / params.proximity_scale_m),
                         0.0)
    home = (params.w_proximity * proximity[:, 0]
            + params.w_density * density_term[:, 0]
            + params.w_contact * contact[:, 0])
    away = (params.w_proximity * proximity[:, 1]
            + params.w_density * density_term[:, 1]
            + params.w_contact * contact[:, 1])

    # No ball, no proximity evidence: absence must not be resolved by picking whoever happens to be
    # standing nearest an imagined ball. Neutral here lets the temporal model hold its state.
    home[~valid] = 0.0
    away[~valid] = 0.0

    ball_held = _forward_fill(ball)
    contacted = tracking["contact"].any(axis=1)
    # A ball at rest with a player on it is a dribble or a restart being taken, not a stoppage; a
    # ball at rest with nobody near it is the shape of a dead ball. Only the second is evidence.
    ball_still = (_displacement(ball_held, window, pitch) < params.static_ball_m) \
        & _seen_throughout(valid, window) & ~contacted

    centroids = tracking["centroid"]
    team_still = np.ones(n, dtype=bool)
    for team in (0, 1):
        moved = _displacement(centroids[:, team], window, pitch)
        team_still &= ~np.isfinite(moved) | (moved < params.static_team_m)

    age = _age(valid)
    long_absence = age >= params.absence_dead_frames

    out_of_pitch = (ball_held[:, 0] < -params.oob_margin) | (ball_held[:, 0] > 1 + params.oob_margin) \
        | (ball_held[:, 1] < -params.oob_margin) | (ball_held[:, 1] > 1 + params.oob_margin)
    out_of_pitch &= age <= window

    if set_pieces is None:
        tagged = np.zeros(n, dtype=bool)
    else:
        tagged = set_piece_mask(set_pieces, n)

    dead = (params.w_dead_static_ball * ball_still
            + params.w_dead_absence * (long_absence & team_still)
            + params.w_dead_out_of_pitch * out_of_pitch
            + params.w_dead_set_piece * tagged).astype(np.float32)

    return {
        "home": home.astype(np.float32),
        "away": away.astype(np.float32),
        "dead": dead,
        "ball_valid": valid,
        "ball_age": age,
        "ball_still": ball_still,
        "team_still": team_still,
        "out_of_pitch": out_of_pitch,
        "set_piece": tagged,
    }
