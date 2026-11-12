"""Stream the tactical ATD text export into compact per-frame arrays.

The text file is 199,319 lines of `frame:p1;p2;...;p46:ball`, each group `x,y,speed` with `NaN`
for a missing observation, coordinates normalised to 0-1. Only x and y are read here: the speed
field has no documented unit in the export (see reports/AUDIT.md, uncertainty 4), and every speed
this pipeline needs is measurable from positions at a known 25 fps, so nothing depends on it.

What is kept per frame, all `float32` except the counts:

    ball         (N,2)  normalised position, NaN where the ball was not tracked
    dmin         (N,2)  metres from the ball to the nearest Home / Away player
    density      (N,2)  sum of exp(-(d/sigma)^2) over each team's players, i.e. players near the ball
    contact      (N,2)  a player of that team is within `near_radius_m`
    centroid     (N,2,2) mean position of each team, normalised, NaN where nobody was seen
    count        (N,2)  how many players of each team were seen

Nothing here decides possession. It is a loader that refuses to lose frames silently: a malformed
line is counted and reported, never dropped quietly.
"""

import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
MISSING = "N"  # first character of "NaN"

HOME, AWAY = 0, 1


def team_names(meta_path):
    """(home, away) club names from the FIFA/EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def team_of_track(meta_path):
    """int8 array over track slots: 0 = home, 1 = away.

    The first `<Team>` in the file is the home side -- the FIFA/EPTS convention, and the one
    `starter/baseline.py` uses. Checked against `team_names()` at call sites so a reordered export
    is a loud failure rather than a silently mirrored match.

    A slot is a tracklet, not a person: 46 slots for 22 players, `ShirtNumber` is the slot index,
    and slots swap between people. That is why every feature below is a team aggregate.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def load(raw_path, n_frames, pitch, team_table, near_radius_m=2.0, density_sigma_m=3.0,
         progress=None):
    """Every per-frame array listed in the module docstring, as a dict."""
    team_table = np.asarray(team_table, dtype=np.int8)
    n_teams = len(team_table)
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    dmin = np.full((n_frames, 2), np.nan, dtype=np.float32)
    density = np.zeros((n_frames, 2), dtype=np.float32)
    contact = np.zeros((n_frames, 2), dtype=bool)
    centroid = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    count = np.zeros((n_frames, 2), dtype=np.int8)

    near_sq = near_radius_m * near_radius_m
    inv_sigma_sq = 1.0 / (density_sigma_m * density_sigma_m)

    malformed = 0
    out_of_range = 0
    n_slots = None

    with open(raw_path) as handle:
        for line, row in enumerate(handle, 1):
            parts = row.rstrip("\r\n").split(":")
            if len(parts) != 3:
                malformed += 1
                continue
            try:
                frame = int(parts[0])
            except ValueError:
                malformed += 1
                continue
            t = frame - 1  # ATD frame_id counts from 1, submissions from 0
            if not 0 <= t < n_frames:
                out_of_range += 1
                continue

            groups = parts[1].split(";")
            if n_slots is None:
                n_slots = len(groups)
            elif len(groups) != n_slots:
                malformed += 1

            ball_group = parts[2]
            if ball_group[:1] == MISSING:
                ball_x = ball_y = np.nan
            else:
                ball_x, ball_y, _ = ball_group.split(",")
                ball_x = float(ball_x)
                ball_y = float(ball_y)
                ball[t] = (ball_x, ball_y)

            sum_x = [0.0, 0.0]
            sum_y = [0.0, 0.0]
            best = [np.inf, np.inf]
            found = [0, 0]

            for slot, group in enumerate(groups):
                if group[:1] == MISSING:
                    continue
                x, y, _ = group.split(",")
                x = float(x)
                y = float(y)
                team = team_table[slot] if slot < n_teams else -1
                if team < 0:
                    continue
                found[team] += 1
                sum_x[team] += x
                sum_y[team] += y

                if ball_x != ball_x:  # NaN ball: no distances to measure
                    continue
                dx = (x - ball_x) * length
                dy = (y - ball_y) * width
                d_sq = dx * dx + dy * dy
                if d_sq < best[team]:
                    best[team] = d_sq
                density[t, team] += np.exp(-d_sq * inv_sigma_sq)
                if d_sq <= near_sq:
                    contact[t, team] = True

            for team in (HOME, AWAY):
                if found[team]:
                    count[t, team] = found[team]
                    centroid[t, team] = (sum_x[team] / found[team], sum_y[team] / found[team])
                if best[team] < np.inf:
                    dmin[t, team] = np.sqrt(best[team])

            if progress and line % 25000 == 0:
                progress("  {:,} lines".format(line))

    return {
        "ball": ball,
        "dmin": dmin,
        "density": density,
        "contact": contact,
        "centroid": centroid,
        "count": count,
        "malformed": malformed,
        "out_of_range": out_of_range,
        "n_slots": n_slots,
    }
