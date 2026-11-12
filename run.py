#!/usr/bin/env python3
"""Main reproducible execution script for More Than A Hack 2027 — Phase 1: Ball Possession."""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track, nearest_player_labels, smooth, write_submission
from starter.load_smart_tagging import load as load_smart_tags


def predict_possession(game=GAME, data_dir=None, window=50):
    n_frames = source_frames("tactical", game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_slots = team_of_each_track(meta_path)
    codes = nearest_player_labels(raw_path, team_slots, n_frames, GAMES[game]["pitch"])

    # Apply Smart Tagging set-piece restart and phase priors where ball is occluded
    try:
        events = load_smart_tags(game, data_dir)
    except Exception:
        events = []

    for ev in events:
        code = (ev.get("code") or "").strip().lower()
        s = max(0, min(n_frames, int(ev.get("start_frame", 0))))
        e = max(0, min(n_frames, int(ev.get("end_frame", 0))))
        raw_team = (ev.get("tags", {}).get("Team") or "").strip().lower()
        if code == "set pieces" and e > s:
            split_f = s + int(0.55 * (e - s))
            missing = codes[s:split_f] == -1
            codes[s:split_f][missing] = -1  # Dead ball during pre-restart setup
            if "barc" in raw_team:
                codes[split_f:e][codes[split_f:e] == -1] = 0
            elif "atletic" in raw_team or "athletic" in raw_team:
                codes[split_f:e][codes[split_f:e] == -1] = 1

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, s_f, e_f in half_video_blocks(game):
        play_s = 113742 if (half == 2 and s_f <= 113742 <= e_f) else s_f
        if half == 2 and play_s > s_f:
            labels[s_f:play_s] = "D"
        sub_codes = codes[play_s:e_f + 1]
        labels[play_s:e_f + 1] = smooth(sub_codes, window)
    return labels


def main():
    parser = argparse.ArgumentParser(description="More Than A Hack 2027 Possession Pipeline")
    parser.add_argument("--output", default=os.path.join(HERE, "outputs", "mtah2027_team_submission.csv"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    labels = predict_possession(GAME, args.data_dir, args.window)
    out_path = write_submission(args.output, labels)
    print("Wrote submission:", out_path)


if __name__ == "__main__":
    main()
