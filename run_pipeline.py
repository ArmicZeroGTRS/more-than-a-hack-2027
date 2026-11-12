"""End to end: tactical assets in, validated `frame,label` CSV out.

    python run_pipeline.py                      # defaults, writes submissions/submission.csv
    python run_pipeline.py --config configs/final.json --out submissions/team.csv

Order of operations, and why it is this order
---------------------------------------------
parse once (cached) -> evidence -> Viterbi + spell repair -> CSV -> official validator. The parse is
the only expensive step and it is cached by file size and mtime outside the repository, so iterating
on the model costs seconds rather than a two-minute re-read.

Nothing here reads a ground truth of any kind. The public 5-minute labels are not an input to this
file, and the pipeline is deterministic: same inputs, same bytes out.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from src import atd, config, evidence, model  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "submissions", "submission.csv")
REPORTS = os.path.join(ROOT, "reports")


def cache_dir(override=None):
    """Cache root, outside the repository: the arrays are data-derived and must not be committed."""
    root = override or os.environ.get("MTAH_CACHE_DIR") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "mtah2027", "cache")
    os.makedirs(root, exist_ok=True)
    return root


def load_tracking(paths, params, pitch, n_frames, use_cache=True, cache=None, log=print):
    """Parsed arrays, from cache when the source file and the parse parameters are unchanged."""
    os.makedirs(cache, exist_ok=True)
    stat = os.stat(paths["atd_raw"])
    key = "tracking_{}_{}_{}_{:.2f}_{:.2f}.npz".format(
        stat.st_size, int(stat.st_mtime), n_frames, params.near_radius_m, params.density_sigma_m)
    path = os.path.join(cache, key)

    if use_cache and os.path.isfile(path):
        log("tracking cache: {}".format(path))
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    log("parsing {} ({:.0f} MB) ...".format(os.path.basename(paths["atd_raw"]),
                                            stat.st_size / 1e6))
    started = time.time()
    tracking = atd.load(paths["atd_raw"], n_frames, pitch,
                        atd.team_of_track(paths["atd_meta"]),
                        near_radius_m=params.near_radius_m,
                        density_sigma_m=params.density_sigma_m)
    log("  parsed in {:.1f} s  (malformed lines {}, frames outside range {})".format(
        time.time() - started, tracking["malformed"], tracking["out_of_range"]))
    if use_cache:
        np.savez(path, **tracking)
    return tracking


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path, n_expected):
    """Run the official checks -- the same function the scorer runs, imported, not reimplemented."""
    if os.path.join(ROOT, "evaluation") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics

    frame = possession_metrics.read_label_csv(path)
    return possession_metrics.validate_submission(frame, n_expected)


def diagnostics(labels, spans, tracking, info, params):
    """Structure of our own output. Aggregate only, and written inside the repo's ignored reports/."""
    counts = {name: int(np.sum(labels == name)) for name in ("H", "A", "D", "X")}
    codes = np.full(len(labels), -1, dtype=np.int8)
    for index, name in enumerate(model.NAMES):
        codes[labels == name] = index

    per_half = []
    for half, (start, stop) in enumerate(spans, start=1):
        spells = [(b - a) for a, b, code in model.spells(codes, [(start, stop)])]
        per_half.append({
            "half": half,
            "frames": stop - start,
            "spells": len(spells),
            "mean_spell_s": float(np.mean(spells) / params.fps) if spells else 0.0,
            "median_spell_s": float(np.median(spells) / params.fps) if spells else 0.0,
            "dead_share": float(np.mean(codes[start:stop] == model.DEAD)),
        })

    return {
        "labels": counts,
        "graded_frames": int(sum(stop - start for start, stop in spans)),
        "spells": int(sum(half["spells"] for half in per_half)),
        "per_half": per_half,
        "ball_tracked_share": float(np.mean(np.isfinite(tracking["ball"][:, 0]))),
        "params": {k: v for k, v in info.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="JSON parameter file (see configs/final.json)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the submission CSV")
    parser.add_argument("--cache-dir", help="override the parse cache location")
    parser.add_argument("--no-cache", action="store_true", help="re-parse even if cached")
    args = parser.parse_args(argv)

    started = time.time()
    params = config.load(args.config)
    game = GAME
    pitch = GAMES[game]["pitch"]
    n_frames = source_frames("tactical", game)
    # games.half_video_blocks gives inclusive [first, last]; everything downstream is half-open.
    spans = [(first, last + 1) for _, first, last in half_video_blocks(game)]

    paths = dict(zip(["atd_meta", "atd_raw", "homography", "smart_tagging"],
                     require(["atd_meta", "atd_raw", "homography", "smart_tagging"], game)))
    tracking = load_tracking(paths, params, pitch, n_frames,
                             use_cache=not args.no_cache, cache=cache_dir(args.cache_dir))

    events = None
    try:
        from starter.load_smart_tagging import load as load_smart_tagging
        events = load_smart_tagging(game)
        print("smart tagging: {} instances".format(len(events)))
    except Exception as error:  # optional context only; the pipeline must not depend on it
        print("smart tagging unavailable ({}); continuing without it".format(error))

    print("evidence ...")
    scores = evidence.scores(tracking, params, pitch, set_pieces=events)

    print("temporal model ...")
    labels, codes = model.predict(scores, spans, n_frames, params)

    path = write_submission(args.out, labels)
    reason = validate(path, n_frames)
    print("\nsubmission: {}".format(os.path.relpath(path, ROOT)))
    print("validator : {}".format("PASS" if reason is None else "FAIL -- {}".format(reason)))

    summary = diagnostics(labels, spans, tracking, {
        "game": game,
        "params_file": args.config or "defaults",
        "config": json.loads(json.dumps(params.__dict__)),
    }, params)
    summary["runtime_s"] = round(time.time() - started, 1)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "diagnostics.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("labels    : H {H:,}  A {A:,}  D {D:,}  X {X:,}".format(**summary["labels"]))
    print("spells    : {} total".format(summary["spells"]))
    print("runtime   : {:.1f} s".format(summary["runtime_s"]))
    print("details   : reports/diagnostics.json")
    return 0 if reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
