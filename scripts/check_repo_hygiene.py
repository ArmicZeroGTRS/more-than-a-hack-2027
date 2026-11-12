"""Fail if anything that must stay private is about to enter the public repository.

    python scripts/check_repo_hygiene.py            # every tracked file
    python scripts/check_repo_hygiene.py --staged    # what a commit is about to add

Two rules, both from RULES.md: no match assets, and no derived frame-level exports. The patterns are
the ones `.gitignore` already blocks, re-checked here because `git add -f` and a stale ignore file
both bypass the ignore list, and a leaked file in a public repository cannot be taken back.

Run from the pre-commit hook (`.githooks/pre-commit`) and from CI.
"""

import argparse
import fnmatch
import os
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024

# Matched against the lower-cased path.
FORBIDDEN = (
    "*.mp4", "*.mkv", "*.avi", "*.mov",
    "*_atd.txt", "*_atd.xml",
    "*_homography.csv", "*_smart_tagging.xml",
    "*_possession_gt.csv", "*_possession_gt_meta.json",
    "*.npy", "*.npz", "*.parquet", "*.pkl", "*.pickle",
    "data/*", "submissions/*", "outputs/*", "reports/*",
    "*.pdf",
)

# The shipped baselines are the one deliberate exception: they are computed from tracking alone.
ALLOWED = (
    "data/readme.md",
    "evaluation/results/baseline_*.csv",
    "evaluation/results/readme.md",
)


def _allowed(path):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in ALLOWED)


def _forbidden(path):
    return not _allowed(path) and any(
        fnmatch.fnmatchcase(path, pattern) for pattern in FORBIDDEN)


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name for name in out.decode("utf-8").split("\0") if name]


def staged_files():
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        capture_output=True, check=True).stdout
    return [name for name in out.decode("utf-8").split("\0") if name]


def problems(files, root="."):
    found = []
    for name in files:
        lowered = name.replace("\\", "/").lower()
        if _forbidden(lowered):
            found.append("{}: matches a private-data pattern".format(name))
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES:
            found.append("{}: {:.1f} MB exceeds the {} MB limit".format(
                name, os.path.getsize(path) / 1e6, MAX_BYTES // (1024 * 1024)))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="check only staged files")
    parser.add_argument("--tracked", action="store_true", help="check every tracked file (default)")
    args = parser.parse_args(argv)

    files = staged_files() if args.staged else tracked_files()
    found = problems(files)
    if found:
        print("repository hygiene check FAILED:\n  " + "\n  ".join(found))
        print("\nThese files must not be committed. See RULES.md and .gitignore.")
        return 1
    print("repository hygiene check: OK ({} files)".format(len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
