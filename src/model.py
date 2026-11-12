"""The temporal layer: per-frame evidence in, possession spells out.

Two stages, both offline and deterministic.

1. **Viterbi over three states.** The evidence is a per-frame logit vector; a switch costs a fixed
   number of nats. A frame that is simply uncertain contributes the same amount to every state, so
   the most probable path keeps the current label and the cost is only paid when the evidence
   actually favours a different team for long enough. That is hysteresis expressed as a model
   rather than as two thresholds, and it makes the outcome independent of the order frames are
   visited in.

2. **Spell repair.** A possession is a maximal H or A run; a 6-frame flicker between two sides costs
   two spells in `spellF1` while gaining nothing at frame level. Short runs are therefore merged into
   the longer neighbour, and a short stoppage flanked by the same team is bridged rather than
   allowed to cut one possession in two. Both rules only ever remove structure, never invent it.

The two halves are separate sequences. Halftime is 20,156 frames of X and no possession crosses it,
so smoothing is applied per graded block, never across.
"""

import numpy as np

HOME, AWAY, DEAD = 0, 1, 2
NAMES = ("H", "A", "D")
UNGRADED = "X"


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def emissions(evidence, params):
    """(N,3) log probabilities from the three evidence scores."""
    logits = np.stack([evidence["home"], evidence["away"], evidence["dead"]], axis=1)
    return log_softmax(logits.astype(np.float64) / params.temperature)


def viterbi(log_probability, switch_penalty, dead_penalty):
    """Best state path. Same-state costs nothing; H<->A and H/A<->D cost different amounts."""
    n = len(log_probability)
    labels = np.empty(n, dtype=np.int8)
    back = np.zeros((n, 3), dtype=np.int8)

    p_home, p_away, p_dead = log_probability[0]

    for t in range(1, n):
        e_home, e_away, e_dead = log_probability[t]

        # to HOME: stay, or come from AWAY (switch) / DEAD (restart)
        best_home, from_home = p_home, 0
        candidate = p_away - switch_penalty
        if candidate > best_home:
            best_home, from_home = candidate, 1
        candidate = p_dead - dead_penalty
        if candidate > best_home:
            best_home, from_home = candidate, 2

        best_away, from_away = p_away, 1
        candidate = p_home - switch_penalty
        if candidate > best_away:
            best_away, from_away = candidate, 0
        candidate = p_dead - dead_penalty
        if candidate > best_away:
            best_away, from_away = candidate, 2

        best_dead, from_dead = p_dead, 2
        candidate = p_home - dead_penalty
        if candidate > best_dead:
            best_dead, from_dead = candidate, 0
        candidate = p_away - dead_penalty
        if candidate > best_dead:
            best_dead, from_dead = candidate, 1

        back[t] = (from_home, from_away, from_dead)
        p_home = best_home + e_home
        p_away = best_away + e_away
        p_dead = best_dead + e_dead

    final = (p_home, p_away, p_dead)
    labels[n - 1] = int(np.argmax(final))
    for t in range(n - 2, -1, -1):
        labels[t] = back[t + 1, labels[t + 1]]
    return labels


def runs_of(codes):
    """[(start, stop, code)] maximal constant runs of a 1-D array."""
    if not len(codes):
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)])).tolist()
    return [(bounds[i], bounds[i + 1], int(codes[bounds[i]])) for i in range(len(bounds) - 1)]


def _repair_block(codes, params):
    """One pass of merge-short-runs. Returns True when something changed."""
    runs = runs_of(codes)
    changed = False
    for index, (start, stop, code) in enumerate(runs):
        length = stop - start
        left = runs[index - 1] if index > 0 else None
        right = runs[index + 1] if index + 1 < len(runs) else None

        if code == DEAD:
            # A stoppage flanked by the same team is a tracking break, not a stoppage: bridging it
            # keeps one possession whole, which is worth more than the D frame it costs.
            same_team = left and right and left[2] == right[2] and left[2] != DEAD
            too_short = length < params.min_dead_frames
            bridged = same_team and length < params.bridge_dead_frames
            if too_short or bridged:
                target = _neighbour_label(left, right, code)
                if target is not None:
                    codes[start:stop] = target
                    changed = True
            continue

        if length < params.min_spell_frames:
            target = _neighbour_label(left, right, code)
            if target is not None:
                codes[start:stop] = target
                changed = True
    return changed


def _neighbour_label(left, right, current):
    """Label of the longer adjacent run, or the only adjacent one. Never the current label."""
    options = [run for run in (left, right)
               if run is not None and run[2] != current]
    if not options:
        return None
    return max(options, key=lambda run: (run[1] - run[0], run[0]))[2]


def repair(codes, blocks, params, max_passes=8):
    """Merge short runs inside each graded block until stable. `codes` is modified in place."""
    for _ in range(max_passes):
        changed = False
        for start, stop in blocks:
            window = codes[start:stop]
            if len(window) and _repair_block(window, params):
                codes[start:stop] = window
                changed = True
        if not changed:
            break
    return codes


def predict(evidence, blocks, n_frames, params):
    """Full label array: Viterbi + repair per graded block, X everywhere else."""
    log_probability = emissions(evidence, params)
    codes = np.full(n_frames, -1, dtype=np.int8)

    for start, stop in blocks:
        window = log_probability[start:stop]
        if not len(window):
            continue
        codes[start:stop] = viterbi(window, params.switch_penalty, params.dead_switch_penalty)

    repair(codes, blocks, params)

    labels = np.full(n_frames, UNGRADED, dtype="<U1")
    inside = codes >= 0
    labels[inside] = np.array(NAMES, dtype="<U1")[codes[inside]]
    return labels, codes


def spells(codes, blocks):
    """[(start, stop, code)] possession spells (H or A only), per block."""
    out = []
    for start, stop in blocks:
        for a, b, code in runs_of(codes[start:stop]):
            if code in (HOME, AWAY):
                out.append((start + a, start + b, code))
    return out
