"""Every tunable of the possession engine, in one frozen dataclass.

There is no YAML dependency on purpose: the whole config is a JSON file that mirrors this class,
so `--config` is a plain `json.load` and an unknown key is a hard error instead of a silent typo.

Nothing here is calibrated against any ground truth. The values are football priors -- a player has
to be about a stride from the ball to be controlling it, a possession that lasts under two seconds
is usually a tracking artefact rather than a possession -- and the reasoning for each is in the
module that uses it.
"""

import json
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Params:
    # -- evidence scales, in metres on a 105.3 x 68 pitch -------------------------------------
    proximity_scale_m: float = 1.5   # decay of "the nearest player is this close to the ball"
    density_sigma_m: float = 3.0     # Gaussian width of the player-density-around-ball kernel
    near_radius_m: float = 2.0       # "in contact with the ball" radius

    w_proximity: float = 1.0
    w_density: float = 0.8
    w_contact: float = 1.2

    # -- dead-ball evidence -------------------------------------------------------------------
    w_dead_static_ball: float = 2.5
    w_dead_absence: float = 2.0
    w_dead_out_of_pitch: float = 1.5
    w_dead_set_piece: float = 1.0

    static_ball_s: float = 1.0       # ball has moved < static_ball_m over this long -> stoppage
    static_ball_m: float = 2.0
    static_team_m: float = 1.5       # both team centroids move < this over static_ball_s
    absence_dead_s: float = 2.0      # ball unseen this long *and* teams static -> stoppage
    oob_margin: float = 0.03         # normalised coords are reported slightly outside 0..1

    # -- temporal model -----------------------------------------------------------------------
    temperature: float = 0.7         # sharpening of the per-frame softmax
    # A switch has to be paid for in nats, and the exchange rate is evidence per frame. A clear
    # local advantage is worth roughly 3 nats/frame, so 30 nats means "hold your ground unless the
    # other side has had the better of it for about half a second" and 20 nats is the same for a
    # stoppage, which is a smaller claim. Both are deliberately high: a late transition costs a few
    # IoU points, a fragmented possession costs a whole spell.
    switch_penalty: float = 30.0     # nats charged for H <-> A
    dead_switch_penalty: float = 20.0  # nats charged for H/A <-> D

    # -- spell post-processing ----------------------------------------------------------------
    min_spell_s: float = 1.6         # shorter H/A runs are merged into a neighbour
    min_dead_s: float = 1.0          # shorter D runs are dissolved
    bridge_dead_s: float = 2.0       # a stoppage this short between the same team is bridged

    fps: int = 25

    @property
    def min_spell_frames(self):
        return int(round(self.min_spell_s * self.fps))

    @property
    def min_dead_frames(self):
        return int(round(self.min_dead_s * self.fps))

    @property
    def bridge_dead_frames(self):
        return int(round(self.bridge_dead_s * self.fps))

    @property
    def static_window_frames(self):
        return max(2, int(round(self.static_ball_s * self.fps)))

    @property
    def absence_dead_frames(self):
        return int(round(self.absence_dead_s * self.fps))


def load(path=None):
    """Params from a JSON file (unknown keys rejected), or the defaults."""
    if path is None:
        return Params()
    with open(path) as handle:
        raw = json.load(handle)
    known = {f.name for f in fields(Params)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise KeyError("unknown config keys {}; known: {}".format(unknown, sorted(known)))
    return Params(**raw)


def dump(params, path):
    with open(path, "w") as handle:
        json.dump(asdict(params), handle, indent=2, sort_keys=True)
        handle.write("\n")
