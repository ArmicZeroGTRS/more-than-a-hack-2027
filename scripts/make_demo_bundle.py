"""Build a small synthetic bundle so the pipeline can be previewed without the gated data.

Not competition data: two clubs, two track slots, 600 frames of invented geometry. Enough to exercise
every stage end to end and produce a validator-PASS file.

    python scripts/make_demo_bundle.py <target-dir>
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

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


def write(folder, name, text):
    path = os.path.join(folder, name)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def main(target):
    tactical = os.path.join(target, "tactical")
    os.makedirs(tactical, exist_ok=True)
    write(tactical, "BAR-ATH_tactical_atd.xml", META)

    lines = []
    # 600 frames inside the graded blocks: home on the ball in the first half, away in the second
    for frame in list(range(11186, 11486)) + list(range(110809, 111109)):
        if frame < 100000:
            players, ball = "0.500,0.500,0.0;0.600,0.500,0.0", "0.510,0.500,0.0"
        else:
            players, ball = "0.900,0.500,0.0;0.700,0.500,0.0", "0.710,0.500,0.0"
        lines.append("{}:{}:{}".format(frame, players, ball))
    write(tactical, "BAR-ATH_tactical_atd.txt", "\n".join(lines) + "\n")

    write(tactical, "BAR-ATH_tactical_homography.csv",
          "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
          "11185,1,0,0,0,1,0,0,0,1,0.2,1\n")

    write(tactical, "BAR-ATH_tactical_smart_tagging.xml",
          '<?xml version="1.0"?><file><ALL_INSTANCES><instance>'
          "<start>447.44</start><end>448.44</end><code>SET PIECES</code>"
          "<label><group>Type</group><text>Kick Off (Start)</text></label>"
          "</instance></ALL_INSTANCES></file>")
    print("synthetic bundle: {}".format(tactical))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "mtah-synth"))
