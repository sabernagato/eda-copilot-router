"""Fast regression for the managed KRT 0.23.7 Copilot Router patch.

Runs against a real KiCadRoutingTools tree on a small synthetic board, so the
wrappers are exercised with upstream's own call signatures.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace


if len(sys.argv) != 2:
    raise SystemExit("usage: check-krt-zone-patch.py <KiCadRoutingTools-directory>")

root = os.path.abspath(sys.argv[1])
repository = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.dont_write_bytecode = True
sys.path[:0] = [
    os.path.join(repository, "assets", "krt-patches"),
    os.path.join(root, "py_router"),
    os.path.join(root, "rust_router"),
]

BOARD = """(kicad_pcb (version 20240108) (generator "copilot-router-check")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (net 0 "")
  (net 1 "GND")
  (net 2 "VCC")
  (net 3 "SIG")
  (gr_rect (start 0 0) (end 20 20) (stroke (width 0.1) (type default)) (layer "Edge.Cuts")
    (uuid "00000000-0000-0000-0000-000000000001"))
  (footprint "check:TWO_PAD" (layer "F.Cu") (uuid "00000000-0000-0000-0000-000000000010") (at 4 4)
    (property "Reference" "U1" (at 0 -2 0) (layer "F.SilkS") (uuid "00000000-0000-0000-0000-000000000011"))
    (pad "1" smd rect (at -1 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "GND") (uuid "00000000-0000-0000-0000-000000000012"))
    (pad "2" smd rect (at 1 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "GND") (uuid "00000000-0000-0000-0000-000000000013")))
  (footprint "check:TWO_PAD" (layer "F.Cu") (uuid "00000000-0000-0000-0000-000000000020") (at 4 8)
    (property "Reference" "U2" (at 0 -2 0) (layer "F.SilkS") (uuid "00000000-0000-0000-0000-000000000021"))
    (pad "1" smd rect (at -1 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "GND") (uuid "00000000-0000-0000-0000-000000000022"))
    (pad "2" smd rect (at 1 0) (size 0.6 0.6) (layers "F.Cu") (net 1 "GND") (uuid "00000000-0000-0000-0000-000000000023")))
  (footprint "check:ONE_PAD" (layer "F.Cu") (uuid "00000000-0000-0000-0000-000000000030") (at 16 4)
    (property "Reference" "J1" (at 0 -2 0) (layer "F.SilkS") (uuid "00000000-0000-0000-0000-000000000031"))
    (pad "1" smd rect (at 0 0) (size 0.6 0.6) (layers "F.Cu") (net 3 "SIG") (uuid "00000000-0000-0000-0000-000000000032"))
    (pad "2" smd rect (at 0 12) (size 0.6 0.6) (layers "F.Cu") (net 3 "SIG") (uuid "00000000-0000-0000-0000-000000000033")))
  (zone (net 1) (net_name "GND") (layer "F.Cu") (uuid "00000000-0000-0000-0000-000000000100")
    (name "user-pour") (hatch edge 0.5) (connect_pads (clearance 0.2)) (min_thickness 0.2)
    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))
    (polygon (pts (xy 9 9) (xy 15 9) (xy 15 15) (xy 9 15)))
    (filled_polygon (layer "F.Cu") (pts (xy 10 10) (xy 14 10) (xy 14 14) (xy 10 14))))
  (zone (net 2) (net_name "VCC") (layer "B.Cu") (uuid "00000000-0000-0000-0000-000000000200")
    (name "copilot-router:plane:0:VCC:B.Cu") (hatch edge 0.5) (connect_pads (clearance 0.2)) (min_thickness 0.2)
    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5))
    (polygon (pts (xy 1 1) (xy 19 1) (xy 19 19) (xy 1 19)))
    (filled_polygon (layer "B.Cu") (pts (xy 2 2) (xy 18 2) (xy 18 18) (xy 2 18))))
)
"""

workspace = tempfile.mkdtemp(prefix="copilot-krt-patch-")
board_path = os.path.join(workspace, "check.kicad_pcb")
with open(board_path, "w", encoding="utf-8") as handle:
    handle.write(BOARD)

NET_TOKENS = {"GND": "__TOKEN_GND__", "VCC": "__TOKEN_VCC__", "SIG": "__TOKEN_SIG__"}
NET_SENTINEL = "__COPILOT_ROUTER_EXACT_NET_SCOPE_V1__"
selector_path = os.path.join(workspace, "selectors.json")
with open(selector_path, "w", encoding="utf-8") as handle:
    json.dump({
        "schemaVersion": 1,
        "netSelection": ["GND", "VCC", "SIG"],
        "ripSelection": [],
        "ripAuthorization": ["SIG"],
        "diffPairs": [],
        "selectorTokens": [[token, name] for name, token in NET_TOKENS.items()],
    }, handle)
os.environ["COPILOT_ROUTER_EXACT_SELECTORS_FILE"] = selector_path
os.environ["COPILOT_ROUTER_QFN_PAD_ALLOWLIST"] = json.dumps({"U1": ["1"]})

# Originals, captured before the patch rebinds them, for the alias scan.
import length_matching  # noqa: E402
import net_queries  # noqa: E402
originals = {
    id(net_queries.expand_net_patterns): "expand_net_patterns",
    id(net_queries.matches_net_filter): "matches_net_filter",
    id(net_queries.identify_power_nets): "identify_power_nets",
    id(length_matching.find_nets_matching_patterns): "find_nets_matching_patterns",
}

import copilot_router_krt_patch as patch  # noqa: E402
import bga_fanout  # noqa: E402
import obstacle_cache  # noqa: E402
import obstacle_map  # noqa: E402
import pour_guard  # noqa: E402
import protected_nets  # noqa: E402
import single_ended_routing  # noqa: E402
from kicad_parser import parse_kicad_pcb  # noqa: E402
from routing_config import GridRouteConfig  # noqa: E402

# --- neck-down is upstream's (#1033/#1318) --------------------------------
assert single_ended_routing._apply_neckdown_widths.__module__ == "single_ended_routing", \
    "the retired neck-down override is still installed"
assert single_ended_routing.SHORT_POWER_EDGE_MM == 10.0, \
    "short power edges must keep upstream's #1033 width ladder"

# --- filled copper on upstream's real map and cache -----------------------
pcb = parse_kicad_pcb(board_path)
config = GridRouteConfig(
    track_width=0.2, clearance=0.2, via_size=0.6, via_drill=0.3,
    grid_step=0.1, layers=["F.Cu", "B.Cu"],
)
assert obstacle_map.build_base_obstacle_map.__module__ == "copilot_router_krt_patch"
assert obstacle_cache.precompute_net_obstacles.__module__ == "copilot_router_krt_patch"

# stamp_pads is what 0.23.7's qfn/bga fanout pass; a fixed wrapper signature
# raised TypeError there.
base = obstacle_map.build_base_obstacle_map(pcb, config, [3], stamp_pads=False)
assert base.is_blocked(120, 120, 0), "GND fill interior is not an obstacle on F.Cu"
assert base.is_via_blocked(120, 120), "a via may land inside the GND fill"
assert not base.is_blocked(120, 120, 1), "F.Cu fill leaked onto B.Cu"
assert not base.is_blocked(50, 160, 1), "the cuttable DSL plane became an obstacle"

cache = obstacle_cache.precompute_net_obstacles(pcb, 1, config)
assert any((row == [120, 120, 0]).all() for row in cache.blocked_cells), \
    "GND net cache is missing its fill"
small = obstacle_cache.precompute_net_obstacles(pcb, 1, config, _small_pass=True)
assert any((row == [120, 120]).all() for row in small.blocked_vias), \
    "via-rung pass is missing the fill"
assert not len(patch._zone_obstacles(pcb, 3, config)[0]), "wrong net inherited the zone cache"

first = patch._zone_obstacles(pcb, 1, config)
again = patch._zone_obstacles(pcb, 1, config)
assert first is again, "zone obstacles were not memoized"
assert not first[0].flags.writeable, "memoized cells must be read-only"
extra = patch._zone_obstacles(pcb, 1, config, extra_clearance=0.1)
assert len(extra[0]) > len(first[0]), "memo ignored the extra clearance"

provider_pcb = SimpleNamespace(
    net_id_to_name={1: "VCC", 2: "SIG"},
    nets={},
    exact_fill_provider=lambda: {
        ("VCC", "F.Cu"): [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]],
    },
    board_info=SimpleNamespace(board_bounds=(-1.0, -1.0, 3.0, 3.0)),
)
cells, vias = patch._zone_obstacles(provider_pcb, 1, config)
assert len(cells) and len(vias) and set(cells[:, 2]) == {0}

parsed = patch._parse_obstacle_fills(BOARD)
assert ("GND", "F.Cu") in parsed
assert ("VCC", "B.Cu") not in parsed, "cuttable DSL plane became a fixed obstacle"

# --- pour guard reads the host fill -------------------------------------
outline = [(9.0, 9.0), (15.0, 9.0), (15.0, 15.0), (9.0, 15.0)]
islands, source, _why = pour_guard.fill_islands(
    pcb, {("GND", "F.Cu"): [outline], ("VCC", "B.Cu"): [outline]},
)
assert source == "copilot_host_fill", source
assert islands[("GND", "F.Cu")] == [[(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0)]], \
    "pour guard did not use the host-written fill"
assert islands[("VCC", "B.Cu")] == [], "pour guard would block a cuttable DSL plane"

# --- exact selectors ---------------------------------------------------
ids = {net.name: net_id for net_id, net in pcb.nets.items() if net.name}
assert net_queries.identify_power_nets(pcb, [NET_TOKENS["VCC"]], [0.8]) == {ids["VCC"]: 0.8}, \
    "power-net widths were dropped for exact tokens"
assert net_queries.identify_power_nets(pcb, ["VC?"], [0.8]) == {ids["VCC"]: 0.8}
assert net_queries.expand_net_patterns(pcb, [NET_SENTINEL], quiet=True) == ["GND", "VCC", "SIG"]
assert net_queries.matches_net_filter("SIG", [NET_TOKENS["SIG"]])
assert not net_queries.matches_net_filter("/Sheet/SIG", [NET_TOKENS["SIG"]])
assert "SIG" in protected_nets.exact_names(["GND*"])

stale = sorted(
    f"{module_name}.{name}"
    for module_name, module in list(sys.modules.items())
    if module is not None
    for name, value in list(getattr(module, "__dict__", {}).items())
    if id(value) in originals and module_name not in ("net_queries", "length_matching")
)
assert not stale, f"modules still hold unpatched selectors: {stale}"

# --- QFN allowlist reaches 0.23.7's unescaped accounting -----------------
def track_at(reference):
    pad = next(pad for pad in pcb.footprints[reference].pads if pad.pad_number == "1")
    return {"start": (pad.global_x, pad.global_y), "end": (pad.global_x - 1.0, pad.global_y),
            "layer": "F.Cu", "net_id": pad.net_id}


assert bga_fanout.unserved_nets(pcb.footprints["U1"], pcb, ["GND"], [track_at("U1")], [], 0.2) == [], \
    "an excluded QFN pad was graded as unescaped"
assert bga_fanout.unserved_nets(pcb.footprints["U2"], pcb, ["GND"], [track_at("U2")], [], 0.2) == ["GND"], \
    "the allowlist leaked onto a footprint it does not name"

print(
    f"KRT 0.23.7 Copilot patch: {len(first[0])} fill track cells, {len(first[1])} via cells; "
    "upstream neck-down, host-fill pour guard, exact power nets, QFN allowlist ok"
)
