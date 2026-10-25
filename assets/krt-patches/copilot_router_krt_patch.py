"""Copilot Router policy on top of KiCad Routing Tools 0.23.7.

The router loads this module before every KRT CLI it runs.  Each block is a
host contract KRT cannot express on its own command line:

* native filled copper written by the host (EasyEDA) is a net-aware obstacle,
  and KRT's #1260 pour guard reads that same fill instead of a KiCad refill;
* exact DSL net names, rip authorizations, power nets and differential pairs
  survive KRT's glob/sheet-leaf selector semantics;
* per-pad QFN fanout opt-outs;
* bounded parameters for the explicitly enabled final rescue.

Retired with 0.23.7, because upstream now does it natively:

* ``SHORT_POWER_EDGE_MM = 0`` and the local neck-down/taper rewrite.  KRT
  #1033 keeps a wide net's width piece by piece (merged back into runs, so a
  taper spans the whole run), lays a short edge at full width and necks only
  to the widest width that fits, and #1318 caps the pad-entry zone at the pad
  instead of narrowing it whole.  Re-applying the old override would both
  regress that and crash: ``_apply_neckdown_widths`` now takes ``neck_w``,
  ``end_caps`` and ``pcb_data``.

Every wrapper forwards unknown positional and keyword arguments, so an
upstream parameter added after 0.23.7 reaches KRT instead of raising.
"""

from __future__ import annotations

import os
import re
import sys
import copy
import json
import inspect
from collections import OrderedDict
from typing import Dict, FrozenSet, List, NamedTuple, Tuple

import numpy as np

import kicad_exact_fill
import obstacle_cache
import obstacle_map
import single_ended_routing
import routing_defaults
from routing_config import DiffPairNet, GridCoord
from routing_utils import build_layer_map


def _require_upstream_contract():
    """Refuse a KRT tree older than the behavior this patch leaves upstream."""
    missing = []
    neckdown = inspect.signature(single_ended_routing._apply_neckdown_widths)
    if "end_caps" not in neckdown.parameters:
        missing.append("_apply_neckdown_widths(end_caps=...) (#1318)")
    for module, name in (
            (single_ended_routing, "_widen_fitting_pieces"),
            (obstacle_map, "_rasterize_polygon_box"),
            (obstacle_map, "_box_masked_cells"),
            (obstacle_cache, "_unique_rows")):
        if not hasattr(module, name):
            missing.append(f"{module.__name__}.{name}")
    try:
        import pour_guard  # noqa: F401  (#1260)
    except ImportError:
        missing.append("pour_guard (#1260)")
    if missing:
        raise RuntimeError(
            "copilot_router_krt_patch requires KiCadRoutingTools >= 0.23.7; "
            "this tree lacks " + ", ".join(missing)
        )


_require_upstream_contract()


def _bounded_rescue_default(name, cast, minimum):
    """Apply a host-owned rescue bound without patching the managed KRT tree."""
    raw = os.environ.get(f"COPILOT_ROUTER_{name}")
    if raw is None:
        return
    try:
        value = cast(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"COPILOT_ROUTER_{name} is invalid: {raw!r}") from error
    if value < minimum:
        raise RuntimeError(f"COPILOT_ROUTER_{name} must be >= {minimum}")
    setattr(routing_defaults, name, value)


_bounded_rescue_default("RESCUE_GRID_STEP", float, 0.01)
_bounded_rescue_default("RESCUE_CLEARANCE_STEPS", int, 1)
_bounded_rescue_default("RESCUE_MAX_WINDOW_CELLS", int, 10_000)
_bounded_rescue_default("RESCUE_MAX_EDGES_PER_NET", int, 1)
_bounded_rescue_default("RESCUE_MAX_ITERATIONS", int, 1_000)


# ----------------------------------------------------------- filled copper
#
# KRT parses zone outlines for connectivity, but its ordinary maze map still
# does not stamp the actual ``filled_polygon`` copper (0.23.7's pour guard
# blocks only uncut islands of out-of-scope nets).  The host writes an
# already-filled board, so its fills augment KRT's existing per-net obstacle
# caches.  The cache of the net being routed is removed by KRT in the normal
# way, so a zone stays conductive for its owner without becoming hundreds of
# fake track terminals.

_ORIGINAL_BUILD_BASE = obstacle_map.build_base_obstacle_map
_ORIGINAL_PRECOMPUTE = obstacle_cache.precompute_net_obstacles
_BUILD_BASE_SIGNATURE = inspect.signature(_ORIGINAL_BUILD_BASE)
_PRECOMPUTE_SIGNATURE = inspect.signature(_ORIGINAL_PRECOMPUTE)
_CACHE_ATTRIBUTE = "_copilot_router_host_fill"
_ROUTER_PLANE_PREFIX = "copilot-router:plane:"
_ZONE_NAME_RE = re.compile(r'\(name\s+"((?:[^"\\]|\\.)*)"\)')
_EMPTY_CELLS = np.empty((0, 3), dtype=np.int32)
_EMPTY_VIAS = np.empty((0, 2), dtype=np.int32)
_ZONE_OBSTACLE_MEMO: "OrderedDict[tuple, tuple]" = OrderedDict()
_ZONE_OBSTACLE_MEMO_SIZE = 64


class _HostFill(NamedTuple):
    by_name: Dict[Tuple[str, str], List[list]]
    by_id: Dict[Tuple[int, str], List[list]]
    plane_uuids: FrozenSet[str]
    has_fill: bool


def _call_arguments(signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _parse_host_fills(text: str):
    """(fills, DSL plane uuids, any fill) from the host-written board.

    DSL planes (``copilot-router:plane:*``) are intentionally cuttable: the
    router pours them again after routing, so they are never obstacles.
    """
    from kicad_parser import uuid_or_tstamp

    id_to_name = {
        int(match.group(1)): match.group(2)
        for match in re.finditer(r'\(net\s+(\d+)\s+"((?:[^"\\]|\\.)*)"\)', text)
    }
    output = {}
    plane_uuids = set()
    has_fill = False
    position = 0
    while True:
        zone_start = text.find("(zone", position)
        if zone_start < 0:
            break
        block, position = kicad_exact_fill._balanced_block(text, zone_start)
        filled_at = block.find("(filled_polygon")
        has_fill = has_fill or filled_at >= 0
        head = block[:filled_at] if filled_at >= 0 else block
        zone_name = _ZONE_NAME_RE.search(head)
        if zone_name and zone_name.group(1).startswith(_ROUTER_PLANE_PREFIX):
            outline_at = [index for index in (head.find("(polygon"), head.find("(keepout"))
                          if index >= 0]
            plane_uuids.add(uuid_or_tstamp(head[:min(outline_at)] if outline_at else head))
            continue
        name_match = kicad_exact_fill._NET_NAME_RE.search(head) \
            or kicad_exact_fill._NET_STR_RE.search(head)
        net_name = name_match.group(1) if name_match else None
        if net_name is None:
            net_match = kicad_exact_fill._NET_ID_RE.search(head)
            if net_match:
                net_name = id_to_name.get(int(net_match.group(1)))
        if not net_name:
            continue
        filled_position = 0
        while True:
            polygon_start = block.find("(filled_polygon", filled_position)
            if polygon_start < 0:
                break
            polygon_block, filled_position = kicad_exact_fill._balanced_block(block, polygon_start)
            layer_match = kicad_exact_fill._LAYER_RE.search(polygon_block)
            if not layer_match:
                continue
            polygon = [
                (float(x), float(y))
                for x, y in kicad_exact_fill._XY_RE.findall(polygon_block)
            ]
            if len(polygon) >= 3:
                output.setdefault((net_name, layer_match.group(1)), []).append(polygon)
    return output, frozenset(plane_uuids), has_fill


def _parse_obstacle_fills(text: str):
    """Parse actual fills, excluding intentionally cuttable DSL planes."""
    return _parse_host_fills(text)[0]


def _net_names(pcb_data) -> Tuple[Dict[int, str], Dict[str, int]]:
    by_id = dict(getattr(pcb_data, "net_id_to_name", None) or {})
    for net_id, net in (getattr(pcb_data, "nets", None) or {}).items():
        name = getattr(net, "name", None)
        if name:
            by_id.setdefault(int(net_id), str(name))
    return by_id, {name: net_id for net_id, name in by_id.items()}


def _host_fill(pcb_data) -> _HostFill:
    cached = getattr(pcb_data, _CACHE_ATTRIBUTE, None)
    if cached is not None:
        return cached
    provider = getattr(pcb_data, "exact_fill_provider", None)
    plane_uuids = frozenset()
    if callable(provider):
        fills = provider() or {}
        has_fill = bool(fills)
    else:
        path = getattr(pcb_data, "source_path", "") or ""
        if not path or not os.path.isfile(path):
            fills, has_fill = {}, False
        else:
            with open(path, "r", encoding="utf-8") as source:
                fills, plane_uuids, has_fill = _parse_host_fills(source.read())
    _by_id, by_name = _net_names(pcb_data)
    indexed = {
        (by_name[name], layer): polygons
        for (name, layer), polygons in fills.items()
        if name in by_name and polygons
    }
    cached = _HostFill(fills, indexed, plane_uuids, has_fill)
    setattr(pcb_data, _CACHE_ATTRIBUTE, cached)
    return cached


def _filled_copper(pcb_data):
    return _host_fill(pcb_data).by_id


def _unique(chunks: List[np.ndarray], columns: int) -> np.ndarray:
    nonempty = [chunk for chunk in chunks if len(chunk)]
    if not nonempty:
        return np.empty((0, columns), dtype=np.int32)
    return obstacle_cache._unique_rows(np.concatenate(nonempty).astype(np.int32, copy=False))


def _zone_obstacles(pcb_data, net_id: int, config, extra_clearance: float = 0.0,
                    tracks: bool = True):
    """(track cells, via cells) of one net's filled copper.

    Memoized on the fill object and every resolved margin, because KRT
    rebuilds base maps and net caches many times per process (rung passes,
    reconciliation laps, fanout probes).  Each entry holds the fill object
    itself, so its id cannot be recycled while the entry lives.  The arrays
    are read-only.
    """
    fills = _filled_copper(pcb_data)
    owned = [(layer, polygons) for (zone_net_id, layer), polygons in fills.items()
             if zone_net_id == net_id]
    if not owned:
        return _EMPTY_CELLS, _EMPTY_VIAS
    layer_map = build_layer_map(config.layers)
    clearance = config.obstacle_clearance(net_id)
    margins = []
    for layer, polygons in owned:
        if layer not in layer_map:
            continue
        layer_clearance = config.layer_clearance(layer, clearance) + extra_clearance
        margins.append((
            layer,
            layer_map[layer],
            config.route_reserve_width(layer) / 2 + layer_clearance,
            config.via_size / 2 + layer_clearance,
            polygons,
        ))
    clip = getattr(pcb_data.board_info, "board_bounds", None)
    key = (
        id(fills), net_id, float(config.grid_step), tuple(config.layers),
        tuple(clip) if clip else None, tracks,
        tuple((layer, track, via) for layer, _index, track, via, _polygons in margins),
    )
    cached = _ZONE_OBSTACLE_MEMO.get(key)
    if cached is not None and cached[0] is fills:
        _ZONE_OBSTACLE_MEMO.move_to_end(key)
        return cached[1]

    coord = GridCoord(config.grid_step)
    track_chunks: List[np.ndarray] = []
    via_chunks: List[np.ndarray] = []
    for _layer, layer_index, track_margin, via_margin, polygons in margins:
        # One margin for both passes, so a via-only rung pass reuses KRT's
        # polygon raster cache entry from the full pass.
        margin = max(track_margin, via_margin, config.grid_step)
        for polygon in polygons:
            gx_lo, gy_lo, nx, _ny, inside, edge_distance = obstacle_map._rasterize_polygon_box(
                polygon, coord, margin, clip_bounds=clip,
            )
            if inside is None:
                continue
            if tracks:
                track_mask = inside | (edge_distance < track_margin)
                if track_mask.any():
                    gx, gy = obstacle_map._box_masked_cells(gx_lo, gy_lo, nx, track_mask)
                    track_chunks.append(np.column_stack([
                        gx, gy, np.full(len(gx), layer_index, dtype=np.int32),
                    ]))
            via_mask = inside | (edge_distance < via_margin)
            if via_mask.any():
                gx, gy = obstacle_map._box_masked_cells(gx_lo, gy_lo, nx, via_mask)
                via_chunks.append(np.column_stack([gx, gy]))
    result = (_unique(track_chunks, 3), _unique(via_chunks, 2))
    for array in result:
        array.setflags(write=False)
    _ZONE_OBSTACLE_MEMO[key] = (fills, result)
    while len(_ZONE_OBSTACLE_MEMO) > _ZONE_OBSTACLE_MEMO_SIZE:
        _ZONE_OBSTACLE_MEMO.popitem(last=False)
    return result


def _append_rows(current: np.ndarray, extra: np.ndarray, columns: int) -> np.ndarray:
    if not len(extra):
        return current
    if not len(current):
        return extra.copy()
    return _unique([current, extra], columns)


def _precompute_net_obstacles(*args, **kwargs):
    data = _ORIGINAL_PRECOMPUTE(*args, **kwargs)
    call = _call_arguments(_PRECOMPUTE_SIGNATURE, args, kwargs)
    # KRT's small/per-net via rungs recurse through this wrapper with
    # _small_pass=True and keep only blocked_vias, so the fill reaches
    # blocked_vias_small and blocked_vias_rungs at each rung's own via size.
    small_pass = bool(call.get("_small_pass", False))
    cells, vias = _zone_obstacles(
        call["pcb_data"], call["net_id"], call["config"],
        call.get("extra_clearance", 0.0), tracks=not small_pass,
    )
    if not small_pass:
        data.blocked_cells = _append_rows(data.blocked_cells, cells, 3)
    data.blocked_vias = _append_rows(data.blocked_vias, vias, 2)
    return data


def _build_base_obstacle_map(*args, **kwargs):
    result = _ORIGINAL_BUILD_BASE(*args, **kwargs)
    call = _call_arguments(_BUILD_BASE_SIGNATURE, args, kwargs)
    pcb_data = call["pcb_data"]
    config = call["config"]
    extra_clearance = call.get("extra_clearance", 0.0)
    static_base = bool(call.get("static_base", False))
    # A via-rung sub-build (#530) contributes only its via cells to the
    # parent map's rung, so skip the track stamps there.
    tracks = not call.get("_rung_pass", False)
    routed = set(call.get("nets_to_route") or [])
    net_ids = sorted({net_id for net_id, _layer in _filled_copper(pcb_data) if net_id not in routed})
    add_cells = (
        result.add_static_blocked_cells_batch
        if static_base and hasattr(result, "add_static_blocked_cells_batch")
        else result.add_blocked_cells_batch
    )
    add_vias = (
        result.add_static_blocked_vias_batch
        if static_base and hasattr(result, "add_static_blocked_vias_batch")
        else result.add_blocked_vias_batch
    )
    for net_id in net_ids:
        cells, vias = _zone_obstacles(pcb_data, net_id, config, extra_clearance, tracks=tracks)
        if len(cells):
            add_cells(cells)
        if len(vias):
            add_vias(vias)
    return result


obstacle_cache.precompute_net_obstacles = _precompute_net_obstacles
obstacle_map.build_base_obstacle_map = _build_base_obstacle_map


def _install_pour_guard_host_fill():
    """Feed KRT's #1260 pour guard the fill the host actually poured.

    Upstream asks a live pcbnew or a KiCad refill for the exact fill, and
    without KiCad (every EasyEDA host) it blocks the whole zone OUTLINE,
    which covers clearance gaps and removed islands too.  The host-written
    ``filled_polygon`` blocks are the same copper this patch stamps as
    obstacles.  DSL planes stay crossable, as everywhere else in the router.
    A static zone the host left unfilled keeps upstream's fallback.
    """
    import pour_guard
    from kicad_parser import pour_zones

    original = pour_guard.fill_islands

    def fill_islands(pcb_data, wanted, *args, **kwargs):
        if getattr(pcb_data, "exact_fill_provider", None) is not None:
            return original(pcb_data, wanted, *args, **kwargs)
        host = _host_fill(pcb_data)
        if not host.has_fill:
            return original(pcb_data, wanted, *args, **kwargs)
        islands = {}
        fallback = {}
        for key in wanted:
            polygons = host.by_name.get(key)
            if polygons:
                islands[key] = [list(polygon) for polygon in polygons]
                continue
            outlines = [
                list(zone.polygon) for zone in pour_zones(pcb_data.zones)
                if zone.layer == key[1] and len(zone.polygon or ()) >= 3
                and zone.uuid not in host.plane_uuids
                and pour_guard._zone_net_name(pcb_data, zone) == key[0]
            ]
            if outlines:
                fallback[key] = outlines
            else:
                islands[key] = []
        source, why = "copilot_host_fill", ""
        if fallback:
            extra, fallback_source, why = original(pcb_data, fallback, *args, **kwargs)
            islands.update(extra)
            source = f"{source}+{fallback_source}"
        return islands, source, why

    pour_guard.fill_islands = fill_islands


_install_pour_guard_host_fill()


# ------------------------------------------------------- differential pairs

_EXPLICIT_DIFF_PAIRS_ENV = "COPILOT_ROUTER_DIFF_PAIRS"


def _parse_explicit_diff_pairs(raw: str):
    """Validate the host-owned exact P/N mapping passed to route_diff.py."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{_EXPLICIT_DIFF_PAIRS_ENV} must contain valid JSON: {error}"
        ) from error
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError(
            f"{_EXPLICIT_DIFF_PAIRS_ENV} must be a non-empty array of [P, N] pairs"
        )

    pairs = []
    owned = set()
    for index, item in enumerate(parsed):
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError(
                f"{_EXPLICIT_DIFF_PAIRS_ENV}[{index}] must be [positive, negative]"
            )
        positive, negative = item
        if not isinstance(positive, str) or not positive.strip():
            raise RuntimeError(
                f"{_EXPLICIT_DIFF_PAIRS_ENV}[{index}][0] must be a non-empty string"
            )
        if not isinstance(negative, str) or not negative.strip():
            raise RuntimeError(
                f"{_EXPLICIT_DIFF_PAIRS_ENV}[{index}][1] must be a non-empty string"
            )
        if positive == negative:
            raise RuntimeError(
                f"{_EXPLICIT_DIFF_PAIRS_ENV}[{index}] must contain distinct nets"
            )
        duplicate = next((name for name in (positive, negative) if name in owned), None)
        if duplicate is not None:
            raise RuntimeError(
                f"{_EXPLICIT_DIFF_PAIRS_ENV} assigns net {duplicate!r} more than once"
            )
        owned.update((positive, negative))
        pairs.append((positive, negative))
    return pairs


def _install_explicit_diff_pairs():
    """Make the router DSL's exact pairs authoritative over KRT name inference.

    route_diff.py imports ``find_differential_pairs`` after this packaged patch
    is loaded.  The wrapper activates only for a call whose selected net names
    contain every explicit member, leaving secondary pattern queries and all
    ordinary KRT invocations on their upstream behavior.
    """
    explicit_pairs = list(_EXACT_SELECTOR_SCOPE.get("diffPairs", ()))
    source = _EXACT_SELECTORS_FILE_ENV
    if not explicit_pairs:
        raw = os.environ.get(_EXPLICIT_DIFF_PAIRS_ENV)
        if not raw:
            return
        explicit_pairs = _parse_explicit_diff_pairs(raw)
        source = _EXPLICIT_DIFF_PAIRS_ENV
    import net_queries

    original = net_queries.find_differential_pairs
    # 0.23.7's route_diff asks once per concern (#1293 coupled floors,
    # layer-scoped widths, the route itself); announce each pairing once.
    announced = set()

    def find_differential_pairs(pcb_data, patterns, *args, **kwargs):
        selected = {str(pattern) for pattern in (patterns or [])}
        # One invocation may legitimately select only one unrelated matched or
        # repair member while the authoritative sidecar describes every pair
        # in the full workflow.  Apply only complete pairs present in this
        # invocation; absence from a scoped membership query is not malformed
        # input and must not force all global pair members into the call.
        applicable_pairs = [
            pair for pair in explicit_pairs
            if pair[0] in selected and pair[1] in selected
        ]
        if not applicable_pairs:
            return original(pcb_data, patterns, *args, **kwargs)

        explicit_members = {name for pair in applicable_pairs for name in pair}

        ids_by_name = {
            str(net.name): int(net_id)
            for net_id, net in pcb_data.nets.items()
            if net_id != 0 and getattr(net, "name", None)
        }
        missing = sorted(explicit_members.difference(ids_by_name))
        if missing:
            raise RuntimeError(
                "Explicit differential-pair nets are missing from the PCB: "
                + ", ".join(missing)
            )

        output = {}
        for index, (positive, negative) in enumerate(applicable_pairs, start=1):
            pair_name = f"COPILOT_EXPLICIT_{index}"
            output[pair_name] = DiffPairNet(
                base_name=pair_name,
                p_net_id=ids_by_name[positive],
                n_net_id=ids_by_name[negative],
                p_net_name=positive,
                n_net_name=negative,
            )
        signature = tuple(applicable_pairs)
        if signature not in announced:
            announced.add(signature)
            print(
                f"Using {len(output)} explicit differential pair(s) from "
                f"{source}"
            )
        return output

    net_queries.find_differential_pairs = find_differential_pairs


# ------------------------------------------------------------- QFN fanout

def _install_qfn_pad_allowlist():
    """Keep QFN geometry intact while suppressing explicitly excluded pads.

    KRT's CLI can filter nets but not individual logical pads.  The router DSL
    needs both component- and pad-level fanout opt-outs, so the adapter passes a
    JSON allowlist.  Disabled pads retain their physical geometry for layout
    analysis and collision checks; only their copied net identity is cleared.

    0.23.7's qfn_fanout CLI grades ``unescaped_nets`` with
    ``bga_fanout.unserved_nets`` on the board's own footprint, so the same
    filtered view goes there too: an excluded pad is not owed copper.
    """
    raw = os.environ.get("COPILOT_ROUTER_QFN_PAD_ALLOWLIST")
    if not raw:
        return
    import qfn_fanout
    import bga_fanout

    parsed = json.loads(raw)
    allowlists = {
        str(component): {str(number) for number in numbers}
        for component, numbers in parsed.items()
        if isinstance(numbers, list)
    }

    def allowed_view(footprint):
        allowed = allowlists.get(str(getattr(footprint, "reference", "")))
        if allowed is None:
            return footprint
        filtered_footprint = copy.copy(footprint)
        filtered_footprint.pads = []
        for source_pad in footprint.pads:
            pad = copy.copy(source_pad)
            if str(getattr(pad, "pad_number", "")) not in allowed:
                pad.net_id = 0
                pad.net_name = ""
            filtered_footprint.pads.append(pad)
        return filtered_footprint

    original_generate = qfn_fanout.generate_qfn_fanout
    original_unserved = bga_fanout.unserved_nets

    def generate_qfn_fanout(footprint, *args, **kwargs):
        return original_generate(allowed_view(footprint), *args, **kwargs)

    def unserved_nets(footprint, *args, **kwargs):
        return original_unserved(allowed_view(footprint), *args, **kwargs)

    qfn_fanout.generate_qfn_fanout = generate_qfn_fanout
    bga_fanout.unserved_nets = unserved_nets


# ---------------------------------------------------------- exact selectors

_EXACT_SELECTORS_FILE_ENV = "COPILOT_ROUTER_EXACT_SELECTORS_FILE"
_EXACT_NET_SENTINEL = "__COPILOT_ROUTER_EXACT_NET_SCOPE_V1__"
_EXACT_RIP_SENTINEL = "__COPILOT_ROUTER_EXACT_RIP_SCOPE_V1__"


def _load_exact_selector_scope():
    path = os.environ.get(_EXACT_SELECTORS_FILE_ENV)
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as source:
        parsed = json.load(source)
    if not isinstance(parsed, dict) or parsed.get("schemaVersion") != 1:
        raise RuntimeError(
            f"{_EXACT_SELECTORS_FILE_ENV} must reference a version-1 JSON object"
        )
    output = {}
    for key in ("netSelection", "ripSelection", "ripAuthorization"):
        values = parsed.get(key, [])
        if not isinstance(values, list) or any(
                not isinstance(name, str) or not name.strip() for name in values):
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.{key} must be an array of non-empty strings"
            )
        output[key] = tuple(dict.fromkeys(name.strip() for name in values))
    selected_names = set(output["netSelection"]) | set(output["ripSelection"])
    sentinels = {}
    for key, default in (("netSentinel", _EXACT_NET_SENTINEL),
                         ("ripSentinel", _EXACT_RIP_SENTINEL)):
        value = parsed.get(key, default)
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.{key} must be a non-empty string"
            )
        if value in selected_names:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.{key} collides with an exact raw net name"
            )
        sentinels[key] = value
    if sentinels["netSentinel"] == sentinels["ripSentinel"]:
        raise RuntimeError(
            f"{_EXACT_SELECTORS_FILE_ENV} must use distinct net and rip sentinels"
        )
    output.update(sentinels)
    selector_tokens = parsed.get("selectorTokens", [])
    if not isinstance(selector_tokens, list):
        raise RuntimeError(
            f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens must be an array of [token, name] pairs"
        )
    normalized_tokens = []
    owned_tokens = set()
    owned_token_names = set()
    for index, item in enumerate(selector_tokens):
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)):
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens[{index}] must be [non-empty token, non-empty name]"
            )
        token, name = item
        if name not in selected_names:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens[{index}] names a net outside the exact selection"
            )
        if token in selected_names:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens[{index}] collides with an exact raw net name"
            )
        if token in owned_tokens or name in owned_token_names:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens must assign one unique token per net"
            )
        owned_tokens.add(token)
        owned_token_names.add(name)
        normalized_tokens.append((token, name))
    if selected_names and owned_token_names != selected_names:
        missing = sorted(selected_names.difference(owned_token_names))
        raise RuntimeError(
            f"{_EXACT_SELECTORS_FILE_ENV}.selectorTokens is missing exact nets: "
            + ", ".join(missing)
        )
    output["selectorTokens"] = tuple(normalized_tokens)
    diff_pairs = parsed.get("diffPairs", [])
    if not isinstance(diff_pairs, list):
        raise RuntimeError(
            f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs must be an array of [P, N] pairs"
        )
    normalized_pairs = []
    owned = set()
    for index, item in enumerate(diff_pairs):
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs[{index}] must be [positive, negative]"
            )
        positive, negative = item
        if not isinstance(positive, str) or not positive.strip():
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs[{index}][0] must be a non-empty string"
            )
        if not isinstance(negative, str) or not negative.strip():
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs[{index}][1] must be a non-empty string"
            )
        positive = positive.strip()
        negative = negative.strip()
        if positive == negative:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs[{index}] must contain distinct nets"
            )
        duplicate = next((name for name in (positive, negative) if name in owned), None)
        if duplicate is not None:
            raise RuntimeError(
                f"{_EXACT_SELECTORS_FILE_ENV}.diffPairs assigns net {duplicate!r} more than once"
            )
        owned.update((positive, negative))
        normalized_pairs.append((positive, negative))
    output["diffPairs"] = tuple(normalized_pairs)
    return output


_EXACT_SELECTOR_SCOPE = _load_exact_selector_scope()


def _install_exact_rip_overrides():
    """Preserve exact override authority after host-side fnmatch escaping.

    KRT correctly treats CLI selectors as globs, while protected_nets uses a
    raw token without glob metacharacters as the deliberate permission to rip
    protected copper.  A literal KiCad name such as ``DATA[0]`` therefore must
    travel in two forms: escaped through the CLI selector and raw through this
    private, validated authorization channel.  KiCad-locked copper remains
    non-overridable in upstream filter_rippable_names.
    """
    authorized = set(_EXACT_SELECTOR_SCOPE.get("ripAuthorization", ()))
    # Keep developer-override compatibility without making normal invocations
    # pay Windows' environment-block cost for a JSON array of every net.
    raw = os.environ.get("COPILOT_ROUTER_EXACT_RIP_NETS")
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or any(
                not isinstance(name, str) or not name.strip() for name in parsed):
            raise RuntimeError(
                "COPILOT_ROUTER_EXACT_RIP_NETS must be a JSON array of non-empty strings"
            )
        authorized.update(name.strip() for name in parsed)
    if not authorized:
        return
    import protected_nets
    original = protected_nets.exact_names

    def exact_names(patterns, *args, **kwargs):
        return set(original(patterns, *args, **kwargs)) | authorized

    protected_nets.exact_names = exact_names


def _exact_selector_pairs(key: str):
    raw = os.environ.get(key)
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise RuntimeError(f"{key} must be a JSON array of [pattern, name] pairs")
    output = []
    for index, item in enumerate(parsed):
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)):
            raise RuntimeError(f"{key}[{index}] must be [non-empty pattern, non-empty name]")
        output.append((item[0], item[1]))
    return output


def _literal_net_filter_pattern(name: str):
    pattern = name.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")
    if pattern.startswith("-"):
        pattern = "[-]" + pattern[1:]
    return "\\" + pattern if name.startswith("!") else pattern


def _exact_selector_map(names, sentinel, selector_tokens):
    mapping = {}

    def add(token, resolved):
        resolved = tuple(resolved)
        previous = mapping.get(token)
        if previous is not None and previous != resolved:
            raise RuntimeError(
                "Exact host selector token collision for " + repr(token)
            )
        mapping[token] = resolved

    names = tuple(dict.fromkeys(names))
    if names:
        add(sentinel, names)
    for name in names:
        # KRT passes raw names into recovery filters after expansion. CLI glob
        # spellings use collision-free opaque tokens from the sidecar instead
        # of sharing a namespace with legal raw KiCad names.
        add(name, (name,))
    for token, name in selector_tokens:
        if name in names:
            add(token, (name,))
    return mapping


def _install_exact_net_selection():
    """Make host-owned DSL net names exact, including hierarchical aliases.

    Upstream intentionally lets a bare ``SIG`` pattern also select
    ``/Sheet/SIG``.  That is convenient for the CLI but unsafe for an exact DSL
    and for targeted rip authorization. Calls made with the compact sentinel,
    encoded CLI tokens, or raw names returned by an earlier exact expansion
    are narrowed; every other native KRT query keeps its upstream
    wildcard/sheet-leaf semantics.
    """
    selector_tokens = _EXACT_SELECTOR_SCOPE.get("selectorTokens", ())
    net_selection = _EXACT_SELECTOR_SCOPE.get("netSelection", ())
    rip_selection = _EXACT_SELECTOR_SCOPE.get("ripSelection", ())
    net_sentinel = _EXACT_SELECTOR_SCOPE.get("netSentinel", _EXACT_NET_SENTINEL)
    rip_sentinel = _EXACT_SELECTOR_SCOPE.get("ripSentinel", _EXACT_RIP_SENTINEL)
    selectors = [
        _exact_selector_map(
            net_selection,
            net_sentinel,
            selector_tokens,
        ),
        _exact_selector_map(
            rip_selection,
            rip_sentinel,
            selector_tokens,
        ),
    ]
    owned_tokens = {token for token, _name in selector_tokens}
    if net_selection:
        owned_tokens.add(net_sentinel)
    if rip_selection:
        owned_tokens.add(rip_sentinel)
    # Legacy mappings remain accepted for developer invocations. Precompute
    # them once as token -> immutable exact-name tuple as well.
    for key in ("COPILOT_ROUTER_EXACT_NET_SELECTION",
                "COPILOT_ROUTER_EXACT_RIP_SELECTION"):
        pairs = _exact_selector_pairs(key)
        if pairs:
            mapping = {}
            for pattern, name in pairs:
                mapping[pattern] = (name,)
                mapping[name] = (name,)
                owned_tokens.add(pattern)
            selectors.append(mapping)
    selectors = [mapping for mapping in selectors if mapping]
    if not selectors:
        return

    # A host-owned token must have one meaning across every exact scope.  Do
    # this check before choosing a scope so a colliding legacy developer
    # mapping cannot silently win by list order.
    resolved_by_token = {}
    for mapping in selectors:
        for token, names in mapping.items():
            previous = resolved_by_token.get(token)
            if previous is not None and previous != names:
                raise RuntimeError(
                    "Exact host selector token collision for " + repr(token)
                )
            resolved_by_token[token] = names

    import net_queries
    original_expand = net_queries.expand_net_patterns
    original_matches = net_queries.matches_net_filter
    original_power = net_queries.identify_power_nets

    missing = object()
    resolve_cache = {}
    available_cache = {}

    def resolve(patterns):
        requested = tuple(patterns or ())
        if not requested:
            return None
        cached = resolve_cache.get(requested, missing)
        if cached is not missing:
            return cached
        for mapping in selectors:
            if all(pattern in mapping for pattern in requested):
                ordered = tuple(dict.fromkeys(
                    name for pattern in requested for name in mapping[pattern]
                ))
                result = (ordered, frozenset(ordered))
                resolve_cache[requested] = result
                return result
        resolve_cache[requested] = None
        return None

    def exact_or_native(patterns, context):
        exact = resolve(patterns)
        if exact is None and any(pattern in owned_tokens for pattern in (patterns or ())):
            raise RuntimeError(
                f"Exact host {context} selector mixes an owned token with "
                "patterns outside its exact scope"
            )
        return exact

    def available_names(pcb_data):
        key = id(pcb_data)
        cached = available_cache.get(key)
        if cached is not None and cached[0] is pcb_data:
            return cached[1]
        available = {
            str(net.name) for net in pcb_data.nets.values()
            if getattr(net, "name", None)
        }
        for pads in pcb_data.pads_by_net.values():
            available.update(
                str(pad.net_name) for pad in pads if getattr(pad, "net_name", None)
            )
        frozen = frozenset(available)
        available_cache[key] = (pcb_data, frozen)
        return frozen

    def expand_net_patterns(pcb_data, patterns, *args, **kwargs):
        exact = exact_or_native(patterns, "net")
        if exact is None:
            return original_expand(pcb_data, patterns, *args, **kwargs)
        ordered, exact_set = exact
        absent = sorted(exact_set.difference(available_names(pcb_data)))
        if absent:
            raise RuntimeError("Exact host net selector is absent from the PCB: " + ", ".join(absent))
        return list(ordered)

    def matches_net_filter(net_name, patterns, *args, **kwargs):
        exact = exact_or_native(patterns, "net filter")
        if exact is None:
            return original_matches(net_name, patterns, *args, **kwargs)
        return net_name in exact[1]

    def identify_power_nets(pcb_data, patterns, widths, *args, **kwargs):
        # The adapter sends --power-nets as opaque exact tokens, which
        # upstream's plain fnmatch never matches: every power width was
        # silently dropped.  Resolve each token to its literal name, keeping
        # upstream's first-pattern-wins order.
        if exact_or_native(patterns, "power-net") is None:
            return original_power(pcb_data, patterns, widths, *args, **kwargs)
        if len(patterns) != len(widths):
            raise ValueError(
                f"patterns ({len(patterns)}) and widths ({len(widths)}) must have same length"
            )
        width_by_name = {}
        for pattern, width in zip(patterns, widths):
            for name in resolve((pattern,))[0]:
                width_by_name.setdefault(name, width)
        return {
            net_id: width_by_name[net.name]
            for net_id, net in pcb_data.nets.items()
            if net_id != 0 and getattr(net, "name", None) in width_by_name
        }

    net_queries.expand_net_patterns = expand_net_patterns
    net_queries.matches_net_filter = matches_net_filter
    net_queries.identify_power_nets = identify_power_nets

    # Length matching has its own glob matcher and routing_common imports it
    # into a module-local alias.  Resolve the same opaque selector tokens here
    # so names containing glob syntax (for example DATA[0]) remain literal.
    import length_matching
    original_length_match_finder = length_matching.find_nets_matching_patterns

    def find_nets_matching_patterns(all_net_names, patterns, *args, **kwargs):
        exact = exact_or_native(patterns, "length-match")
        if exact is None:
            return original_length_match_finder(all_net_names, patterns, *args, **kwargs)
        ordered, _exact_set = exact
        available = frozenset(str(name) for name in all_net_names)
        # Unlike expand_net_patterns(), this matcher receives a dynamic
        # universe rather than every net present in the PCB.  Upstream calls it
        # both with the successfully routed subset during length matching and
        # with a one-item list when asking whether the current Phase-3 net is a
        # member of a matched group.  Requiring every exact group member to be
        # present therefore turns an ordinary partial route (and even a simple
        # membership query) into a process-killing contract error.
        #
        # Keep host-owned names literal and ordered, but intersect them with
        # the caller's current universe.  PCB-level absence is still rejected
        # by expand_net_patterns() above, where the complete PCB is available.
        return [name for name in ordered if name in available]

    length_matching.find_nets_matching_patterns = find_nets_matching_patterns
    # Rebind aliases in modules that imported the originals before this
    # patch. Modules imported later receive the patched attributes normally.
    rebound = {
        "expand_net_patterns": expand_net_patterns,
        "matches_net_filter": matches_net_filter,
        "identify_power_nets": identify_power_nets,
        "find_nets_matching_patterns": find_nets_matching_patterns,
    }
    originals = {
        "expand_net_patterns": original_expand,
        "matches_net_filter": original_matches,
        "identify_power_nets": original_power,
        "find_nets_matching_patterns": original_length_match_finder,
    }
    for module_name, module in list(sys.modules.items()):
        # The namespace itself, never getattr: a lazy module's __getattr__
        # (scipy's) must not import anything on our behalf.
        namespace = getattr(module, "__dict__", None)
        if namespace is None or module_name in ("net_queries", "length_matching"):
            continue
        for name, replacement in rebound.items():
            if namespace.get(name) is originals[name]:
                namespace[name] = replacement


_install_exact_net_selection()
_install_exact_rip_overrides()
_install_explicit_diff_pairs()
_install_qfn_pad_allowlist()
