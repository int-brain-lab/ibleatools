"""The planned trajectory of an insertion, as the channel positions the localizer starts from.

The ephys-only localizer starts from the planned position of every channel (or, without a plan,
searches the whole brain: :mod:`ephysatlas.realtime.whole_brain`). They come from one of four
places:

- :func:`plan_from_release` -- the ``x/y/z_target`` columns of the feature release (what the
  offline evaluation of the localizer used; for insertions in the release);
- :func:`plan_from_alyx` -- the insertion's best Alyx trajectory (micro-manipulator, else
  planned, else histology track) through :func:`ephysatlas.feature_computation.add_target_coordinates`;
- :func:`plan_from_trajectory` -- coordinates typed in by the experimenter (IBL convention: entry
  ``x``, ``y``, ``z`` in µm from bregma, ``depth`` in µm, ``theta``, ``phi`` in degrees, Needles
  frame, as Alyx stores micro-manipulator trajectories);
- :func:`plan_from_pinpoint` -- one probe of a `Pinpoint <https://pinpoint.allenneuraldynamics.org>`_
  v5 experiment file (the JSON Pinpoint exports), as the Open Ephys plugin reads it.

:func:`save_plan` writes a small JSON file that the SpikeGLX and Open Ephys tools read
(:func:`load_plan`, which also reads Pinpoint files). Positions are in metres (IBL-Allen
coordinates, origin bregma), one row per channel in row order: top of the probe first, row ``r``
is channel ``C - 1 - r``. A plan of another probe than a Neuropixels 1.0 (a Pinpoint
Neuropixels 2.0, say) also has ``heights_um``: the height above the tip of each position (IBL
convention, the tip-most site row at 20 µm). The localizer reads a plan as the straight line of
the shank it localizes (:meth:`ephysatlas.realtime.probe.ProbeLayout.plan_rows`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

PLAN_FORMAT = "ephysatlas-plan/1"


def _top_first(channel_xyz: np.ndarray) -> np.ndarray:
    """``[C, 3]`` positions in channel order (channel 0 at the tip) -> row order (top first)."""
    return np.asarray(channel_xyz, dtype=np.float64)[::-1].copy()


def plan_from_release(pid: str, vintage: Optional[str] = None) -> dict:
    """The planned channel positions of an insertion of the feature release."""
    from ephysatlas.alignment.data import load_probe
    from ephysatlas.alignment.models import VINTAGE

    probe = load_probe(str(pid), vintage or VINTAGE)
    if probe is None:
        raise KeyError(f"{pid} is not in the feature release {vintage or VINTAGE}")
    return dict(
        pid=str(pid),
        source=f"feature release {vintage or VINTAGE} (x/y/z_target)",
        planned_xyz_m=np.asarray(probe["planned_xyz"], dtype=np.float64),
    )


def plan_from_alyx(pid: str, one=None, axial_um: Optional[np.ndarray] = None) -> dict:
    """The planned channel positions from the insertion's best Alyx trajectory.

    Args:
        pid: Insertion id.
        one: ``one.api.ONE`` instance (created when None).
        axial_um: ``[C]`` axial position of each channel (channel order, µm from the tip);
            Neuropixels 1.0 by default.
    """
    from ephysatlas.feature_computation import add_target_coordinates

    if one is None:
        from one.api import ONE

        one = ONE(base_url="https://alyx.internationalbrainlab.org")
    axial = _np1_axial_um() if axial_um is None else np.asarray(axial_um, dtype=float)
    channels = add_target_coordinates(
        pid=str(pid), one=one, channels=dict(axial_um=axial)
    )
    xyz = np.column_stack(
        [channels["x_target"], channels["y_target"], channels["z_target"]]
    )
    return dict(pid=str(pid), source="alyx trajectory", planned_xyz_m=_top_first(xyz))


def plan_from_trajectory(
    x: float,
    y: float,
    z: float,
    depth: float,
    theta: float,
    phi: float,
    axial_um: Optional[np.ndarray] = None,
) -> dict:
    """The planned channel positions of a typed-in trajectory (IBL Needles-frame convention).

    Args:
        x, y, z: Entry point, µm from bregma (ML, AP, DV).
        depth: Insertion depth along the probe, µm.
        theta: Polar angle from vertical, degrees.
        phi: Azimuth, degrees.
        axial_um: ``[C]`` channel axial positions (channel order); Neuropixels 1.0 by default.
    """
    from ephysatlas.feature_computation import add_target_coordinates

    traj = dict(x=x, y=y, z=z, depth=depth, theta=theta, phi=phi)
    axial = _np1_axial_um() if axial_um is None else np.asarray(axial_um, dtype=float)
    channels = add_target_coordinates(traj_dict=traj, channels=dict(axial_um=axial))
    xyz = np.column_stack(
        [channels["x_target"], channels["y_target"], channels["z_target"]]
    )
    return dict(
        pid="",
        source="typed-in trajectory",
        trajectory=traj,
        planned_xyz_m=_top_first(xyz),
    )


def _np1_axial_um() -> np.ndarray:
    import neuropixel

    return np.asarray(neuropixel.trace_header(version=1)["y"], dtype=float)


def save_plan(plan: dict, path) -> Path:
    """Write a plan (from one of the ``plan_from_*`` functions) to a JSON file."""
    xyz = np.asarray(plan["planned_xyz_m"], dtype=np.float64)
    doc = dict(
        format=PLAN_FORMAT,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        pid=plan.get("pid", ""),
        source=plan.get("source", ""),
        trajectory=plan.get("trajectory"),
        coordinates="IBL-Allen, metres, origin bregma (x ML, y AP, z DV)",
        row_order="top of the probe first: row r is channel n_channels - 1 - r",
        n_channels=int(len(xyz)),
        planned_xyz_m=xyz.round(9).tolist(),
    )
    if plan.get("heights_um") is not None:
        doc["heights_um"] = np.asarray(plan["heights_um"], dtype=float).round(3).tolist()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def load_plan(path, probe: Optional[str] = None, shank: Optional[int] = None) -> dict:
    """Read a plan file, or one probe of a Pinpoint v5 experiment (see :func:`plan_from_pinpoint`,
    which takes ``probe`` and ``shank``): the JSON fields, with ``planned_xyz_m`` as a ``[C, 3]``
    array (and ``heights_um`` as a ``[C]`` array, or None)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if _is_pinpoint_v5(doc):
        return plan_from_pinpoint(path, probe, shank=shank)
    if doc.get("format") != PLAN_FORMAT:
        raise ValueError(
            f"{path} is neither an {PLAN_FORMAT} file nor a Pinpoint v5 experiment"
        )
    doc["planned_xyz_m"] = np.asarray(doc["planned_xyz_m"], dtype=np.float64)
    heights = doc.get("heights_um")
    doc["heights_um"] = None if heights is None else np.asarray(heights, dtype=float)
    return doc


# ---------------------------------------------------------------------------------------------
# Pinpoint v5 experiments (as Source/Localization/Realtime/PlanFile.cpp of the Open Ephys plugin)
# ---------------------------------------------------------------------------------------------

# Bregma of the allen_mouse atlas in Pinpoint's ASR frame (mm from the atlas origin: AP, DV, ML),
# that of iblatlas, and the scaling of the qiu2018_mouse (MRI) atlas to the Allen CCF about bregma
# (ML, AP, DV), that of iblatlas' MRITorontoAtlas.
PINPOINT_ALLEN_BREGMA_MM = (5.400, 0.332, 5.739)
PINPOINT_QIU2018_SCALE = (0.952, 1.031, 0.885)


def _is_pinpoint_v5(doc) -> bool:
    return (
        isinstance(doc, dict)
        and "probes" in doc
        and "probeInterfaceProbes" in doc
        and "atlas" in doc
    )


def pinpoint_probe_names(path) -> list:
    """The probe names of a Pinpoint v5 experiment file (empty for any other file)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p["name"] for p in doc["probes"]] if _is_pinpoint_v5(doc) else []


def plan_from_pinpoint(
    path, probe: Optional[str] = None, n_channels: int = 384, shank: Optional[int] = None
) -> dict:
    """The planned channel positions of one probe of a Pinpoint v5 experiment file.

    Pinpoint keeps the tip in atlas ASR mm (AP, DV, ML from the atlas origin) and the rotation as
    ``[roll, yaw, pitch]`` (rad; Babylon ``RotationYawPitchRoll``), under which the probe's local
    +z, up the shank from the tip, is ``(cos yaw cos pitch, -sin pitch, sin yaw cos pitch)`` in
    ASR. A contact sits its probeinterface ``y`` above the bottom of the probe contour, the tip.
    The ``allen_mouse`` atlas is the CCF, read from the IBL bregma; the ``qiu2018_mouse`` (MRI)
    atlas is read from its own bregma and scaled to the CCF as iblatlas' MRITorontoAtlas.

    A Neuropixels 1.0 gives its 384 bank-0 channels (as the Open Ephys plugin). Any other probe
    (a Neuropixels 2.0, say) gives one position per contact height of the shank ``shank``, with
    their ``heights_um`` above the tip (IBL convention: tip-most row at 20 µm), for
    :meth:`~ephysatlas.realtime.probe.ProbeLayout.plan_rows`. On a probe of several shanks the
    positions are those of the line through Pinpoint's tip: the localized shank's lateral offset
    from the shank Pinpoint aligns (``shankAlignmentIndex``, default 0) is not applied (the
    plan's ``warning`` says so).

    Args:
        path: The experiment file.
        probe: The probe's name (e.g. ``"Probe A"``); may be omitted when the file holds one.
        n_channels: Channels of the recording (bank 0 of a Neuropixels 1.0: 384).
        shank: The shank localized (default 0), for a probe other than a Neuropixels 1.0.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not _is_pinpoint_v5(doc):
        raise ValueError(f"{path} is not a Pinpoint v5 experiment")
    probes = doc["probes"]
    names = [p["name"] for p in probes]
    chosen = [
        p for p in probes if (p["name"] == probe if probe else len(probes) == 1)
    ]
    if not chosen:
        if not probes:
            raise ValueError(f"{path} has no probes")
        raise ValueError(
            f"{path} has {len(probes)} probes ({', '.join(names)}): choose one"
            if not probe
            else f"{path} has no probe {probe} ({', '.join(names)})"
        )
    p = chosen[0]

    atlas = doc["atlas"]["name"]
    scale = np.ones(3)
    if atlas.startswith("allen_mouse"):
        bregma = np.array(PINPOINT_ALLEN_BREGMA_MM)
    elif atlas.startswith("qiu2018_mouse"):
        manifest = doc["atlas"].get("manifest", {})
        bregma = np.asarray(
            manifest.get("bregma") or doc["referenceCoordinate"], dtype=float
        )
        scale = np.array(PINPOINT_QIU2018_SCALE)
    else:
        raise ValueError(
            f"its atlas is {atlas}: plan in the allen_mouse or qiu2018_mouse atlas"
        )
    if bregma.shape != (3,):
        raise ValueError(f"no bregma for the {atlas} atlas")

    definition = doc["probeInterfaceProbes"][p["probeInterfaceIdentifier"]]
    annotations = definition.get("annotations", {}) or {}
    model = annotations.get("model_name")
    to_mm = {"m": 1000.0, "mm": 1.0}.get(definition.get("si_units", "um"), 1e-3)
    contacts = np.asarray(definition["contact_positions"], dtype=float)
    shank_ids = np.asarray(
        [str(s) for s in definition.get("shank_ids") or [""] * len(contacts)]
    )
    shanks = sorted({s for s in shank_ids if s != ""}, key=_shank_key)
    tips = annotations.get("shank_tips") or []
    n_shanks = max(len(shanks), len(tips), 1)
    np1 = (model is None or str(model).upper().startswith(("NP1", "PRB_1"))) and (
        n_shanks == 1
    )
    if np1 and len(contacts) < n_channels:
        raise ValueError(
            f"{p['name']} has {len(contacts)} contacts, the recording {n_channels} channels"
        )
    tip = np.asarray(p["tipPosition"], dtype=float)
    _, yaw, pitch = np.asarray(p["rotation"], dtype=float)
    up = np.array(
        [np.cos(yaw) * np.cos(pitch), -np.sin(pitch), np.sin(yaw) * np.cos(pitch)]
    )
    aligned = int(p.get("shankAlignmentIndex") or 0)
    warning = ""
    if np1:
        tip_y = min(float(q[1]) for q in definition["probe_planar_contour"])
        height = (contacts[:n_channels, 1] - tip_y) * to_mm  # channel order, mm above tip
        heights_um = None
    else:
        shank = int(shank or 0)
        if shanks:
            if shank >= len(shanks):
                raise ValueError(
                    f"{p['name']} ({model}) has {len(shanks)} shanks, not a shank {shank}"
                )
            on = shank_ids == shanks[shank]
        else:
            on = np.ones(len(contacts), dtype=bool)
        if aligned < len(tips):
            tip_y = float(tips[aligned][1])
        else:
            tip_y = min(float(q[1]) for q in definition["probe_planar_contour"])
        ys = np.unique(contacts[on, 1])[::-1]  # contact heights of the shank, top first
        height = (ys - tip_y) * to_mm
        heights_um = (ys - ys.min()) * to_mm * 1e3 + 20.0
        if n_shanks > 1 and shank != aligned:
            offset = (
                abs(float(tips[shank][0]) - float(tips[aligned][0])) * to_mm * 1e3
                if max(shank, aligned) < len(tips)
                else float("nan")
            )
            warning = (
                f"{p['name']} is a {n_shanks}-shank {model}: the plan is the line through "
                f"Pinpoint's tip (shank {aligned}); shank {shank}'s lateral offset "
                f"({offset:.0f} µm) is not applied"
            )
    d = tip[None, :] + height[:, None] * up[None, :] - bregma[None, :]  # ASR mm from bregma
    xyz = np.column_stack(
        [d[:, 2] / scale[0], -d[:, 0] / scale[1], -d[:, 1] / scale[2]]
    ) * 1e-3
    return dict(
        pid="",
        source=f"Pinpoint {doc.get('version', 'v5')}, {p['name']}"
        + ("" if np1 else f" ({model}, shank {shank})"),
        probe=p["name"],
        planned_xyz_m=_top_first(xyz) if np1 else xyz,
        heights_um=heights_um,
        warning=warning,
    )


def _shank_key(s: str):
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)
