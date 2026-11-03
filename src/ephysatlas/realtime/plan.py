"""The planned trajectory of an insertion, as the channel positions the localizer starts from.

The ephys-only localizer needs the planned position of every channel before the recording
starts. They come from one of three places:

- :func:`plan_from_release` -- the ``x/y/z_target`` columns of the feature release (what the
  offline evaluation of the localizer used; for insertions in the release);
- :func:`plan_from_alyx` -- the insertion's best Alyx trajectory (micro-manipulator, else
  planned, else histology track) through :func:`ephysatlas.feature_computation.add_target_coordinates`;
- :func:`plan_from_trajectory` -- coordinates typed in by the experimenter (IBL convention: entry
  ``x``, ``y``, ``z`` in µm from bregma, ``depth`` in µm, ``theta``, ``phi`` in degrees, Needles
  frame, as Alyx stores micro-manipulator trajectories).

:func:`save_plan` writes a small JSON file that the SpikeGLX and Open Ephys tools read
(:func:`load_plan`). Positions are in metres (IBL-Allen coordinates, origin bregma), one row per
channel in row order: top of the probe first, row ``r`` is channel ``C - 1 - r``.
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def load_plan(path) -> dict:
    """Read a plan file: the JSON fields, with ``planned_xyz_m`` as a ``[C, 3]`` array."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path} is not an {PLAN_FORMAT} file")
    doc["planned_xyz_m"] = np.asarray(doc["planned_xyz_m"], dtype=np.float64)
    return doc
