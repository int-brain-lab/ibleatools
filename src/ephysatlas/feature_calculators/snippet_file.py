"""Feature calculator for snippet files written by ``ephysatlas.snippet_extractor``.

The file holds short AP and LF snippets of one or more recordings as int16 with the gains
to convert them to volts. NP2 LF is derived during extraction; this reader loads the stored
samples for both bands. The snippets of the selected recording are the only windows that
can be read.

Functions
---------
list_recordings
    Names of the recordings in a snippet file.

Classes
-------
SnippetFileFeatureCalculator
    Compute OOP features from the snippets stored in one HDF5 snippet file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import ibldsp.voltage
import numpy as np
import pandas as pd
import scipy.stats

from ephysatlas.feature_computation import add_target_coordinates

from .aggregate import compute_aggregated_features
from .base import BaseFeatureCalculator
from .types import FeatureComputationOptions, RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


def list_recordings(h5_file: str | Path) -> list[str]:
    """List the recordings stored in a snippet file.

    Args:
        h5_file (str | Path): HDF5 file written by ``extract_snippets``.

    Returns:
        list[str]: Recording names, in h5py's (alphabetical) order.

    Raises:
        ValueError: If the file has no recordings group.
    """
    with h5py.File(h5_file, "r") as f:
        if not isinstance(f.get("recordings"), h5py.Group):
            raise ValueError(f"{h5_file} is not a snippet file")
        return list(f["recordings"])


class SnippetFileFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for one recording of a snippet file.

    Args:
        h5_file (str | Path): HDF5 file written by ``extract_snippets``.
        recording (str, optional): Recording to read. It is required when the file holds more
            than one recording.
        shank (int, optional): Shank to read. It is required when the file holds more than
            one shank, and then only that shank's channels are returned.
        name (str, optional): Recording identifier used as the OOP ``pid`` in outputs.
            Defaults to ``<file stem>_<recording>``, with ``_shank<N>`` appended when ``shank``
            is given.
        channel_metadata (pd.DataFrame, optional): Channel metadata returned as it is by
            ``load_channel_metadata``, e.g. with histology. Defaults to a table built from
            the stored geometry.

    Attributes:
        recording (str): Name of the recording being read.
        windows (list[SnippetWindow]): One window per stored snippet.
        trajectory (dict | None): Planned trajectory stored in the file, if any.

    Raises:
        ValueError: If the file has no recordings group, holds several recordings
            and ``recording`` is not given, does not hold the requested recording, holds several
            shanks and ``shank`` is not given, does not hold the requested shank, or the
            selected recording is missing AP or LF data.
    """

    def __init__(
        self,
        h5_file: str | Path,
        recording: str | None = None,
        shank: int | None = None,
        name: str | None = None,
        channel_metadata: pd.DataFrame | None = None,
    ) -> None:
        self.h5_file = Path(h5_file)
        self._channel_metadata = channel_metadata
        recordings = list_recordings(self.h5_file)
        if recording is None:
            if len(recordings) != 1:
                raise ValueError(
                    f"{self.h5_file} holds recordings {recordings}; "
                    "pass recording to select one"
                )
            recording = recordings[0]
        elif recording not in recordings:
            raise ValueError(
                f"{self.h5_file} has no recording {recording!r}; it holds {recordings}"
            )
        self.recording = recording
        # Only the small arrays are read here. The snippets are read in load_raw_snippet.
        with h5py.File(self.h5_file, "r") as file:
            f = file["recordings"][recording]
            # Extraction always stores both bands, including derived LF for NP2.
            for band in ("ap", "lf"):
                if band not in f:
                    raise ValueError(
                        f"{self.h5_file} recording {recording!r} is missing "
                        f"required band {band!r}"
                    )
            neuropixel_version = int(f.attrs["neuropixel_version"])
            self._duration = float(f.attrs["duration"])
            self._probe_model = str(f.attrs["probe_model"]) or pd.NA
            self._referencing_scheme = str(f.attrs["referencing_scheme"]) or pd.NA
            self._t_start = f["t_start"][:]
            self._geometry = {key: f["geometry"][key][:] for key in f["geometry"]}
            self._bands = {
                band: {
                    "fs": float(f[band].attrs["fs"]),
                    "sample2volts": f[band]["sample2volts"][:],
                }
                for band in ("ap", "lf")
            }
            self._histology = (
                {
                    key: f["histology/channels"][key].asstr()[:]
                    if key == "acronym"
                    else f["histology/channels"][key][:]
                    for key in ("x", "y", "z", "atlas_id", "acronym")
                }
                if "channels" in f.get("histology", {})
                else None
            )
            self.trajectory = (
                {key: float(value) for key, value in f["trajectory"].attrs.items()}
                if "trajectory" in f
                else None
            )

        shanks = np.unique(self._geometry["shank"])
        if shank is None and len(shanks) > 1:
            raise ValueError(
                f"{self.h5_file} recording {self.recording!r} holds shanks "
                f"{shanks.tolist()}; pass shank to select one"
            )
        if shank is not None and shank not in shanks:
            raise ValueError(
                f"{self.h5_file} recording {self.recording!r} has no shank {shank}; "
                f"it holds {shanks.tolist()}"
            )
        # Rows of the stored arrays that belong to the selected shank.
        self._rows = (
            np.arange(len(self._geometry["shank"]))
            if shank is None
            else np.flatnonzero(self._geometry["shank"] == shank)
        )

        default_name = f"{self.h5_file.stem}_{self.recording}" + (
            "" if shank is None else f"_shank{shank}"
        )
        super().__init__(
            name=name or default_name, neuropixel_version=neuropixel_version
        )
        self._vote_labels = None
        self.windows = [
            SnippetWindow(float(t), self._duration, self._duration)
            for t in self._t_start
        ]

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Read one stored snippet and convert it to volts.

        Args:
            window (SnippetWindow): One of ``self.windows``.

        Returns:
            RawSnippet: Stored AP and LF arrays shaped ``(channels, samples)`` in volts,
            with their sampling rates. LF sample positions already include the extraction
            offset; no further shift or derivation is applied here.

        Raises:
            ValueError: If the window is not one of the stored snippets.
        """
        match = np.isclose(self._t_start, window.t_start)
        if not (
            match.any()
            and np.isclose(self._duration, window.duration_ap)
            and np.isclose(self._duration, window.duration_lf)
        ):
            raise ValueError(
                f"Window {window} is not stored in {self.h5_file} recording "
                f"{self.recording!r}. Stored windows: "
                f"{self.windows}"
            )
        index = int(np.flatnonzero(match)[0])

        raw = {}
        with h5py.File(self.h5_file, "r") as file:
            f = file["recordings"][self.recording]
            for band, info in self._bands.items():
                # Same conversion as spikeglx.Reader.read, so the volts are identical.
                data = f[band]["data"][index][self._rows].astype(np.float32)
                data *= info["sample2volts"][self._rows, np.newaxis]
                raw[band] = data
        return RawSnippet(
            raw_ap=raw["ap"],
            raw_lf=raw["lf"],
            fs_ap=self._bands["ap"]["fs"],
            fs_lf=self._bands["lf"]["fs"],
        )

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Return the stored geometry of the selected channels.

        Returns:
            dict[str, np.ndarray]: The spikeglx geometry keys (``adc``, ``col``, ``flag``,
            ``ind``, ``row``, ``sample_shift``, ``shank``, ``x``, ``y``).
        """
        return {key: values[self._rows] for key, values in self._geometry.items()}

    def load_channel_metadata(self) -> pd.DataFrame:
        """Build channel metadata from the stored geometry.

        Returns:
            pd.DataFrame: The ``channel_metadata`` given at init if any. Otherwise columns
            ``channel``, ``rawInd`` (the channel's index in the original file),
            ``axial_um``, ``lateral_um`` and ``shank``, as in
            ``SpikeGLXFileFeatureCalculator.load_channel_metadata``, plus ``x``, ``y``,
            ``z`` (metres), ``atlas_id`` and ``acronym`` when the recording has a channel
            table. Only the selected shank's rows are returned.
        """
        if self._channel_metadata is not None:
            return self._channel_metadata
        geometry = self.load_geometry()
        channels = pd.DataFrame(
            {
                "channel": np.arange(len(geometry["x"])),
                "rawInd": geometry["ind"],
                "axial_um": np.asarray(geometry["y"], dtype=float),
                "lateral_um": np.asarray(geometry["x"], dtype=float),
                "shank": np.asarray(geometry["shank"], dtype=float),
            }
        )
        if self._histology is not None:
            # The table has one row per data row, so the shank's rows are taken the same way.
            channels = channels.assign(
                **{key: values[self._rows] for key, values in self._histology.items()}
            )
        return channels

    def available_duration(self) -> tuple[float, float]:
        """Return the end of the last stored snippet for AP and LF.

        Returns:
            tuple[float, float]: The end time of the last snippet for each band, measured
            from the original recording's start. Gaps between snippets are not readable.
        """
        end = float(self._t_start.max() + self._duration)
        return end, end

    def _resolve_channel_labels(
        self,
        raw: RawSnippet,
        channels: pd.DataFrame,
        channel_labels: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return channel labels, voting over all stored snippets when none are given.

        The SpikeGLX calculator labels bad channels once per recording (majority vote over
        10 short pieces). Here the base class would label every snippet on its own, so the
        labels are voted over the stored snippets instead, once per calculator.

        Args:
            raw (RawSnippet): Snippet being computed. Not used for the vote.
            channels (pd.DataFrame): Channel metadata; a ``labels`` or ``channel_labels``
                column takes priority over the vote.
            channel_labels (np.ndarray, optional): Explicit labels, returned as they are.

        Returns:
            np.ndarray: Labels per selected channel.
        """
        if channel_labels is not None:
            return channel_labels
        for column in ("labels", "channel_labels"):
            if column in channels.columns:
                return channels[column].to_numpy()
        if self._vote_labels is None:
            per_snippet = []
            for window in self.windows:
                snippet = self.load_raw_snippet(window)
                labels, _ = ibldsp.voltage.detect_bad_channels(
                    snippet.raw_ap, fs=snippet.fs_ap
                )
                per_snippet.append(labels)
            self._vote_labels = scipy.stats.mode(
                np.stack(per_snippet), axis=0, keepdims=False
            ).mode
        return self._vote_labels

    def enrich_channel_metadata(
        self, channels: pd.DataFrame, options: FeatureComputationOptions
    ) -> pd.DataFrame:
        """Add probe metadata and, if the file stores one, trajectory target coordinates.

        Args:
            channels (pd.DataFrame): Channel metadata.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: ``channels`` with ``probe_model`` and ``referencing_scheme``
            columns, plus ``x_target``, ``y_target`` and ``z_target`` when a trajectory is
            stored and ``options.include_trajectory`` is set.

        Raises:
            ValueError: If the trajectory is required but the file has none.
        """
        # "string" dtype keeps these columns concat-compatible across probes.
        channels = channels.assign(
            probe_model=pd.Series(
                self._probe_model, index=channels.index, dtype="string"
            ),
            referencing_scheme=pd.Series(
                self._referencing_scheme, index=channels.index, dtype="string"
            ),
        )
        if not options.include_trajectory:
            return channels
        if self.trajectory is None:
            if options.require_trajectory:
                raise ValueError(
                    f"No trajectory stored in {self.h5_file} recording {self.recording!r}"
                )
            LOGGER.info(
                "No trajectory stored in %s recording %r", self.h5_file, self.recording
            )
            return channels
        channel_dict = {column: channels[column].to_numpy() for column in channels}
        return pd.DataFrame(
            add_target_coordinates(channels=channel_dict, traj_dict=self.trajectory)
        )

    def compute_denoised_features(
        self,
        n_snippets: int | None = None,
        duration_ap: float | None = None,
        duration_lf: float | None = None,
        options: FeatureComputationOptions | None = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Compute, aggregate and denoise features over the stored snippets.

        Only the stored snippets can be read, so the arguments of the base method are
        accepted for compatibility and must match the file if given.

        Args:
            n_snippets (int, optional): Must equal the number of stored snippets.
            duration_ap (float, optional): Must equal the stored snippet duration.
            duration_lf (float, optional): Must equal the stored snippet duration.
            options (FeatureComputationOptions, optional): Options of every snippet computation.
            n_jobs (int): Snippets computed in parallel.

        Returns:
            pd.DataFrame: As ``BaseFeatureCalculator.compute_denoised_features``.

        Raises:
            ValueError: If an argument differs from what the file holds.
        """
        if n_snippets is not None and n_snippets != len(self.windows):
            raise ValueError(
                f"{self.h5_file} recording {self.recording!r} holds {len(self.windows)} "
                f"snippets, not {n_snippets}"
            )
        for duration in (duration_ap, duration_lf):
            if duration is not None and not np.isclose(duration, self._duration):
                raise ValueError(
                    f"{self.h5_file} recording {self.recording!r} holds {self._duration} s "
                    f"snippets, not {duration} s"
                )
        return compute_aggregated_features(self, self.windows, options, n_jobs=n_jobs)
