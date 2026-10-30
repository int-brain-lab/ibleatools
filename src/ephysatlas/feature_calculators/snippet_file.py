"""Feature calculator for snippet files written by ``ephysatlas.snippet_extractor``.

The file holds a few short raw AP (and, for NP1, LF) snippets of one probe as int16 with the
gains to convert them to volts. The snippets are the only windows that can be read.

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
import scipy.signal
import scipy.stats

from ephysatlas.feature_computation import add_target_coordinates

from .aggregate import compute_aggregated_features
from .base import BaseFeatureCalculator
from .types import FeatureComputationOptions, RawSnippet, SnippetWindow

LOGGER = logging.getLogger(__name__)


def derive_lf(ap: np.ndarray, fs_ap: float) -> tuple[np.ndarray, float]:
    """Derive the LF band from NP2 wideband AP data.

    Args:
        ap (np.ndarray): AP data in volts, shaped ``(channels, samples)``.
        fs_ap (float): AP sampling rate in Hz (30 kHz for NP2).

    Returns:
        tuple[np.ndarray, float]: float32 LF data shaped ``(channels, samples // 12)``
        and its sampling rate ``fs_ap / 12``.
    """
    # NP2Converter's filter as it is now (neuropixel.py:336, :741). butter() gets no fs, so
    # Wn=0.2 is relative to the 15 kHz Nyquist of the 30 kHz input: the real cutoff is ~3 kHz,
    # not 1 kHz. Kept to match the existing NP2 LF data; see CLAUDE.md, "LF for NP2".
    sos = scipy.signal.butter(N=2, Wn=1000 / 2500 / 2, btype="lowpass", output="sos")
    lf = scipy.signal.sosfiltfilt(sos, ap, axis=-1)[:, ::12]
    return lf.astype(np.float32), fs_ap / 12


class SnippetFileFeatureCalculator(BaseFeatureCalculator):
    """Feature calculator for one snippet file.

    Args:
        h5_file (str | Path): HDF5 file written by ``extract_snippets``.
        shank (int, optional): Shank to read. It is required when the file holds more than
            one shank, and then only that shank's channels are returned.
        name (str, optional): Recording identifier used as the OOP ``pid`` in outputs.
            Defaults to the file stem, with ``_shank<N>`` appended when ``shank`` is given.
        channel_metadata (pd.DataFrame, optional): Channel metadata returned as it is by
            ``load_channel_metadata``, e.g. with histology. Defaults to a table built from
            the stored geometry.

    Attributes:
        windows (list[SnippetWindow]): One window per stored snippet.
        trajectory (dict | None): Planned trajectory stored in the file, if any.

    Raises:
        ValueError: If the file has a format version other than 1, holds several shanks and ``shank`` is not given, or does
            not hold the requested shank.
    """

    def __init__(
        self,
        h5_file: str | Path,
        shank: int | None = None,
        name: str | None = None,
        channel_metadata: pd.DataFrame | None = None,
    ) -> None:
        self.h5_file = Path(h5_file)
        self._channel_metadata = channel_metadata
        # Only the small arrays are read here. The snippets are read in load_raw_snippet.
        with h5py.File(self.h5_file, "r") as f:
            format_version = int(f.attrs["format_version"])
            if format_version != 1:
                raise ValueError(
                    f"{self.h5_file} has format version {format_version}; "
                    "this reader supports version 1"
                )
            neuropixel_version = int(f.attrs["neuropixel_version"])
            self._probe_has_lf_band = bool(f.attrs["probe_has_lf_band"])
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
                if band in f
            }
            self.trajectory = (
                {key: float(value) for key, value in f["trajectory"].attrs.items()}
                if "trajectory" in f
                else None
            )

        shanks = np.unique(self._geometry["shank"])
        if shank is None and len(shanks) > 1:
            raise ValueError(
                f"{self.h5_file} holds shanks {shanks.tolist()}; pass shank to select one"
            )
        if shank is not None and shank not in shanks:
            raise ValueError(
                f"{self.h5_file} has no shank {shank}; it holds {shanks.tolist()}"
            )
        # Rows of the stored arrays that belong to the selected shank.
        self._rows = (
            np.arange(len(self._geometry["shank"]))
            if shank is None
            else np.flatnonzero(self._geometry["shank"] == shank)
        )

        default_name = self.h5_file.stem + ("" if shank is None else f"_shank{shank}")
        super().__init__(
            name=name or default_name, neuropixel_version=neuropixel_version
        )
        self._warned_no_lf = False
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
            RawSnippet: AP and LF arrays shaped ``(channels, samples)`` in volts. The LF
            is the stored one if the file has it. For a probe without an LF band (NP2) it
            is derived from the AP data. ``raw_lf`` is None for an NP1 file without LF:
            NP1 AP is high-pass filtered in hardware, so LF cannot be derived from it.

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
                f"Window {window} is not stored in {self.h5_file}. Stored windows: "
                f"{self.windows}"
            )
        index = int(np.flatnonzero(match)[0])

        raw = {}
        with h5py.File(self.h5_file, "r") as f:
            for band, info in self._bands.items():
                # Same conversion as spikeglx.Reader.read, so the volts are identical.
                data = f[band]["data"][index][self._rows].astype(np.float32)
                data *= info["sample2volts"][self._rows, np.newaxis]
                raw[band] = data
        fs_ap = self._bands["ap"]["fs"]
        if "lf" in self._bands:
            raw_lf, fs_lf = raw["lf"], self._bands["lf"]["fs"]
        elif not self._probe_has_lf_band:
            raw_lf, fs_lf = derive_lf(raw["ap"], fs_ap)
        else:
            raw_lf, fs_lf = None, None
            if not self._warned_no_lf:
                LOGGER.warning(
                    "%s has no LF data although the probe records LF; no LF features",
                    self.h5_file,
                )
                self._warned_no_lf = True
        return RawSnippet(raw_ap=raw["ap"], raw_lf=raw_lf, fs_ap=fs_ap, fs_lf=fs_lf)

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
            ``SpikeGLXFileFeatureCalculator.load_channel_metadata``.
        """
        if self._channel_metadata is not None:
            return self._channel_metadata
        geometry = self.load_geometry()
        return pd.DataFrame(
            {
                "channel": np.arange(len(geometry["x"])),
                "rawInd": geometry["ind"],
                "axial_um": np.asarray(geometry["y"], dtype=float),
                "lateral_um": np.asarray(geometry["x"], dtype=float),
                "shank": np.asarray(geometry["shank"], dtype=float),
            }
        )

    def available_duration(self) -> tuple[float, float | None]:
        """Return the end of the last stored snippet for AP and LF.

        Returns:
            tuple[float, float | None]: ``(duration_ap, duration_lf)``. ``duration_lf`` is
            None for an NP1 file without LF data. LF is available when it is stored or
            derivable (probe without an LF band).
        """
        end = float(self._t_start.max() + self._duration)
        return end, (
            end if "lf" in self._bands or not self._probe_has_lf_band else None
        )

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
                raise ValueError(f"No trajectory stored in {self.h5_file}")
            LOGGER.info("No trajectory stored in %s", self.h5_file)
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
                f"{self.h5_file} holds {len(self.windows)} snippets, not {n_snippets}"
            )
        for duration in (duration_ap, duration_lf):
            if duration is not None and not np.isclose(duration, self._duration):
                raise ValueError(
                    f"{self.h5_file} holds {self._duration} s snippets, not {duration} s"
                )
        return compute_aggregated_features(self, self.windows, options, n_jobs=n_jobs)
