import functools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
import scipy.signal

import ibldsp.plots
import ibldsp.voltage
from brainbox.io.one import SpikeSortingLoader
import brainbox.ephys_plots

import ephysatlas.features
import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.regionclassifier
import ephysatlas.plots
import ephysatlas.fixtures


"""
Electrophysiological data visualization and analysis reveal module.

This module provides comprehensive visualization tools for electrophysiological data
analysis, including feature visualization, classifier results, histology slices,
voltage traces, and bad channel detection. It serves as a high-level interface
for creating publication-ready figures from electrophysiological recordings.

The module includes:
- Feature visualization with histology overlays
- Classifier prediction results and confidence analysis
- Histology slice visualization with probe trajectories
- AP and LFP voltage trace visualization
- Bad channel detection and analysis
- Automated figure saving and management

Classes:
    AtlasReveal: Main class for creating comprehensive electrophysiological visualizations

Functions:
    save_figure: Decorator for automatically saving figures with configurable options

Constants:
    STREAM (bool): Default streaming mode for data loading

Examples:
    >>> from ephysatlas.reveal import AtlasReveal
    >>> import one.alf.io as alfio
    >>> 
    >>> # Initialize reveal object
    >>> one = alfio.One()
    >>> pid = "0228bcfd-632e-49bd-acd4-c334cf9213e9"
    >>> reveal = AtlasReveal(one=one, pid=pid)
    >>> 
    >>> # Create feature visualization
    >>> fig, axs = reveal.figure_01_features_with_histology_columns()
    >>> 
    >>> # Create classifier results visualization
    >>> fig, axs = reveal.figure_02_classifier_results()

Note:
    This module integrates with the IBL (International Brain Laboratory) ecosystem
    and provides automated figure generation for electrophysiological data analysis.
    It includes built-in figure saving capabilities and supports both raw and
    processed data visualization.

See Also:
    ephysatlas.features : Feature extraction and processing
    ephysatlas.plots : Basic plotting utilities
    ephysatlas.anatomy : Anatomical classification and atlas functionality
    ephysatlas.regionclassifier : Brain region classification models
"""

"""
Figure 01: ce - Features with checkerboard pattern, make sure to add the Atlas ID, Cosmos and the unique atlas ID next to it
Figure 02: bcg - Prediction of vanilla model + confidence
Figure 03: bcg - Histology slices
Figure 04: a(c) - AP band snippet (raw / destriped)
Figure 05: a(c) - LF band snippet (raw / destriped)
Figure 06: ad - Bad channel AP (NB: also plot the actual outcome from the dataframe, ie. the one in ALF)
Figure 07: h(ci) - Raster + behaviour start/stop times + snippets (computed and the one displayed) + spike sorting version

Data types:
a- raw data
b- target coordinates
c- ground truth: ephys aligned coordinates
d- bad channels
e- features (denoised)
f- encoding model - outlier predictions
g- decoding model - region predictions
h- spike sorting data
i- behaviour events
"""


def save_figure(func):
    """Decorator that optionally saves figures returned by methods.

    The decorated method should return a figure or a list of figures as its first return value.

    Args:
        func: The function to be decorated.

    Returns:
        function: Wrapped function with figure saving capability.

    Note:
        The decorated method should return a tuple where the first element is a figure or list of figures.
        The decorator will automatically save figures if save_dir is provided.
    """

    @functools.wraps(func)
    def wrapper(self, *args, save_dir=None, overwrite=False, filename=None, **kwargs):
        # Save figures if save_dir is provided
        method_name = func.__name__
        filename = f"{self.pid}_{method_name}.png" if filename is None else filename
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            file = next(save_dir.glob(filename), None)
            if file is not None and file.exists() and overwrite is False:
                return None

        result = func(self, *args, **kwargs)
        figures = result[0]
        if save_dir is not None:
            if isinstance(figures, list):
                # Multiple figures
                for i, fig in enumerate(figures):
                    filename = f"{Path(filename).stem}_{i}{Path(filename).suffix}"
                    fig.savefig(save_dir / filename, dpi=128, bbox_inches="tight")
            else:
                figures.savefig(save_dir / filename, dpi=128, bbox_inches="tight")

        return result

    return wrapper


class AtlasReveal:
    STREAM = True

    def __init__(self, one=None, pid=None, df_pid=None):
        self.atlas = ephysatlas.anatomy.ClassifierAtlas()
        self.one = one
        self.df_pid = df_pid
        self.pid = pid
        self.ssl = SpikeSortingLoader(pid=self.pid, one=self.one)
        self.sr_ap = self.ssl.raw_electrophysiology(band="ap", stream=self.STREAM)
        self.sr_lf = self.ssl.raw_electrophysiology(band="lf", stream=self.STREAM)

    @property
    def x_list(self):
        # TODO: get the feature set from the model if loaded
        return ephysatlas.features.voltage_features_set()

    @property
    def xy(self):
        return self.df_pid[["lateral_um", "axial_um"]].to_numpy()

    @staticmethod
    def _aggregate_dephs(df_pid):
        """Aggregate data by depths.

        Args:
            df_pid (pd.DataFrame): DataFrame containing probe data.

        Returns:
            pd.DataFrame: DataFrame aggregated by axial_um with mean values for numeric columns
                and mode values for label columns (Cosmos_id, Allen_id).
        """
        df_depths = df_pid.groupby("axial_um").mean(numeric_only=True)
        columns_labels = ["Cosmos_id", "Allen_id"]
        daggs = {
            k: pd.NamedAgg(column=k, aggfunc=lambda x: x.mode().iloc[0])
            for k in columns_labels
        }
        df_aids = df_pid.groupby("axial_um").agg(**daggs)
        for col in columns_labels:
            df_depths[col] = df_aids[col].values
        return df_depths

    @save_figure
    def figure_01_features_with_histology_columns(self, scaler=None, df_pid=None):
        """Create feature visualization with histology columns.

        This method creates a comprehensive visualization showing electrophysiological features
        plotted in channel space with histology overlays.

        Args:
            scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for normalizing features.
                If provided, features are scaled to [-1.2, 1.2] range. Defaults to None.
            df_pid (pd.DataFrame, optional): DataFrame containing probe data. If None, uses self.df_pid.
                This is useful for displaying raw features if needed. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure.
                - axs (matplotlib.axes.Axes): The axes containing the plot.
        """
        # option to override the default df_pid: this is useful for displaying the raw features if needed
        df_pid = df_pid if df_pid is not None else self.df_pid
        if scaler is not None:
            kwargs = {"scaler": scaler, "vmin": -1.2, "vmax": 1.2}
        else:
            kwargs = {}
        fig, axs = ephysatlas.plots.figure_features_channel_space(
            df_pid,
            self.x_list,
            self.xy,
            pid=self.pid,
            mapping="Allen",
            cmap="cividis",
            br=self.atlas.regions,
            **kwargs,
        )
        return fig, axs

    @staticmethod
    def _plot_raw_ephys(voltage, fs, xy, regions=None, df_pid=None, **kwargs):
        """Plot raw electrophysiological data with brain regions and voltage traces.

        Args:
            voltage (np.ndarray): Voltage data array.
            fs (float): Sampling frequency in Hz.
            xy (np.ndarray): Channel coordinates array.
            regions (iblatlas.regions.BrainRegions, optional): Brain regions object for plotting.
                Defaults to None.
            df_pid (pd.DataFrame, optional): DataFrame containing probe data. Defaults to None.
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure.
                - axs (matplotlib.axes.Axes): Array of axes containing the plots.
        """
        fig, axs = plt.subplots(
            1, 3, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 14, 0.4]}
        )
        brainbox.ephys_plots.plot_brain_regions(
            df_pid["atlas_id"].values,
            channel_depths=xy[:, 1],
            brain_regions=regions,
            ax=axs[0],
        )
        ibldsp.plots.voltageshow(voltage, fs=fs, ax=axs[1], cax=axs[2], **kwargs)
        axs[1].xaxis.set_ticks_position("bottom")
        axs[1].xaxis.set_label_position("bottom")
        fig.tight_layout()
        return fig, axs

    @save_figure
    def figure_02_classifier_results(self, df_predictions=None, path_model=None):
        """Create classifier results visualization.

        This method creates a comprehensive visualization showing the results of the channel regions classifier,
        including true labels, predictions, confidence scores, and cumulative probabilities.

        Args:
            df_predictions (pd.DataFrame, optional): DataFrame containing classifier predictions.
                If None, predictions are computed using the loaded model. Defaults to None.
            path_model (Path, optional): Path to the trained model directory. Required if df_predictions is None.
                Defaults to None.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure.
                - axs (matplotlib.axes.Axes): Array of axes containing the plots.

        Note:
            The figure shows:
            - Brain regions with Allen labels
            - True labels (Allen and Cosmos)
            - Classifier predictions
            - Confidence scores
            - Cumulative probabilities across depths
        """
        # Figure 02: results of the channel regions classifer
        classifier, model_info = ephysatlas.regionclassifier.load_model(path_model)
        rids = np.array(model_info["CLASSES"])
        xy = self.df_pid[["lateral_um", "axial_um"]].to_numpy()

        if df_predictions is None:
            print("No predictions provided, loading them from the model...")
            probas = classifier.predict_proba(
                self.df_pid.loc[:, model_info["FEATURES"]]
            )
            df_predictions = pd.DataFrame(
                probas,
                columns=[str(c) for c in model_info["CLASSES"]],
                index=self.df_pid.index,
            )
            df_predictions["prediction"] = rids[np.argmax(probas, axis=1)]
            df_predictions["confidence"] = np.max(probas, axis=1)

        df_pid_merged = self.df_pid.merge(
            df_predictions, left_index=True, right_index=True
        )
        accuracy = sklearn.metrics.accuracy_score(
            df_pid_merged["prediction"].values, df_pid_merged["Cosmos_id"].values
        )
        df_pid_merged["confidence"] = np.max(
            df_pid_merged.loc[:, [str(c) for c in rids]], axis=1
        )
        df_pid_merged["true_label_score"] = df_pid_merged.apply(
            lambda row: row[str(row["Cosmos_id"])], axis=1
        )
        df_depths = df_pid_merged.drop("acronym", axis=1).groupby("axial_um").mean()

        fig, axs = plt.subplots(
            1,
            8,
            figsize=(10, 6),
            gridspec_kw={"width_ratios": [0.3, 1, 0.3, 1, 1, 1, 0.2, 5]},
        )
        # brain regions column with ALlen leaf labels
        ax = axs[0]
        brainbox.ephys_plots.plot_brain_regions(
            self.df_pid["atlas_id"].values,
            channel_depths=xy[:, 1],
            brain_regions=self.atlas.regions,
            display=True,
            ax=ax,
        )
        ax = axs[1]
        ephysatlas.plots.plot_probe_rect2(
            xy,
            color=ephysatlas.plots.get_color_br(
                self.df_pid, self.atlas.regions, mapping="Allen"
            ),
            ax=ax,
        )
        ax.set_title("Allen True labels")

        # brain regions column with Cosmos leaf labels
        ax = axs[2]
        brainbox.ephys_plots.plot_brain_regions(
            self.df_pid["Cosmos_id"].values,
            channel_depths=xy[:, 1],
            brain_regions=self.atlas.regions,
            display=True,
            ax=ax,
        )
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

        ax = axs[3]
        ephysatlas.plots.plot_probe_rect2(
            xy,
            color=ephysatlas.plots.get_color_br(
                self.df_pid, self.atlas.regions, mapping="Cosmos"
            ),
            ax=ax,
        )
        ax.set_title("Cosmos True labels")

        # show predictions
        ax = axs[4]
        ephysatlas.plots.plot_probe_rect2(
            xy,
            color=self.atlas.regions.get(df_pid_merged["prediction"].values).rgb / 255,
            ax=ax,
        )
        ax.set_title("Cosmos prediction")

        # show confidence
        ax = axs[5]
        ephysatlas.plots.plot_probe_rect2(
            xy,
            color=ephysatlas.plots.get_color_feat(
                df_pid_merged["confidence"], cmap_name="magma", min_val=0, max_val=1
            ),
            ax=ax,
            colorbar=True,
        )
        ax.set_title("Confidence")

        axs[6].set_axis_off()

        # show cumulative probabilities
        ax = axs[7]
        ephysatlas.plots.plot_cumulative_probas(
            df_depths.loc[:, [str(c) for c in rids]].values,
            df_depths.index,
            np.array(rids),
            regions=self.atlas.regions,
            ax=ax,
        )
        ax.set_title("Classifier predicted probabilities")

        # Move y-axis to the right side
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
        )

        # Optional: Add a y-axis label
        ax.set_ylabel(
            "Depth (μm)", rotation=270, labelpad=15
        )  # Adjust labelpad as needed

        fig.suptitle(
            f"PID {self.pid} \n accuracy {accuracy:0.2} \n confidence {np.mean(df_pid_merged['confidence']): 0.2} \n true label score {np.mean(df_pid_merged['true_label_score']): 0.2}",
            y=0.08,
            fontweight="bold",
        )
        return fig, axs

    @save_figure
    def figure_03_histology_slices(self):
        """Create histology slice visualization with probe trajectories.

        This method creates a visualization showing three orthogonal slices through the brain atlas
        with overlaid probe trajectories, including both planned and aligned coordinates.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure.
                - axs (matplotlib.axes.Axes): Array of axes containing the three slice views.

        Note:
            The figure shows:
            - Coronal slice (AP view) at median y-coordinate
            - Sagittal slice (ML view) at median x-coordinate
            - Horizontal slice (DV view) at median z-coordinate
            - Both planned (target) and aligned (actual) probe trajectories
        """
        fig, axs = plt.subplots(
            1, 3, figsize=(14, 5), gridspec_kw={"width_ratios": self.atlas.bc.nxyz}
        )
        fig.suptitle(f"Probe {self.pid}")
        self.atlas.plot_cslice(
            ap_coordinate=np.median(self.df_pid["y"]),
            volume="annotation",
            ax=axs[0],
            alpha=0.7,
        )
        self.atlas.plot_sslice(
            ml_coordinate=np.median(self.df_pid["x"]),
            volume="annotation",
            ax=axs[1],
            alpha=0.7,
        )
        self.atlas.plot_hslice(
            dv_coordinate=np.median(self.df_pid["z"]),
            volume="annotation",
            ax=axs[2],
            alpha=0.7,
        )
        plot_args = dict(linewidth=2, label="aligned channels")
        x, y, z = (
            self.df_pid["x"].values * 1e6,
            self.df_pid["y"].values * 1e6,
            self.df_pid["z"].values * 1e6,
        )
        axs[0].plot(x, z, **plot_args)
        axs[1].plot(y, z, **plot_args)
        axs[2].plot(x, y, **plot_args)
        plot_args = dict(linewidth=2, label="planned coordinates")
        xt, yt, zt = (
            self.df_pid["x_target"].values * 1e6,
            self.df_pid["y_target"].values * 1e6,
            self.df_pid["z_target"].values * 1e6,
        )
        axs[0].plot(xt, zt, **plot_args)
        axs[1].plot(yt, zt, **plot_args)
        axs[2].plot(xt, yt, **plot_args)
        for ax in axs:
            ax.legend()
            ax.set(xlabel="um", ylabel="um")
        fig.tight_layout()
        return fig, axs

    @save_figure
    def figure_04_ap_voltage(self):
        """Create AP band voltage visualization.

        This method creates visualizations showing AP band voltage traces, comparing raw and preprocessed data.
        The data is filtered and destriped to show the effects of preprocessing.

        Returns:
            tuple: A tuple containing:
                - figs (list): List of two figures showing raw and preprocessed AP data.
                - axs (list): List of axes arrays for each figure.

        Note:
            The method shows:
            - Raw AP voltage traces with high-pass filtering
            - Preprocessed AP voltage traces after destriping
            - Both visualizations include brain region overlays and channel information
            - Data is extracted from a 1-second window starting at 600 seconds
        """
        t0, duration = 600, 1
        channel_labels = True  # TODO
        AP_XLIM = (0.47, 0.53)
        raw = self.sr_ap[
            slice(int(self.sr_ap.fs * t0), int((t0 + duration) * self.sr_ap.fs)),
            : -self.sr_ap.nsync,
        ].T
        butter_kwargs = {"N": 3, "Wn": 300 / self.sr_ap.fs * 2, "btype": "highpass"}
        sos = scipy.signal.butter(**butter_kwargs, output="sos")
        butt = scipy.signal.sosfiltfilt(sos, raw)
        # k_filter=None means no CAR nor spatial filter
        destripe = ibldsp.voltage.destripe(
            butt, fs=self.sr_ap.fs, channel_labels=channel_labels
        )
        kwargs = dict(
            xy=self.xy, regions=self.atlas.regions, xlim=AP_XLIM, df_pid=self.df_pid
        )
        fig0, axs0 = self._plot_raw_ephys(
            butt, fs=self.sr_ap.fs, title=f"AP raw {self.pid}", **kwargs
        )
        fig1, axs1 = self._plot_raw_ephys(
            destripe, fs=self.sr_ap.fs, title=f"AP preprocessed {self.pid}", **kwargs
        )
        return [fig0, fig1], [axs0, axs1]

    @save_figure
    def figure_05_lfp_voltage(self):
        """Create LFP voltage and CSD visualization.

        This method creates visualizations showing LFP voltage traces and current source density (CSD) analysis.
        The data is filtered, destriped, and processed to show both voltage and CSD representations.

        Returns:
            tuple: A tuple containing:
                - figs (list): List of two figures showing preprocessed LFP and CSD data.
                - axs (list): List of axes arrays for each figure.

        Note:
            The method shows:
            - Preprocessed LFP voltage traces after filtering and destriping
            - Current source density (CSD) analysis with Cadzow denoising
            - Both visualizations include brain region overlays and channel information
            - Data is extracted from a 4-second window starting at 600 seconds
            - CSD is computed with 5x decimation and 200 Hz maximum frequency
        """
        CSD_RANGE_AM3 = 10_000
        LF_XLIM = (1, 3)

        t0, duration = 600, 4
        channel_labels = True  # TODO

        raw = self.sr_lf[
            slice(int(self.sr_lf.fs * t0), int((t0 + duration) * self.sr_lf.fs)),
            : -self.sr_lf.nsync,
        ].T
        butter_kwargs = {"N": 3, "Wn": 2 / self.sr_lf.fs * 2, "btype": "highpass"}
        sos = scipy.signal.butter(**butter_kwargs, output="sos")
        butt = scipy.signal.sosfiltfilt(sos, raw)
        # k_filter=None means no CAR nor spatial filter
        preproc = ibldsp.voltage.destripe_lfp(
            butt, fs=self.sr_lf.fs, channel_labels=channel_labels, k_filter=None
        )
        csd = scipy.signal.decimate(preproc, q=5, zero_phase=True)
        csd = ibldsp.cadzow.cadzow_denoiser(
            csd, rank=5, fs=self.sr_lf.fs / 5, fmax=125,
            nswx=64, gap_threshold=2.0, ppca_k=2.0, h=self.sr_lf.geometry
        )
        csd = ibldsp.voltage.current_source_density(csd, h=self.sr_lf.geometry)

        kwargs = dict(
            xy=self.xy, regions=self.atlas.regions, xlim=LF_XLIM, df_pid=self.df_pid
        )
        fig0, axs0 = self._plot_raw_ephys(
            preproc, fs=self.sr_lf.fs, title=f"LFP preprocessed {self.pid}", **kwargs
        )
        fig1, axs1 = self._plot_raw_ephys(
            csd,
            fs=self.sr_lf.fs / 5,
            title=f"CSD {self.pid}",
            cbar_label="Current Density (A.m-3)",
            scaling=1,
            vrange=CSD_RANGE_AM3,
            **kwargs,
        )
        return [fig0, fig1], [axs0, axs1]

    @save_figure
    def figure_06_bad_channels(self):
        """Create bad channel detection visualization.

        This method creates a visualization showing the results of bad channel detection
        on AP band voltage data, including channel labels and feature analysis.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The created figure.
                - axs (matplotlib.axes.Axes): The axes containing the bad channel analysis.

        Note:
            The method shows:
            - Raw AP voltage traces
            - Bad channel detection results
            - Channel features used for detection
            - Data is extracted from a 1-second window starting at 600 seconds
        """
        t0, duration = 600, 1
        raw = self.sr_ap[
            slice(int(self.sr_ap.fs * t0), int((t0 + duration) * self.sr_ap.fs)),
            : -self.sr_ap.nsync,
        ].T
        ichannels, xfeats = ibldsp.voltage.detect_bad_channels(raw, fs=self.sr_ap.fs)
        fig, axs = ibldsp.plots.show_channels_labels(
            raw,
            self.sr_ap.fs,
            ichannels,
            xfeats,
        )
        return fig, axs
