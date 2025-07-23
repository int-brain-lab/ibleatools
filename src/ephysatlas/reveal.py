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


def save_figure(func):
    """
    Decorator that optionally saves figures returned by methods.
    The decorated method should return a figure or a list of figures as its first return value.
    """

    @functools.wraps(func)
    def wrapper(self, *args, save_dir=None, overwrite=False, **kwargs):
        # Save figures if save_dir is provided
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            method_name = func.__name__
            file = next(save_dir.glob(f"{self.pid}_{method_name}*.png"), None)
            if file is not None and file.exists() and overwrite is False:
                return None

        result = func(self, *args, **kwargs)
        figures = result[0]
        if save_dir is not None:
            if isinstance(figures, list):
                # Multiple figures
                for i, fig in enumerate(figures):
                    filename = f"{self.pid}_{method_name}_{i}.png"
                    fig.savefig(save_dir / filename, dpi=128, bbox_inches="tight")
            else:
                # Single figure
                filename = f"{self.pid}_{method_name}_0.png"
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
        """
        Aggregate by depths
        :return:
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
    def figure_01_features_with_histology_columns(self, scaler=None):
        if scaler is not None:
            kwargs = {"scaler": scaler, "vmin": -1, "vmax": 1}
        else:
            kwargs = {}
        fig, axs = ephysatlas.plots.figure_features_channel_space(
            self.df_pid,
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
            f"PID {self.pid} \n accuracy {accuracy:0.2} \n confidence {np.mean(df_pid_merged['confidence']): 0.2}",
            y=0.08,
            fontweight="bold",
        )
        return fig, axs

    @save_figure
    def figure_03_histology_slices(self):
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
        csd = ibldsp.cadzow.cadzow_np1(
            csd, fs=self.sr_lf.fs / 5, fmax=200, rank=4, h=self.sr_lf.geometry
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
