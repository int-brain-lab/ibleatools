"""Regenerate the figures of the ibleatools documentation site (docs/figures/).

Run it from the repository root, whole or cell by cell (``#%%``):

    python docs/figures/make_figures.py

- Cells 1-3 read only the fixtures committed under ``tests/fixtures/``.
- Cell 4 downloads a public model from the Hugging Face Hub, and the histology of its two
  example probes from the IBL public database through ONE. No account is needed.
- Cell 5 downloads the public spatial encoder, and the Allen atlas volumes through iblatlas.
- Cell 6 copies conceptual figures out of the ephys-atlas slide decks and paper-figure PDFs.
  They are not in this repository: point the ``EPHYS_ATLAS_FIGURES_DATA`` environment
  variable at the folder that holds them, or the cell is skipped.
- Cell 7 draws the overview of the home page from the data of cells 1 and 2 (run them first)
  and a sagittal slice of the Allen atlas through iblatlas. No account is needed.

Cells 1-5 draw with the figure functions of ``ephysatlas.plots``, as the scripts in
``examples/`` do. Regenerate and commit a figure whenever its code changes.
"""

# %%
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch  # noqa: F401 -- before ephysatlas, see explanation/install-variants
import matplotlib

matplotlib.use("Agg")  # headless: figures are only ever saved to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ephysatlas.plots
from ephysatlas.anatomy import NEW_VOID, ClassifierRegions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("make_figures")

# Run from the repository root, or as a script from anywhere
REPO = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
FIGURES = REPO / "docs" / "figures"
FIXTURES = REPO / "tests" / "fixtures"
FEATURES_W28 = FIXTURES / "features" / "2025_W28"

# The fixture probe used by the depth-resolved figures: it crosses Isocortex, HPF and TH
PID = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"

# The features of the heatmaps, grouped as in the ephys-atlas paper figure (LF, CSD, AP, spikes)
FEATURE_GROUPS = {
    "LF": [
        "rms_lf",
        "psd_delta",
        "psd_theta",
        "psd_alpha",
        "psd_beta",
        "psd_gamma",
        "psd_lfp",
    ],
    "CSD": [
        "rms_lf_csd",
        "psd_delta_csd",
        "psd_theta_csd",
        "psd_alpha_csd",
        "psd_beta_csd",
        "psd_gamma_csd",
        "psd_lfp_csd",
    ],
    "AP": ["rms_ap", "cor_ratio"],
    "Spikes": [
        "spike_count",
        "alpha_mean",
        "alpha_std",
        "peak_val",
        "trough_val",
        "tip_val",
        "depolarisation_slope",
        "repolarisation_slope",
        "recovery_slope",
        "polarity",
    ],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# The iblatlas regions plus the decoders' extra 'void_fluid' class (id 2000): plain
# BrainRegions does not know that id. No download, the region table ships with iblatlas
br = ClassifierRegions()
br.add_new_region(NEW_VOID)


# %% 1. A destriped AP and LF snippet, as the feature functions see it
# tests/fixtures holds 1 s of destriped AP (30 kHz) and LF (2.5 kHz), 384 channels, in volts
ap = np.load(FIXTURES / "ap_destriped.npy").astype(np.float32)
lf = np.load(FIXTURES / "lf_destriped.npy").astype(np.float32)

# 50 ms of AP is enough to see single spikes; the LF needs the whole second
fig, axs = ephysatlas.plots.figure_voltage_snippets(
    {
        "AP band (300 Hz – 10 kHz), destriped": (ap[:, :1500], 30_000),
        "LF band (< 200 Hz), destriped": (lf, 2_500),
    }
)
# A photograph-like image: JPEG is several times smaller than PNG here
fig.savefig(
    FIGURES / "destriped_snippet_ap_lf.jpg", dpi=110, pil_kwargs={"quality": 85}
)
plt.close(fig)
logger.info("wrote destriped_snippet_ap_lf.jpg")


# %% 2. Z-scored features down one probe, beside the histology
df = pd.read_parquet(FEATURES_W28 / "raw_ephys_features_denoised.pqt")
channels = pd.read_parquet(FEATURES_W28 / "channels.pqt")

# Each feature is z-scored over every channel of the 13 fixture probes, then drawn for one
# probe, sorted by depth, beside its Cosmos regions
fig, axs = ephysatlas.plots.figure_features_heatmap(
    df,
    PID,
    channels=channels,
    feature_groups=FEATURE_GROUPS,
    regions=br,
    title=f"Denoised features along probe {PID[:8]} (2025_W28 fixture)",
)
fig.savefig(FIGURES / "features_zscored_along_probe.png", dpi=110)
plt.close(fig)
logger.info("wrote features_zscored_along_probe.png")


# %% 3. The same probe before and after denoising
from ephysatlas.features import EphysTransformer  # noqa: E402

raw = pd.read_parquet(FEATURES_W28 / "raw_ephys_features.pqt")
# Put the raw table in the denoised units first (rms_* to dB, spike_count to log2), so the
# only remaining difference between the two panels is the total-variation filter
transformer = EphysTransformer()
transformer.fit(raw[FEATURES])
raw_t = transformer.transform(raw[FEATURES])

# Both panels on one scale: the z-score of the first table, the aggregated one
fig, axs = ephysatlas.plots.figure_features_heatmap(
    {"aggregated, before denoising": raw_t, "after denoising": df},
    PID,
    channels=channels,
    feature_groups=FEATURE_GROUPS,
    regions=br,
    show_group_labels=False,
    fontsize=7,
    title=f"Total-variation denoising, probe {PID[:8]} (2025_W28 fixture)",
)
fig.set_size_inches(13, 6)
plt.setp(axs["histology"].texts, fontsize=7)  # so "Isocortex" fits in the narrow bar
fig.savefig(FIGURES / "features_raw_vs_denoised.png", dpi=110)
plt.close(fig)
logger.info("wrote features_raw_vs_denoised.png")


# %% 4. Region predictions along the probe, beside the histology
# Needs network access: the Hugging Face Hub and the IBL public database
from brainbox.io.one import SpikeSortingLoader  # noqa: E402
from one.api import ONE  # noqa: E402

from ephysatlas import load_pretrained  # noqa: E402

model = load_pretrained(
    "int-brain-lab/ea-decoder-channel-transformer", revision="2026_W42"
)
df_example = pd.read_parquet(model.path_model / "example" / "features_sample.parquet")
out = model.predict(df_example)

# The two example probes are public IBL insertions: their histology is open data
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    silent=True,
)

# Three panels per probe: histology, predicted region, probabilities
pids = out.index.get_level_values("pid").unique()
fig, axs = plt.subplots(
    1,
    3 * len(pids),
    figsize=(4 * len(pids), 7),
    width_ratios=[1, 1, 2] * len(pids),
    constrained_layout=True,
)
for k, pid in enumerate(pids):
    pred = out.loc[pid].join(df_example.loc[pid, "axial_um"])
    ephysatlas.plots.figure_predictions_along_probe(
        pred,
        pred["axial_um"],
        model.config["classes"],  # one id per class, in the order of the p_ columns
        histology=SpikeSortingLoader(pid=pid, one=one).load_channels(),
        regions=br,
        mapping=model.config["region_map"],
        ylim=(0, 3840),
        fig=fig,
        axs=axs[3 * k : 3 * k + 3],
    )
    axs[3 * k].set_ylabel(f"probe {pid[:8]}\ndepth from probe tip (µm)")
fig.suptitle("ProbeTransformer (2026_W42) on its two example probes, Cosmos regions")
fig.savefig(FIGURES / "predictions_along_probe.png", dpi=110)
plt.close(fig)
logger.info("wrote predictions_along_probe.png")

# One probe with the legend, the exact call of the Plot section of predictions-along-probe
pid = "91fb4ed3-ee27-4d7e-ba5d-f61202ed9884"
pred = out.loc[pid].join(df_example.loc[pid, "axial_um"])
fig, axs = ephysatlas.plots.figure_predictions_along_probe(
    pred,
    pred["axial_um"],
    model.config["classes"],
    histology=SpikeSortingLoader(pid=pid, one=one).load_channels(),
    mapping=model.config["region_map"],
    show_legend=True,
)
fig.savefig(FIGURES / "predictions_one_probe.png", dpi=110)
plt.close(fig)
logger.info("wrote predictions_one_probe.png")


# %% 5. Features predicted on a coronal plane by the spatial encoder
# Needs the Hugging Face Hub, and the Allen atlas volumes (iblatlas downloads them once)
from iblatlas.atlas import AllenAtlas  # noqa: E402

ba = AllenAtlas(res_um=25)
encoder = load_pretrained("int-brain-lab/ea-encoder-channel", revision="2026_W42")

# Every 4th voxel (100 µm) of the coronal plane at AP = -2 mm, inside the brain, with the
# Cosmos outlines
fig, axs = ephysatlas.plots.figure_encoder_coronal(
    encoder,
    ["psd_delta", "rms_ap", "spike_amplitude"],
    ap_m=-2000e-6,
    brain_atlas=ba,
    scales={"spike_amplitude": 1e6},  # volts -> µV
    units={
        "psd_delta": "dB rel. V²/Hz",
        "rms_ap": "dB rel. V",
        "spike_amplitude": "µV",
    },
)
fig.suptitle(
    "Spatial encoder (2026_W42): expected features on the coronal plane AP = −2 mm"
)
fig.savefig(FIGURES / "encoder_coronal_maps.png", dpi=110)
plt.close(fig)
logger.info("wrote encoder_coronal_maps.png")


# %% 6. Conceptual figures copied from the ephys-atlas slides and paper-figure PDFs
from PIL import Image  # noqa: E402

source = Path(
    os.environ.get("EPHYS_ATLAS_FIGURES_DATA", REPO.parent / "ephys_atlas_figures_data")
)
DECK_ATLAS = (
    "2026-08-24_OSSS_04_Alignment_EphysAtlas.pptx"  # OSSS 2026, P. Rai / O. Winter
)
DECK_SORTING = (
    "2026-08-24_OSSS_03_Spike-Sorting-Preprocessing.pptx"  # OSSS 2026, O. Winter
)
FIGURE_1 = "figure1_dataset_summary_cosmos_features_ibl_style_v2.pdf"

# output file: (source file, slide image(s) or None, crop box as fractions of the width and
# height (left, top, right, bottom) or None)
COPIES = {
    "voltage_by_region.jpg": (  # slide 29
        DECK_ATLAS,
        ["image30.png", "image32.png", "image35.png"],
        None,
    ),
    "dataset_insertions.jpg": (
        DECK_ATLAS,
        "image42.png",
        (0, 0, 1, 0.89),
    ),  # s30, no cursor
    "feature_bands.png": (DECK_ATLAS, "image36.png", None),  # slide 32
    "decoder_confusion_matrix.png": (DECK_ATLAS, "image51.png", None),  # slide 35
    "decoder_shap.png": (DECK_ATLAS, "image47.png", None),  # slide 46
    "encoder_inputs.png": (DECK_ATLAS, "image55.png", None),  # slide 42
    "encoder_maps.png": (
        DECK_ATLAS,
        "image44.png",
        (0, 0.05, 1, 1),
    ),  # s39, panel c only
    "alignment_gui.jpg": (DECK_ATLAS, "image41.png", None),  # slide 45
    "destriping_steps.png": (DECK_SORTING, "image20.png", None),  # slide 12
    "bad_channels.jpg": (DECK_SORTING, "image4.png", None),  # slide 9
    "waveform_points.png": ("example waveform points.png", None, None),
    "raw_to_features.png": (FIGURE_1, None, (0, 0.30, 1, 1)),  # panel d
    "auto_alignment.png": ("figure5_auto_alignment.pdf", None, None),
}

if not source.exists():
    logger.warning(
        f"{source} not found: set EPHYS_ATLAS_FIGURES_DATA, skipping the copies"
    )
else:
    for name, (filename, member, crop) in COPIES.items():
        if filename.endswith(".pptx"):
            with zipfile.ZipFile(source / filename) as z:
                images = [
                    Image.open(z.open(f"ppt/media/{m}")).convert("RGB")
                    for m in np.atleast_1d(member)
                ]
            # Several slide images side by side, at a common height
            height = min(im.height for im in images)
            images = [
                im.resize((round(im.width * height / im.height), height))
                for im in images
            ]
            image = Image.new(
                "RGB",
                (sum(im.width for im in images) + 20 * (len(images) - 1), height),
                "white",
            )
            for i, im in enumerate(images):
                image.paste(im, (sum(m.width + 20 for m in images[:i]), 0))
        elif filename.endswith(".pdf"):
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-png",
                        "-r",
                        "250",
                        "-singlefile",
                        source / filename,
                        f"{tmp}/page",
                    ],
                    check=True,
                )
                image = Image.open(f"{tmp}/page.png").convert("RGB")
        else:
            image = Image.open(source / filename).convert("RGB")
        if crop is not None:
            w, h = image.size
            image = image.crop(
                (
                    round(crop[0] * w),
                    round(crop[1] * h),
                    round(crop[2] * w),
                    round(crop[3] * h),
                )
            )
        image.thumbnail((1600, 1600))  # in place, keeps the aspect ratio
        if name.endswith(".jpg"):
            image.save(FIGURES / name, quality=85, optimize=True)
        else:
            # A 256-colour palette keeps line art and heatmaps sharp at a fraction of the size
            image = image.quantize(colors=256, dither=Image.Dither.NONE)
            image.save(FIGURES / name, optimize=True)
        logger.info(f"wrote {name} ({(FIGURES / name).stat().st_size / 1024:.0f} kB)")


# %% 7. The overview of the home page: a recording, its features, its brain regions
# A schematic drawn from real data: a Neuropixels probe in a sagittal slice of the Allen atlas,
# the fixture voltage of cell 1, and the features and histology of probe PID from cell 2.
# Everything is placed on one canvas of 960 x 290 units, y pointing down.
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from iblatlas.atlas import AllenAtlas  # noqa: E402

ba = AllenAtlas(res_um=25)
ML_M, AP_M = (
    -1200e-6,
    -2000e-6,
)  # the sagittal plane, and the AP of the vertical probe (m)

# The z-scored features of probe PID, as in cell 2, in its channel order sorted by depth
ch = channels.loc[PID]
mean, std = df[FEATURES].mean(), df[FEATURES].std()
z = ((df.loc[PID, FEATURES] - mean) / std).reindex(ch.index)
order = np.argsort(ch["axial_um"].to_numpy(), kind="stable")

fig = plt.figure(figsize=(12, 12 * 290 / 960))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

# 1. The sagittal slice, plain, cropped to the brain: rows are DV (0 = dorsal), columns AP
# (0 = anterior), at x = 4, y = 78 and 185 units wide
sagittal = ba.label[:, ba.bc.x2i(ML_M), :].T
rows, cols = np.nonzero(sagittal)
r0, c0 = rows.min(), cols.min()
sagittal = sagittal[r0 : rows.max() + 1, c0 : cols.max() + 1]
# A border of void voxels: the brain reaches the edge of the volume, and the outline must close
sagittal = np.pad(sagittal, 1)
c0 -= 1  # the AP column of the probe, below, counts the padding
cosmos = ba.regions.mappings["Cosmos"][sagittal]  # Cosmos region index of each voxel
scale = 185 / sagittal.shape[1]  # canvas units per voxel
x0, y0 = 4, 78
xs = x0 + (np.arange(sagittal.shape[1]) + 0.5) * scale
ys = y0 + (np.arange(sagittal.shape[0]) + 0.5) * scale
ax.imshow(
    np.where(sagittal[..., None] > 0, [0.945, 0.93, 0.9, 1], [1, 1, 1, 0]),
    extent=[x0, x0 + sagittal.shape[1] * scale, y0 + sagittal.shape[0] * scale, y0],
    interpolation="nearest",
    aspect="auto",
)
# Light region boundaries, and a darker outline of the brain
ax.contour(
    xs, ys, cosmos, levels=np.unique(cosmos) + 0.5, colors="#999999", linewidths=0.25
)
ax.contour(
    xs, ys, (sagittal > 0).astype(float), levels=[0.5], colors="#555555", linewidths=0.8
)

# The probe, vertical at AP_M: its 384 channels span the deepest 3.84 mm, from the brain
# surface down (Isocortex, HPF then TH at this position, like probe PID)
iap = ba.bc.y2i(AP_M) - c0
px = x0 + (iap + 0.5) * scale
entry = y0 + np.argmax(sagittal[:, iap] > 0) * scale
tip = entry + 3840 / 25 * scale
ax.add_patch(Rectangle((px - 7, 40), 14, 14, color="#555555"))  # the base
ax.add_patch(Rectangle((px - 2, 54), 4, tip - 58, fc="#c8c8c8", ec="#555555", lw=0.5))
ax.add_patch(
    Polygon(
        [(px - 2, tip - 4), (px + 2, tip - 4), (px, tip)],
        fc="#c8c8c8",
        ec="#555555",
        lw=0.5,
    )
)
for k, y in enumerate(
    np.arange(entry, tip - 4, 1.6)
):  # the checkerboard of recording sites
    ax.add_patch(
        Rectangle((px - 1.6 + 2 * (k % 2), y), 1.2, 1.2, color="#c9a227", lw=0)
    )
# The flex cable leaving the base, and dashed lines from the top and bottom recording sites to
# the top and bottom of the voltage image: each row of the image is a channel
ax.add_patch(
    PathPatch(
        MplPath(
            [(px + 7, 45), (px + 18, 43), (px + 36, 33)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        ),
        fill=False,
        ec="#777777",
        lw=2,
    )
)
for y_site, y_image in [(entry, 62), (tip, 206)]:
    ax.plot([px + 2, 196], [y_site, y_image], color="#999999", lw=0.9, ls=(0, (3, 3)))

# 50 ms of the fixture AP voltage (cell 1), tip at the bottom, without its top 32 channels:
# they are outside the brain, with about twice the voltage of the others
volt = ap[:352, :1500]
vmax = 4 * np.median(np.std(volt, axis=1))
ax.imshow(
    volt,
    extent=[196, 294, 206, 62],
    origin="lower",
    cmap="Greys_r",
    vmin=-vmax,
    vmax=vmax,
    aspect="auto",
)
ax.add_patch(Rectangle((196, 62), 98, 144, fill=False, ec="#999999", lw=0.8))

# 2. The z-scored features of probe PID (cell 2), tip at the bottom, in their groups
ax.imshow(
    z.to_numpy()[order],
    extent=[400, 592, 220, 50],
    origin="lower",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    aspect="auto",
    interpolation="none",
)
x = 400
for name, group in FEATURE_GROUPS.items():
    width = 192 * len(group) / len(FEATURES)
    ax.text(x + width / 2, 44, name, ha="center", fontsize=11, color="#555555")
    x += width
    ax.plot([x, x], [50, 220], color="w", lw=1.5)
ax.add_patch(Rectangle((400, 50), 192, 170, fill=False, ec="#999999", lw=0.8))

# The learned models, drawn as a small network between two arrows
arrow = dict(arrowstyle="-|>", color="#555555", lw=2, mutation_scale=18)
ax.annotate("", xy=(380, 135), xytext=(305, 135), arrowprops=arrow)
ax.annotate("", xy=(720, 135), xytext=(612, 135), arrowprops=arrow)
layers = [
    [(646, 70), (646, 90), (646, 110)],
    [(668, 66), (668, 86), (668, 106)],
    [(690, 86)],
]
for layer, next_layer in zip(layers[:-1], layers[1:]):
    for xa, ya in layer:
        for xb, yb in next_layer:
            ax.plot([xa, xb], [ya, yb], color="#888888", lw=0.8)
for xn, yn in sum(layers, []):
    ax.add_patch(Circle((xn, yn), 4.5, fc="white", ec="#888888", lw=0.8))

# 3. The Cosmos regions of probe PID from its histology: what a correct prediction returns
ax_r = fig.add_axes([760 / 960, 1 - 220 / 290, 66 / 960, 170 / 290])
cosmos_id = br.remap(ch["atlas_id"].to_numpy(), source_map="Allen", target_map="Cosmos")
ephysatlas.plots.plot_region_bar(
    ax_r, ch["axial_um"].to_numpy(), cosmos_id, regions=br, fontsize=10
)
ax_r.set_yticks([])
# A light frame on all four sides: the ibl_style of ephysatlas.plots hides the top and right
plt.setp(list(ax_r.spines.values()), color="#999999", visible=True)

# The labels: (x, y, text, font size, colour, bold)
for x, y, text, size, color, bold in [
    (150, 248, "Neuropixels recording", 15, "#222222", True),
    (150, 268, "mouse brain, 384 channels", 12.5, "#666666", False),
    (245, 56, "channels × time", 11, "#666666", False),
    (342, 122, "compute", 12.5, "#444444", False),
    (342, 155, "features", 12.5, "#444444", False),
    (496, 248, "Per-channel features", 15, "#222222", True),
    (496, 268, "one row per channel", 12.5, "#666666", False),
    (666, 158, "learned models", 12.5, "#444444", False),
    (666, 175, "trained on", 10.5, "#777777", False),
    (666, 190, "IBL recordings", 10.5, "#777777", False),
    (793, 248, "Brain regions", 15, "#222222", True),
    (793, 268, "a region per channel", 12.5, "#666666", False),
]:
    ax.text(
        x,
        y,
        text,
        ha="center",
        fontsize=size,
        color=color,
        fontweight="bold" if bold else "normal",
    )
ax.set(xlim=(0, 960), ylim=(290, 0))  # imshow and contour rescale the canvas: reset it
fig.savefig(FIGURES / "overview.png", dpi=150)
plt.close(fig)
logger.info(
    f"wrote overview.png ({(FIGURES / 'overview.png').stat().st_size / 1024:.0f} kB)"
)
