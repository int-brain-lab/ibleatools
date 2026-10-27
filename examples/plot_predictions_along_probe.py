"""Plot region predictions along the probe, beside the histology.

The whole-probe region decoder (ProbeTransformer) ships an example of two complete probes.
This script predicts them, and draws for each probe its Cosmos regions from histology, the
predicted region and the class probabilities as stacked bands.

Needs network access, and no account: the model comes from the Hugging Face Hub, and the
histology of the two example probes is open data in the IBL public database (openalyx).
On your own probe, pass your predictions and your channel table instead.

Run it whole or cell by cell (``# %%`` cells). The figure is saved to ``figures/``.
"""

# %% Imports and predictions
import torch  # noqa: F401 -- first: on macOS, torch after scikit-learn can segfault
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

import ephysatlas.plots
from ephysatlas import load_pretrained

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

outdir = Path("figures")
outdir.mkdir(exist_ok=True)

model = load_pretrained(
    "int-brain-lab/ea-decoder-channel-transformer", revision="2026_W42"
)
df = pd.read_parquet(model.path_model / "example" / "features_sample.parquet")
out = model.predict(df)

# %% One probe: histology, predicted region, probabilities
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    silent=True,
)
pid = out.index.get_level_values("pid")[0]
histology = SpikeSortingLoader(pid=pid, one=one).load_channels()  # axial_um, atlas_id
# The depth of each predicted channel: the decoder drops channels with non-finite features
pred = out.loc[pid].join(df.loc[pid, "axial_um"])

fig, axs = ephysatlas.plots.figure_predictions_along_probe(
    pred,
    pred["axial_um"],
    model.config["classes"],
    histology=histology,
    mapping=model.config["region_map"],
    show_legend=True,
)
fig.savefig(outdir / f"predictions_{pid[:8]}.png", dpi=110)
logger.info("wrote %s", outdir / f"predictions_{pid[:8]}.png")

# %% Both example probes on one figure: pass three axes per probe
pids = out.index.get_level_values("pid").unique()
fig, axs = plt.subplots(
    1,
    3 * len(pids),
    figsize=(4 * len(pids), 7),
    width_ratios=[1, 1, 2] * len(pids),
    constrained_layout=True,
)
for k, pid in enumerate(pids):
    pred = out.loc[pid].join(df.loc[pid, "axial_um"])
    ephysatlas.plots.figure_predictions_along_probe(
        pred,
        pred["axial_um"],
        model.config["classes"],
        histology=SpikeSortingLoader(pid=pid, one=one).load_channels(),
        mapping=model.config["region_map"],
        ylim=(0, 3840),  # the same depth range for both probes
        fig=fig,
        axs=axs[3 * k : 3 * k + 3],
    )
    axs[3 * k].set_ylabel(f"probe {pid[:8]}\ndepth from probe tip (µm)")
fig.suptitle("ProbeTransformer (2026_W42) on its two example probes, Cosmos regions")
fig.savefig(outdir / "predictions_along_probe.png", dpi=110)
logger.info("wrote %s", outdir / "predictions_along_probe.png")
plt.close("all")
