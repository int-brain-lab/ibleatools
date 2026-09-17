"""Plot the features the spatial encoder predicts on a coronal plane of the brain.

The encoder predicts a channel's features from its position. Predicted on every 100 µm of a
coronal plane, they make a map of what a probe would record there.

Needs network access, and no account: the encoder comes from the Hugging Face Hub, and
iblatlas downloads the Allen atlas volumes on first use.

Run it whole or cell by cell (``# %%`` cells). The figure is saved to ``figures/``.
"""

# %% Imports, atlas and encoder
import torch  # noqa: F401 -- first: on macOS, torch after scikit-learn can segfault
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas

import ephysatlas.plots
from ephysatlas import load_pretrained

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

outdir = Path("figures")
outdir.mkdir(exist_ok=True)

ba = AllenAtlas(res_um=25)
encoder = load_pretrained("int-brain-lab/ea-encoder-channel", revision="2026_W42")
logger.info("the encoder predicts %d features", len(encoder.features))

# %% An LF, an AP and a spike-shape feature on the coronal plane AP = -2 mm
fig, axs = ephysatlas.plots.figure_encoder_coronal(
    encoder,
    ["psd_delta", "rms_ap", "spike_amplitude"],
    ap_m=-2000e-6,  # metres from bregma
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
fig.savefig(outdir / "encoder_coronal_maps.png", dpi=110)
logger.info("wrote %s", outdir / "encoder_coronal_maps.png")
plt.close("all")
