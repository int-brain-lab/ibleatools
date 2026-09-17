"""Plot destriped voltage snippets: the AP and LF bands, and raw against destriped AP.

Cell 1 uses the 1 s of destriped voltage committed under ``tests/fixtures`` (384 channels,
AP at 30 kHz and LF at 2.5 kHz, in volts), so it runs without any recording. Cell 2 reads a
snippet of your own SpikeGLX recording: point ``ap_file`` and ``lf_file`` at it.

Run it from the repository root, whole or cell by cell (``# %%`` cells). The figures are saved
to ``figures/``.
"""

# %% Imports
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ephysatlas.plots
from ephysatlas.feature_calculators import (
    SnippetWindow,
    SpikeGLXFileFeatureCalculator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# %% 1. The destriped AP and LF bands of the committed snippet
ap = np.load(REPO / "tests" / "fixtures" / "ap_destriped.npy")  # (384, 30000), volts
lf = np.load(REPO / "tests" / "fixtures" / "lf_destriped.npy")  # (384, 2500), volts

# 50 ms of AP is enough to see single spikes; the LF needs the whole second
fig, axs = ephysatlas.plots.figure_voltage_snippets(
    {
        "AP band (300 Hz – 10 kHz), destriped": (ap[:, :1500], 30_000),
        "LF band (< 200 Hz), destriped": (lf, 2_500),
    }
)
fig.savefig(outdir / "destriped_snippet_ap_lf.png", dpi=110)
logger.info("wrote %s", outdir / "destriped_snippet_ap_lf.png")

# %% 2. Raw against destriped AP, on a recording of your own
ap_file = Path("/data/probe00.ap.cbin")
lf_file = Path("/data/probe00.lf.cbin")
if not ap_file.exists():
    logger.warning("%s not found: point ap_file and lf_file at a recording", ap_file)
else:
    calc = SpikeGLXFileFeatureCalculator(ap_file=ap_file, lf_file=lf_file)
    snippet = calc.get_destriped_snippet(
        SnippetWindow(t_start=300.0, duration_ap=1.0, duration_lf=1.0)
    )
    fs = snippet.raw.fs_ap
    n = int(0.05 * fs)  # 50 ms
    fig, axs = ephysatlas.plots.figure_voltage_snippets(
        {
            "AP, raw": (snippet.raw.raw_ap[:, :n], fs),
            "AP, destriped": (snippet.des_ap[:, :n], fs),
        }
    )
    fig.savefig(outdir / "raw_vs_destriped_ap.png", dpi=110)
    logger.info("wrote %s", outdir / "raw_vs_destriped_ap.png")
plt.close("all")
