"""Extract snippets of a Petersen lab session, with its histology.

Edit the paths and run by # %% cells; the paths follow the lab's server layout (raw_data/
and processed_data/). Their channel_locations.json lists only the channels Kilosort kept,
so the extractor pairs the entries with the recorded channels through channels.rawInd.npy
and computes the other channels from the alignment, prev_alignments.json.
Whisker stims run through the whole recording, so the snippets are placed between stims
rather than at the automatic times.
"""

# %% Paths and settings
import logging
from pathlib import Path

import h5py
import numpy as np
import spikeglx

from ephysatlas.snippet_extractor import extract_snippets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA = Path("/data/IBL")  # the folder that holds raw_data/ and processed_data/
SUBJECT, SESSION, RUN = "MH035", "MH035_20250514_142713", "MH035_g0"
PROBES = [0, 1]
# A recording can be added to a file only once: use a new file to run again.
output_file = Path("snippet_exports/petersen.h5")
duration = 5.0  # s, the extraction default
guard = 1.0  # s kept clear of each stim; the stim artefact lasts under 0.5 s
margin = 300.0  # s left out at each end, as for the automatic times

raw = DATA / "raw_data" / SUBJECT / "Recording" / SESSION / "Ephys" / RUN
processed = DATA / "processed_data" / SUBJECT / SESSION / "Ephys" / f"catgt_{RUN}"

# %% Extract each probe, with the snippets between whisker stims
for i in PROBES:
    ap_file = raw / f"{RUN}_imec{i}" / f"{RUN}_t0.imec{i}.ap.bin"
    ibl = processed / f"{RUN}_imec{i}" / "ibl_format"
    # Stim times on this probe's clock. A window starts guard after a stim and ends guard
    # before the next one; five of them are taken, spread over the recording.
    stims = np.loadtxt(ibl.parent / f"whisker_stim_times_to_imec{i}.txt")
    length = float(
        spikeglx.read_meta_data(ap_file.with_suffix(".meta"))["fileTimeSecs"]
    )
    starts = stims[:-1] + guard
    ok = stims[1:] - guard - starts >= duration
    ok &= (starts > margin) & (starts + duration < length - margin)
    starts = starts[ok]
    t_starts = np.round(starts[np.linspace(0, len(starts) - 1, 5).astype(int)], 3)
    extract_snippets(
        ap_file,
        output_file,
        name=f"{SESSION}_imec{i}",
        t_starts=t_starts,
        duration=duration,
        channel_locations=ibl / "channel_locations.json",
        xyz_picks=ibl / "xyz_picks.json",
        alignment=ibl / "prev_alignments.json",
        channel_index=ibl / "channels.rawInd.npy",
    )

# %% Check the output: every channel has a brain location
with h5py.File(output_file) as f:
    for name, rec in f["recordings"].items():
        logger.info(
            "%s: %d channels, %d with a brain location, snippets at %s s",
            name,
            len(rec["geometry/ind"]),
            len(rec["histology/channels/x"]),
            rec["t_start"][()],
        )
