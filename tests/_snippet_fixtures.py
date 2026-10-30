"""Test support: write tiny SpikeGLX recordings for the snippet extractor tests.

The ``.meta`` files in ``tests/fixtures/snippet_extractor`` are copies of real recordings
(an NP1.0 and an NP2.4 single-shank file). ``write_spikeglx_recording`` pairs them with random
int16 data, so the extractor can be tested without the real recordings of hundreds of GB.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures" / "snippet_extractor"


def write_spikeglx_recording(
    folder: Path,
    template: str,
    n_seconds: float,
    edits: Mapping[str, str | None] | None = None,
    with_lf: bool = True,
    seed: int = 0,
) -> Path:
    """Write ``rec_g0_t0.imec0.ap.bin`` and, optionally, the ``.lf.bin`` with their metas.

    Args:
        folder: Folder to write into. It is created if missing.
        template: Name of a meta pair in ``tests/fixtures/snippet_extractor``. ``"np1"`` reads
            ``np1.ap.meta`` and ``np1.lf.meta``.
        n_seconds: Recording length. Each file holds ``round(n_seconds * imSampRate)`` samples.
        edits: Changes to the meta text of both bands. A key (with or without the ``~``) maps
            to its new value, or to ``None`` to delete the line. A band whose meta doesn't
            have the key is left as it is, e.g. the NP1 LF meta has no ``snsShankMap``.
            ``fileSizeBytes`` and ``fileTimeSecs`` are set from the written file unless
            ``edits`` names them. The metas are written with CRLF line endings, as
            SpikeGLX does.
        with_lf: Also write the LF band.
        seed: Seed of the random data.

    Returns:
        Path of the AP ``.bin`` file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    edits = {key.lstrip("~"): value for key, value in (edits or {}).items()}
    for band in ("ap", "lf") if with_lf else ("ap",):
        meta = {}  # key as written in the file (it may start with "~") -> value text
        for line in (FIXTURES / f"{template}.{band}.meta").read_text().splitlines():
            key, value = line.split("=", maxsplit=1)
            meta[key] = value
        for key in [key for key in meta if key.lstrip("~") in edits]:
            value = edits[key.lstrip("~")]
            if value is None:
                del meta[key]
            else:
                meta[key] = value

        # The file is written after the edits, so its length and channel count follow
        # the edited sample rate and channel count.
        fs = float(meta["imSampRate"])
        n_samples = round(n_seconds * fs)
        n_channels = int(meta["nSavedChans"])
        data = rng.integers(-500, 501, size=(n_samples, n_channels), dtype=np.int16)
        bin_file = folder / f"rec_g0_t0.imec0.{band}.bin"
        data.tofile(bin_file)

        # spikeglx.Reader takes the length from fileTimeSecs.
        from_file = {"fileSizeBytes": data.nbytes, "fileTimeSecs": n_samples / fs}
        for key, value in from_file.items():
            if key not in edits:
                meta[key] = str(value)
        meta_text = "".join(f"{key}={value}\n" for key, value in meta.items())
        bin_file.with_suffix(".meta").write_text(meta_text, newline="\r\n")
    return folder / "rec_g0_t0.imec0.ap.bin"
