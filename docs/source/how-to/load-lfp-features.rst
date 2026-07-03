Load LFP Features
==================

This guide covers downloading and loading the **full-recording compressed LFP**
archives produced by `lfpack <https://github.com/int-brain-lab/lfpack>`_: lossy
HDF5 encodings (>100× compression, RMSE < 25 µV) of the entire LFP trace for
every insertion, at two compression levels.

S3 Layout
---------

.. code-block:: text

    aggregates/atlas/projects/{project}/
    │
    └── lfp_aggregates/
        ├── lf_compressed_all.h5              default level    (ε=150, α=28)  ~23 GB
        └── lf_compressed_aggressive_all.h5   aggressive level (ε=450, α=96)  ~12 GB

Each archive is a single multi-recording HDF5 file with one top-level group per
insertion (``pid``), produced by ``lfpack.merge_h5``. The "aggressive" level trades
fidelity for size; use "default" unless disk/bandwidth is the binding constraint.

Downloading
-----------

.. code-block:: python

    from pathlib import Path
    from one.api import ONE
    import ephysatlas.data

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    # default level (~23 GB)
    ephysatlas.data.download_lfp_features(local_path, project=project, one=one)

    # aggressive level (~12 GB)
    ephysatlas.data.download_lfp_features(
        local_path, project=project, one=one, level='aggressive'
    )

Loading
-------

.. code-block:: python

    from pathlib import Path
    import ephysatlas.data

    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'
    pid = '00a824c0-e060-495f-9ebc-79c82fef4c67'

    sr = ephysatlas.data.read_lfp_features(local_path / project, pid)
    traces = sr[0:2500, :]          # (2500, nc) float32, volts
    sr.nc, sr.fs                    # channel count, sample rate (Hz)

``read_lfp_features`` returns an ``lfpack.LFPackReader``, a drop-in replacement for
``spikeglx.Reader`` — chunks are decompressed on demand. Pass ``bin_channels=`` to
sum adjacent channels on read, or ``scale=`` to open a coarser pyramidal level.

See also
--------

* :doc:`s3-architecture` — complete S3 folder layout
* :doc:`load-cells-features` — spike-triggered LFP (per-cell, not full recording)
