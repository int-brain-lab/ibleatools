Load LFP Features
==================

This guide covers downloading and loading the **full-recording compressed LFP**
archives (release **v04**) produced by `lfpack <https://github.com/int-brain-lab/lfpack>`_:
lossy HDF5 encodings of the entire LFP trace for every insertion (1099 PIDs), at three
tiers. See the `lfpack documentation <https://int-brain-lab.github.io/lfpack/>`_ for the
reader API.

.. note::

   Reading v04 (lfpack format 2) requires **lfpack ≥ 1.0.0**
   (``uv pip install -U "lfpack>=1.0.0"``). The v03 archives and the ``mild`` /
   ``aggressive`` levels are deprecated and removed; ``mild`` → ``small``.

Tiers
-----

All tiers use the same codec (ε = 100); they differ only in α, the wavelet-packet
threshold, which is the only parameter that moves decoding.

.. list-table::
   :header-rows: 1

   * - ``level``
     - α
     - File
     - Size
     - File / float32 at 250 Hz
   * - ``small``
     - 14
     - ``lf_compressed_v04_a14_small_all.h5``
     - 11.2 GB
     - 0.57 %
   * - ``default``
     - 7
     - ``lf_compressed_v04_a07_default_all.h5``
     - 21.5 GB
     - 1.10 %
   * - ``fine``
     - 2.5
     - ``lf_compressed_v04_a2p5_fine_all.h5``
     - 46.1 GB
     - 2.35 %

Use ``default`` unless size matters (``small``, which decodes like the former v03
``mild`` at a third of the size) or fidelity matters (``fine``).

Every recording carries brain locations (``ml``/``ap``/``dv``/``atlas_id``/``acronym``),
sync knots, bad-channel ``labels`` and the saturation table.

Known exceptions: PID ``bf96f6d6`` has no bad-channel labels, and two PIDs are fully
flagged bad and decode to zeros (`lfpack#21 <https://github.com/int-brain-lab/lfpack/issues/21>`_).

S3 Layout
---------

.. code-block:: text

    aggregates/atlas/projects/{project}/
    │
    └── lfp_aggregates/
        ├── lf_compressed_v04_a14_small_all.h5    small   (ε=100, α=14)   ~11 GB
        ├── lf_compressed_v04_a07_default_all.h5  default (ε=100, α=7)    ~21.5 GB
        └── lf_compressed_v04_a2p5_fine_all.h5    fine    (ε=100, α=2.5)  ~46 GB

Each archive is a single multi-recording HDF5 file with one top-level group per
insertion (``pid``), produced by ``lfpack.merge_h5``.

Downloading
-----------

.. code-block:: python

    from pathlib import Path
    from one.api import ONE
    import ephysatlas.data

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    # default tier (~21.5 GB)
    ephysatlas.data.download_lfp_features(local_path, project=project, one=one)

    # small tier (~11 GB)
    ephysatlas.data.download_lfp_features(
        local_path, project=project, one=one, level='small'
    )

Loading
-------

.. code-block:: python

    from pathlib import Path
    import ephysatlas.data

    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'
    pid = '00a824c0-e060-495f-9ebc-79c82fef4c67'

    sr = ephysatlas.data.read_lfp_features(local_path.joinpath(project), pid)  # level='default'
    traces = sr[0:2500, :]          # (2500, nc) float32, volts
    sr.nc, sr.fs                    # channel count, sample rate (Hz)

``read_lfp_features`` returns an ``lfpack.LFPackReader``, a drop-in replacement for
``spikeglx.Reader`` — chunks are decompressed on demand. Pass ``bin_channels=`` to
sum adjacent channels on read, or ``scale=`` to open a coarser pyramidal level.

See also
--------

* :doc:`s3-architecture` — complete S3 folder layout
* :doc:`load-cells-features` — spike-triggered LFP (per-cell, not full recording)
