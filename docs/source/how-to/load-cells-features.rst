Load Cells Features
===================

This guide covers downloading and loading the **cells-level feature aggregates**
across all IBL insertions: cell tables, log-binned ACGs, peak-channel waveforms,
spike-triggered population coupling (stPC), and spike-triggered LFP (stLFP).

S3 Layout
---------

.. code-block:: text

    aggregates/atlas/projects/{project}/
    │
    ├── df_probe_details.pqt                      one row per insertion
    │
    └── cells_aggregates/
        ├── clusters.table.pqt                    all cells, QC + anatomy + waveform features  (n_cells × ~59)
        ├── clusters_good.table.pqt               QC-passing cells (bitwise_fail == 0)          (n_good × ~61)
        ├── clusters.acgs_log.npy                 log-binned ACGs, normalised by spike_count    (n_cells × 128) float16
        ├── acgs_log.times.npy                    ACG bin centres in seconds                    (128,) float64
        ├── clusters.waveforms_peak.npy           peak-channel waveform per cell                (n_cells × 128) float16
        ├── clusters_good.stpc.npy                spike-triggered population coupling           (n_good × 1000)    float16
        ├── clusters_good.stlfp.npy               spike-triggered LFP                           (n_good × 250)     float16
        ├── waveforms.voltage.npy                 all neighbourhood traces (~8 GB)              (n_traces × 128)   float16
        └── waveforms.table.pqt                   pid / cluster_id / abs_channel index          (n_traces × 3)

``clusters.acgs_log.npy`` values are in **sp/sp** (normalised by ``spike_count``); the
long-lag asymptote converges to the firing rate in sp/s.
Arrays indexed by cell are row-aligned with ``clusters.table.pqt``.
Arrays indexed by good cell are row-aligned with ``clusters_good.table.pqt``.

Downloading
-----------

.. code-block:: python

    from pathlib import Path
    from one.api import ONE
    import ephysatlas.data

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    # downloads df_probe_details.pqt + cells_aggregates/ (~1 GB, waveforms excluded)
    ephysatlas.data.download_project_data(local_path, project=project, one=one)

    # include waveforms.voltage.npy + waveforms.table.pqt (~8 GB extra)
    ephysatlas.data.download_project_data(local_path, project=project, one=one, large_files=True)

To download only one of the two parts:

.. code-block:: python

    ephysatlas.data.download_probe_details(local_path, project=project, one=one)
    ephysatlas.data.download_cells_features(local_path, project=project, one=one)
    # with neighbourhood waveforms:
    ephysatlas.data.download_cells_features(local_path, project=project, one=one, large_files=True)

Loading
-------

.. code-block:: python

    from pathlib import Path
    import ephysatlas.data

    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    r = ephysatlas.data.read_cells_features(local_path / project)
    df_cells             = r['df_clusters']           # all cells     (n_cells × ~59)
    df_cells_good        = r['df_clusters_good']      # good cells    (n_good × ~61)
    acgs_log             = r['acgs_log']              # (n_cells × 128) float32 — sp/sp
    acgs_log_times       = r['acgs_log_times']        # (128,) seconds
    waveforms_peak       = r['waveforms_peak']        # (n_cells × 128) float32
    stpc                 = r['stpc']                  # (n_good × 1000)    float16 memmap
    stlfp                = r['stlfp']                 # (n_good × 250)     float16 memmap
    # present only when downloaded with large_files=True:
    waveforms            = r.get('waveforms')         # (n_traces × 128)   float16 memmap — ~8 GB
    df_waveforms         = r.get('df_waveforms')      # (n_traces × 3) — pid/cluster_id/abs_channel index

Joining with probe metadata
---------------------------

.. code-block:: python

    df_probes = ephysatlas.data.read_probe_details(local_path / project)
    df = df_cells.merge(df_probes, on='pid', how='left')

See also
--------

* :doc:`s3-architecture` — complete S3 folder layout
* :doc:`load-channel-features` — channel-level features (ephys atlas main dataset)