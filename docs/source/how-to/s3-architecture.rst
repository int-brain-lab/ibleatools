S3 Data Architecture
====================

All IBL electrophysiology atlas data lives in a single private AWS S3 bucket::

    s3://ibl-brain-wide-map-private/aggregates/atlas/

Authentication uses ``one.api.ONE`` via Alyx credentials — pass a logged-in ``ONE``
instance as the ``one`` argument to every download function.

Folder Layout
-------------

.. code-block:: text

    aggregates/atlas/
    │
    ├── features/{project}/{label}/agg_full/     ← channel-level ephys features
    │   ├── raw_ephys_features_denoised.pqt      denoised channel features  (~200 MB)
    │   ├── raw_ephys_features.pqt               raw channel features
    │   ├── channels.pqt                         channel metadata (coordinates, probe info)
    │   └── channels_labels.pqt                  region labels per channel
    │
    ├── features/{project}/{label}_extended/     ← large optional data (cross-correlograms …)
    │
    ├── encoding_volumes/{project}/{label}/      ← 4-D CCF volume
    │   └── brainwide_ephys_atlas_25um.npz       (456 × 528 × 320 × N_features), ~500 MB
    │
    ├── models/{model_name}/                     ← trained region classifier
    │   ├── model.ubj
    │   └── meta.yaml
    │
    └── projects/{project}/                      ← per-project cluster aggregates
        ├── df_probe_details.pqt                 one row per probe insertion
        └── cells_aggregates/
            ├── clusters.table.pqt               all clusters (n_clusters × ~59)
            ├── clusters_good.table.pqt          QC-passing clusters (n_good × ~61)
            ├── clusters.acgs_log.npy            log-binned ACGs, normalised  (n_clusters × 128) float16
            ├── acgs_log.times.npy               ACG bin centres in seconds   (128,) float64
            ├── clusters.waveforms_peak.npy      peak-channel waveform        (n_clusters × 128) float16
            ├── clusters_good.stpc.npy           spike-triggered population coupling  (n_good × 1000) float16
            ├── clusters_good.stlfp.npy          spike-triggered LFP                  (n_good × 250)  float16
            ├── waveforms.voltage.npy            neighbourhood traces (~8 GB)         (n_traces × 128) float16
            └── waveforms.table.pqt              pid/cluster_id/abs_channel index     (n_traces × 3)

Versioning
----------

Channel features and encoding volumes are versioned with a **weekly label** of the form
``YYYY_Www`` (e.g. ``2025_W28``, ``2026_W12``).

.. code-block:: python

    import ephysatlas.data
    from one.api import ONE

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    labels = ephysatlas.data.list_available_labels(one=one, project='ea_active')
    latest = ephysatlas.data.get_latest_label(one=one, project='ea_active')

Projects
--------

Two main projects exist:

* ``ea_active`` — default project, updated weekly with the latest features
* ``ibl_neuropixel_brainwide_01`` — frozen brainwide map dataset

Download Functions
------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - S3 path
   * - :func:`ephysatlas.data.download_tables`
     - ``features/{project}/{label}/agg_full/``
   * - :func:`ephysatlas.data.download_encoding_volume`
     - ``encoding_volumes/{project}/{label}/``
   * - :func:`ephysatlas.data.download_probe_details`
     - ``projects/{project}/df_probe_details.pqt``
   * - :func:`ephysatlas.data.download_cells_features`
     - ``projects/{project}/cells_aggregates/``
   * - :func:`ephysatlas.data.download_project_data`
     - probe details + cell aggregates (convenience wrapper)

See also
--------

* :doc:`load-channel-features` — load channel-level features
* :doc:`load-cells-features` — load cells features (stPC, stLFP)
