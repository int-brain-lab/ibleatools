Load Cluster Features (Waveforms, ACGs, Burstiness)
====================================================

This guide covers downloading and loading the **cluster-level feature aggregates**:
peak-channel waveforms, neighbourhood waveforms, log-binned autocorrelograms (ACGs),
waveform shape features, and burstiness / memory metrics computed across all
IBL insertions.

S3 Folder Architecture
-----------------------

.. code-block:: text

    aggregates/atlas/projects/{project}/
    │
    ├── df_probe_details.pqt                    one row per insertion
    │
    ├── cell_aggregates/                        cluster-level aggregates (~1.5 GB)
    │   ├── good_clusters.pqt                   [existing] QC-passing clusters
    │   ├── good_stpc.npy                       [existing] spike-triggered population coupling
    │   ├── good_stlfp.npy                      [existing] spike-triggered LFP
    │   │
    │   ├── df_clusters.pqt                     all clusters after ssl merge  (925 k × 35)  ~500 MB
    │   ├── avg_waveform_features.pqt           waveform shape features       (925 k × 23)   ~50 MB
    │   ├── df_clusters_extended.pqt            burstiness + memory           (925 k ×  2)   ~10 MB
    │   ├── acgs_log_bins.npy                   log-binned ACGs               (925 k × 128) ~290 MB
    │   ├── acgs_log_times.npy                  ACG bin centres in seconds    (128,)           tiny
    │   └── avg_waveform_peak_channel.npy       peak-channel waveform trace   (925 k × 128) ~180 MB
    │
    └── cell_waveforms/                         full neighbourhood waveforms (~10 GB, optional)
        ├── avg_waveforms.npy                   all neighbourhood traces      (36 M × 128)    ~8 GB
        └── avg_waveforms_index.pqt             pid / cluster_id / abs_channel index (36 M × 3) ~1.5 GB

.. note::

    ``cell_waveforms/`` is only needed when working with full per-channel neighbourhood
    traces (e.g. to reconstruct the spatial waveform across the probe shank).
    Most analyses only need ``cell_aggregates/``.

Array / Table Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 10 35

   * - File
     - Shape
     - Dtype
     - Description
   * - ``df_clusters.pqt``
     - (n_clusters, 35)
     - mixed
     - Cluster table after SpikeSortingLoader merge; includes QC metrics, brain-region coordinates, ``pid``, ``bitwise_fail``
   * - ``avg_waveform_features.pqt``
     - (n_clusters, 23)
     - float32
     - Waveform shape features from ``ibldsp.waveforms.compute_spike_features``: peak/trough timing, slopes, half-width, ``axial_um``, ``lateral_um``, ``peak_channel``
   * - ``df_clusters_extended.pqt``
     - (n_clusters, 2)
     - float32
     - ``burstiness`` and ``memory`` from ISI statistics
   * - ``acgs_log_bins.npy``
     - (n_clusters, 128)
     - float32
     - Log-binned ACG in **spike pairs · s⁻¹** (raw coincident pair counts divided by
       log-bin width in seconds, not normalised by recording duration or firing rate).
       At long lags (τ > 1 s) the asymptotic value converges to λ² × T = λ × n_spikes,
       where λ is the firing rate and T the recording duration.
       To obtain a dimensionless ACG normalised to the Poisson baseline, divide each row
       by ``spike_count`` from ``df_clusters``; the resulting units are sp/s and the
       long-lag asymptote equals the firing rate.
       Rows are aligned with ``df_clusters``.
   * - ``acgs_log_times.npy``
     - (128,)
     - float64
     - Geometric bin centres in seconds, shared across all insertions (range ≈ 1 ms – 2 s)
   * - ``avg_waveform_peak_channel.npy``
     - (n_clusters, 128)
     - float32
     - Average waveform on the peak channel only; rows aligned with ``df_clusters``
   * - ``avg_waveforms.npy``
     - (n_traces, 128)
     - float32
     - All neighbourhood-channel traces flattened; use ``avg_waveforms_index.pqt`` to index
   * - ``avg_waveforms_index.pqt``
     - (n_traces, 3)
     - mixed
     - ``pid``, ``cluster_id``, ``abs_channel`` — row index into ``avg_waveforms.npy``

Downloading
-----------

Use :func:`ephysatlas.data.download_cell_features` for the standard aggregates and
:func:`ephysatlas.data.download_cell_waveforms` for the optional full waveforms:

.. code-block:: python

    from pathlib import Path
    from one.api import ONE
    import ephysatlas.data

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    # ~1.5 GB — cluster tables, ACGs, peak-channel waveforms
    ephysatlas.data.download_cell_features(local_path, project=project, one=one)

    # ~10 GB — full neighbourhood waveforms (optional)
    ephysatlas.data.download_cell_waveforms(local_path, project=project, one=one)

Loading
-------

.. code-block:: python

    from pathlib import Path
    import ephysatlas.data

    local_path = Path('/datadisk/ephys-atlas')
    project = 'ibl_neuropixel_brainwide_01'

    # Returns (df_clusters, df_wf_features, df_clusters_extended,
    #          acgs_log_bins, acgs_log_times, avg_waveform_peak_channel)
    result = ephysatlas.data.read_cluster_features(local_path / project)

    df_clusters               = result['df_clusters']
    df_wf_features            = result['df_wf_features']
    df_clusters_extended      = result['df_clusters_extended']
    acgs_log_bins             = result['acgs_log_bins']          # (n_clusters, 128) float32
    acgs_log_times            = result['acgs_log_times']         # (128,)  seconds
    avg_waveform_peak_channel = result['avg_waveform_peak_channel']  # (n_clusters, 128)

ACG normalisation
~~~~~~~~~~~~~~~~~

The raw ACG values are in **spike pairs · s⁻¹** and scale with firing rate and recording
duration (asymptote = λ² × T).  For most analyses you will want to normalise to a
dimensionless shape comparable across neurons:

.. code-block:: python

    # Divide each row by spike_count → long-lag asymptote ≈ firing rate (sp/s)
    spike_count = df_clusters['spike_count'].values[:, np.newaxis]
    acgs_norm = acgs_log_bins / spike_count      # sp/s, asymptote ≈ firing_rate

    # Further divide by firing_rate → dimensionless, asymptote ≈ 1
    firing_rate = df_clusters['firing_rate'].values[:, np.newaxis]
    acgs_dimless = acgs_norm / firing_rate        # dimensionless, asymptote ≈ 1

Restricting to good units
~~~~~~~~~~~~~~~~~~~~~~~~~

``df_clusters`` contains all sorted clusters. Filter to QC-passing units with:

.. code-block:: python

    good = df_clusters['bitwise_fail'] == 0
    df_good          = df_clusters[good]
    wf_good          = avg_waveform_peak_channel[good.values]
    acgs_good        = acgs_log_bins[good.values]

Loading full neighbourhood waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    # Memory-map recommended — array is ~8 GB
    result_wf = ephysatlas.data.read_cell_waveforms(local_path / project)
    avg_waveforms       = result_wf['avg_waveforms']        # np.memmap (n_traces, 128)
    avg_waveforms_index = result_wf['avg_waveforms_index']  # DataFrame  (n_traces, 3)

    # Example: extract all neighbourhood traces for a single cluster
    pid, cid = df_clusters['pid'].iloc[0], df_clusters.index[0]
    mask = (avg_waveforms_index['pid'] == pid) & (avg_waveforms_index['cluster_id'] == cid)
    cluster_traces = avg_waveforms[mask.values]             # (nc, 128)
