Load Encoding Volumes
======================

This guide covers downloading and loading **encoding volumes** — a pre-computed 4-D
volumetric representation of electrophysiological features on the Allen Common
Coordinate Framework (CCF).

S3 Layout
---------

.. code-block:: text

    aggregates/atlas/encoding_volumes/{project}/{label}/
    └── brainwide_ephys_atlas_{res_um}um.npz   4-D volume (nx, ny, nz, N_features)

Encoding volumes are versioned independently by **vintage label** (``label``) and
**voxel resolution** (``res_um``, in µm). Available vintages:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - ``label``
     - ``res_um``
     - Grid shape
     - S3 file
   * - ``2026_W12``
     - 25
     - (456, 528, 320)
     - ``brainwide_ephys_atlas_25um.npz``
   * - ``2026_W26``
     - 50
     - (228, 264, 160)
     - ``brainwide_ephys_atlas_50um.npz``

Downloading
-----------

.. code-block:: python

    from pathlib import Path
    import numpy as np
    from one.api import ONE
    from ephysatlas.data import download_encoding_volume

    one = ONE()
    local_path = Path("/path/to/local/storage")

    # res_um omitted -> auto-resolves to the finest resolution available for this label
    file_path = download_encoding_volume(local_path, label="2026_W26", one=one)
    data = np.load(file_path, allow_pickle=True)  # allow_pickle required for feature_names

Pass ``res_um`` explicitly to pick a specific resolution when a vintage has more than one:

.. code-block:: python

    file_path = download_encoding_volume(local_path, label="2026_W12", res_um=25, one=one)

Loading
-------

The file contains the following arrays (N = number of features for the vintage, e.g. 41 for
``2026_W12`` and ``2026_W26``):

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 35

   * - Key
     - Shape
     - Dtype
     - Description
   * - ``ephys_atlas_vol``
     - (nx, ny, nz, N)
     - float16
     - 4-D volume: x × y × z × features
   * - ``feature_names``
     - (N,)
     - object
     - Feature name strings
   * - ``mean_per_feature``
     - (N,)
     - float32
     - Per-feature normalisation mean
   * - ``std_per_feature``
     - (N,)
     - float32
     - Per-feature normalisation std deviation
   * - ``grid_shape``
     - (3,)
     - int32
     - Volume grid dimensions [nx, ny, nz]
   * - ``res_um``
     - (1,)
     - int32
     - Voxel resolution in µm

.. code-block:: python

    vol = data["ephys_atlas_vol"]
    feature_names = data["feature_names"]
    idx = np.where(feature_names == "rms_ap")[0][0]
    rms_ap_volume = vol[..., idx]   # (nx, ny, nz) float16

Values are stored in raw (unnormalised) feature units, with ``0.0`` outside the brain
mask. ``mean_per_feature`` / ``std_per_feature`` are provided for optional z-scoring —
they are not pre-applied to ``ephys_atlas_vol``.

See also
--------

* :doc:`s3-architecture` — complete S3 folder layout
* :doc:`load-channel-features` — channel-level (tabular) features
