Release a New Encoding Volume
==============================

This is a maintainer runbook for publishing a new encoding volume vintage to S3, so it
becomes available through :func:`ephysatlas.data.download_encoding_volume` (see
:doc:`load-encoding-volume`).

Source format (2026_W39 onward)
--------------------------------

The ``ea-encoder-channel`` model repo hands off a **sparse row-list** ``.h5`` file, not
the legacy dense ``.npz``. Each row is one in-brain, per-hemisphere voxel:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Shape
     - Description
   * - ``predicted_features``
     - (n_rows, N)
     - float32, raw (unstandardised) feature units
   * - ``feature_names``
     - (N,)
     - object, feature name strings
   * - ``ccf_xyz_index``
     - (n_rows, 3)
     - int16, (i, j, k) index into the dense CCF grid at ``atlas_resolution_um``
   * - ``ccf_xyz_m``
     - (n_rows, 3)
     - float32, CCF coordinates in metres
   * - ``region_id`` / ``hemisphere``
     - (n_rows,)
     - Allen region id / -1,+1 hemisphere (voxels with ``region_id == 0`` already excluded)

Root attrs carry provenance the ``.npz`` never had: ``model_commit``, ``model_repo_id``,
``model_id`` (vintage label), ``atlas_resolution_um``, ``release_checksums_sha1``.

Release steps
-------------

1. **Get the file locally** (e.g. ``~/Downloads/ephys_atlas_{res_um}um_{label}.h5``) and
   read its root attrs to confirm ``model_id``/vintage label and ``atlas_resolution_um``:

   .. code-block:: python

       import h5py
       with h5py.File(h5_path, "r") as f:
           print(dict(f.attrs))

2. **Convert to the legacy dense ``.npz`` schema.** This is a one-off release step, not
   library code, so it lives outside the package: run
   ``ibldevtools/olivier/2026-09-24_release_encoding_volume.py`` (update ``H5_PATH``/
   ``LABEL`` at the top for the new vintage first). It scatters ``predicted_features``
   into a ``(nx, ny, nz, N)`` grid by ``ccf_xyz_index`` (zeros outside the brain mask),
   recomputes ``mean_per_feature``/``std_per_feature`` over the populated voxels only,
   and asserts that the number of nonzero voxels in ``ephys_atlas_vol`` equals the
   source ``n_rows`` and that round-tripped values match ``predicted_features`` (within
   float16 precision) before it lets you proceed to upload.

3. **Upload both files** to the vintage's S3 prefix, using the ``ibl`` AWS profile:

   .. code-block:: bash

       aws s3 cp --profile ibl --only-show-errors brainwide_ephys_atlas_50um.npz \
         s3://ibl-brain-wide-map-private/aggregates/atlas/encoding_volumes/ea_active/2026_W39/brainwide_ephys_atlas_50um.npz

       aws s3 cp --profile ibl --only-show-errors ephys_atlas_50um_2026_W39.h5 \
         s3://ibl-brain-wide-map-private/aggregates/atlas/encoding_volumes/ea_active/2026_W39/brainwide_ephys_atlas_50um.h5

   The ``.h5`` upload is not consumed by any loader yet — it's published purely so the
   richer provenance/sparse format is archived and ready for the format migration below.

4. **Verify** through the actual public API, not just the raw file:

   .. code-block:: python

       from one.api import ONE
       from ephysatlas.data import download_encoding_volume
       import numpy as np

       one = ONE()
       fp = download_encoding_volume(local_path, label="2026_W39", one=one, overwrite=True)
       data = np.load(fp, allow_pickle=True)
       print(data["ephys_atlas_vol"].shape, len(data["feature_names"]))

5. **Update the vintage table** in :doc:`load-encoding-volume` (label, ``res_um``, grid
   shape, feature count if it changed).

Future: switching to the native ``.h5`` format
------------------------------------------------

The sparse row-list format is strictly richer than the dense ``.npz`` (no zero padding,
built-in provenance/checksums, easier to extend with per-voxel metadata) and is worth
adopting as the primary release format once the consumer side is ready. That's a
separate, deliberate migration, not a side effect of a routine release:

* Make ``_list_encoding_volume_resolutions``/``download_encoding_volume`` in
  ``ephysatlas/data.py`` format-aware (currently the discovery regex is
  ``.npz``-only).
* Teach the ``ea-load-encoding-volumes`` skill and :doc:`load-encoding-volume` the
  row-list access pattern instead of (or alongside) the dense-grid one.
* Keep emitting the legacy ``.npz`` for a transition period for any downstream
  consumer that hasn't been audited, then drop the h5-to-npz conversion step once
  nothing depends on it.
