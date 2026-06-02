Load Channel Features
=====================

This guide covers downloading and loading the **channel-level ephys feature aggregates** —
the core dataset of the IBL Ephys Atlas. Each row is one recording channel from one probe
insertion, annotated with brain region and electrophysiological features.

S3 Layout
---------

.. code-block:: text

    aggregates/atlas/features/{project}/{label}/agg_full/
    ├── raw_ephys_features_denoised.pqt   denoised per-channel features  (~200 MB)
    ├── raw_ephys_features.pqt            raw per-channel features
    ├── channels.pqt                      channel metadata (coordinates, probe info)
    └── channels_labels.pqt               region labels per channel

Features are versioned with a weekly ``YYYY_Www`` label. Use
:func:`ephysatlas.data.get_latest_label` to find the most recent vintage.

Downloading
-----------

.. code-block:: python

    from pathlib import Path
    from one.api import ONE
    import ephysatlas.data

    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    local_path = Path('/datadisk/ephys-atlas/features')
    label = '2025_W28'

    # downloads to local_path/ea_active/{label}/agg_full/
    path_features = ephysatlas.data.download_tables(local_path, label=label, one=one)

Pass ``extended=True`` to also fetch the ``{label}_extended/`` folder (large optional
datasets such as cross-correlograms):

.. code-block:: python

    ephysatlas.data.download_tables(local_path, label=label, one=one, extended=True)

Loading
-------

.. code-block:: python

    import ephysatlas.data
    import ephysatlas.anatomy

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df_features = ephysatlas.data.read_features_from_disk(
        path_features, brain_atlas=brain_atlas
    )

:func:`ephysatlas.data.read_features_from_disk` merges ``raw_ephys_features_denoised.pqt``,
``channels.pqt``, and ``channels_labels.pqt`` into a single DataFrame and annotates every
channel with ``Allen_id``, ``Cosmos_id``, and ``Beryl_id`` brain region IDs.

Listing available vintages
--------------------------

.. code-block:: python

    labels = ephysatlas.data.list_available_labels(one=one, project='ea_active')
    print(labels)   # ['2024_W50', '2025_W10', '2025_W28', ...]

    latest = ephysatlas.data.get_latest_label(one=one, project='ea_active')

See also
--------

* :doc:`s3-architecture` — complete S3 folder layout
* :doc:`load-cells-features` — cells features (stPC, stLFP)