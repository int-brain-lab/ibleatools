Basic Feature Extraction
========================

This guide shows you how to perform basic feature extraction from electrophysiological data using a probe insertion ID (PID).

Prerequisites
-------------

* ibleatools installed and configured
* Access to electrophysiological data
* A valid probe insertion ID (PID)

Overview
---------

The basic feature extraction workflow involves:
1. Loading data using a PID
2. Computing features from the raw electrophysiological signals
3. Saving the results for further analysis

Step-by-Step Guide
-------------------

1. **Import Required Modules**

   .. code-block:: python

      from pathlib import Path
      from one.api import ONE
      from ephysatlas.feature_computation import compute_features_from_pid

2. **Set Up Parameters**

   .. code-block:: python

      # Define your PID and parameters
      pid = "your-probe-insertion-id-here"
      output_dir = Path("/path/to/output/directory")
      
      # Optional: specify time range and snippet durations
      t_start = 300.0     # Start time in seconds
      duration_ap = 1.0   # AP snippet length in seconds
      duration_lf = 1.0   # LF snippet length in seconds

3. **Run Feature Extraction**

   .. code-block:: python

      # Compute features (pass a ONE client; it loads the raw data)
      one = ONE()
      result = compute_features_from_pid(
          pid=pid,
          one=one,
          output_dir=output_dir,
          t_start=t_start,
          duration_ap=duration_ap,
          duration_lf=duration_lf,
      )
      
      print(f"Features computed successfully!")
      print(f"Output saved to: {output_dir}")

4. **Verify Results**

   .. code-block:: python

      # Check what was created
      output_path = output_dir / pid
      if output_path.exists():
          print(f"Output directory created: {output_path}")
          
          # List generated files
          for item in output_path.rglob("*"):
              if item.is_file():
                  print(f"  {item.relative_to(output_path)}")

Complete Example
----------------

Here's the complete script from ``examples/feature_extraction_example.py``:

.. literalinclude:: ../../../examples/feature_extraction_example.py
   :language: python
   :caption: Complete basic feature extraction script

Customizing per-feature parameters
-----------------------------------

Per-feature options are passed via ``feature_params`` — either the typed objects
from ``ephysatlas.feature_calculators`` or an equivalent nested dict. Only the
options you set change; everything else keeps its default. For example, to
disable CSD scaling:

.. code-block:: python

   from ephysatlas.feature_calculators import FeatureParams, CsdParams

   result = compute_features_from_pid(
       pid=pid,
       one=one,
       t_start=t_start,
       duration_ap=duration_ap,
       duration_lf=duration_lf,
       feature_params=FeatureParams(csd=CsdParams(scale=False)),
       # equivalently: feature_params={"csd": {"scale": False}}
   )

Using the OOP calculators directly
------------------------------------

``compute_features_from_pid`` and ``compute_features_from_file`` are thin
wrappers over the calculators in ``ephysatlas.feature_calculators``. Use a
calculator directly when you also want the intermediate destriped snippet (for
inspection or plotting), or when working from local SpikeGLX files:

.. code-block:: python

   from ephysatlas.feature_calculators import (
       IBLPIDFeatureCalculator,
       FeatureComputationOptions,
       SnippetWindow,
   )

   calc = IBLPIDFeatureCalculator(pid=pid, one=one)
   window = SnippetWindow(t_start=300.0, duration_ap=1.0, duration_lf=1.0)
   options = FeatureComputationOptions(
       features_to_compute=["lf", "csd", "ap"], output_dir=output_dir
   )
   result = calc.compute_snippet(window, options)

   # Intermediate destriped data, without recomputing features:
   snippet = calc.get_destriped_snippet(window)

For local files use
:class:`ephysatlas.feature_calculators.SpikeGLXFileFeatureCalculator` (or
:func:`ephysatlas.feature_computation.compute_features_from_file`). The full OOP
script is in ``examples/feature_extraction_oop.py``.

Expected Output
---------------

When successful, you should see:
* A new directory structure created under your output directory
* Feature files in various formats (`.pqt`, `.npy`)
* Log messages indicating successful completion
* No error messages

Troubleshooting
---------------

**Common Issues:**

* **PID not found**: Ensure the PID exists in your data repository
* **Permission errors**: Check write permissions for the output directory
* **Memory issues**: For large datasets, consider processing smaller time chunks
* **Missing dependencies**: Ensure all required packages are installed

**Getting Help:**

* Check the logs for detailed error messages
* Verify your data access permissions
* Consult the :doc:`../reference/index` for detailed API documentation

Next Steps
-----------

.. After basic feature extraction, you might want to:
.. * :doc:`visualization-guides` - Visualize your extracted features
.. * :doc:`feature-extraction-examples` - Explore more advanced extraction options
.. * :doc:`../reference/index` - Learn about all available functions and parameters
