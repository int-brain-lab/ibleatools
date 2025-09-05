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

      from ephysatlas.feature_computation import compute_features_from_pid
      from pathlib import Path

2. **Set Up Parameters**

   .. code-block:: python

      # Define your PID and parameters
      pid = "your-probe-insertion-id-here"
      output_dir = Path("/path/to/output/directory")
      
      # Optional: specify time range and duration
      t_start = 300.0  # Start time in seconds
      duration = 5.0   # Duration in seconds

3. **Run Feature Extraction**

   .. code-block:: python

      # Compute features
      result = compute_features_from_pid(
          pid=pid,
          output_dir=output_dir,
          t_start=t_start,
          duration=duration
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
