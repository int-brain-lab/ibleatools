"""Automatic quality control of the Ephys Atlas insertions.

Every insertion of a feature release is labelled per component -- ``alignment``, ``lf``, ``ap``,
``waveform`` -- as ``good``, ``review`` or ``bad`` from the data alone: each test compares the
probe's recording, at its positions, with what the other probes record in the same brain regions
(:mod:`.regions`). No model, no auto-alignment and no manual label enters a decision. :mod:`.levels`
turns the tests into five nested levels of sensitivity, L1 to L5.

- :mod:`.data` -- the insertions of a release and their mice;
- :mod:`.regions`, :mod:`.measures` -- the region references and the measurements;
- :mod:`.levels` -- the tests, their z-scores and the levels;
- :mod:`.pipeline` -- the stages, cached on disk; :mod:`.cases` -- example figures;
- :mod:`.fixture` -- writes the labels at one level as ``ephysatlas.fixtures.pid_qc_filter``.

``scripts/build_pid_qc_filter.py`` runs everything.
"""
