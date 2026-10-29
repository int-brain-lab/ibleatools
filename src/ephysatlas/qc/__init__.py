"""Automatic quality control of the Ephys Atlas insertions.

Every insertion of a feature release is labelled per component -- ``alignment``, ``lf``, ``ap``,
``waveform`` -- as ``good``, ``review`` or ``bad`` from the data alone: each failure case
compares the probe's recording, at its positions, with what the other probes record around the
same positions (:class:`~.modelfree.LocalReference`), or its signal with the signals of all probes
(:mod:`.signal`). No model, no auto-alignment and no manual label enters a decision. :mod:`.levels`
turns the failure cases into robust z-scores and five nested levels of sensitivity, L1 to L5.

- :mod:`.data` -- the insertions of a release and their sessions;
- :mod:`.signal`, :mod:`.regions`, :mod:`.modelfree` -- the measurements;
- :mod:`.levels` -- the failure cases, their calibration and the levels;
- :mod:`.pipeline` -- the stages, cached on disk; :mod:`.cases` -- example figures;
- :mod:`.fixture` -- writes the labels at one level as ``ephysatlas.fixtures.pid_qc_filter``.

``scripts/build_pid_qc_filter.py`` runs everything.
"""
