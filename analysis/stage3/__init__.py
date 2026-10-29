"""Stage 3, Python side: the production models exported for the web page and the golden fixtures.

    cd analysis
    .venv/Scripts/python.exe -m stage3 export      # fit on ALL persons, write app/src/predict/models/*.json
    .venv/Scripts/python.exe -m stage3 fixtures    # golden fixtures: synthetic (committed) and real (local only)
    .venv/Scripts/python.exe -m stage3 all         # both
    .venv/Scripts/python.exe -m pytest stage3/tests -q -p no:cacheprovider

Modules (Python is the ground truth the TypeScript port reproduces: docs/stage3/PORT_SPEC.md):

* ``fit``       - every model fitted on all persons (P01..P20, docs/09 D2 cap) with the code paths that were
                  evaluated in Stage 2: lane m4's ``fit_production``, lane m1's ``make_linear_select`` (PHYS6) and the
                  registry reference ``example_yaw_logreg``; cached under ``analysis/data/stage3_cache`` (git-ignored);
* ``export``    - the JSON files of ``app/src/predict/models/`` and their manifest;
* ``runtime``   - the models as Python objects, either the in-memory fits or REBUILT FROM THE JSON (the sklearn and
                  Stage 2 classes with the exported parameters), behind one interface;
* ``reference`` - ``reference_predict``: every intermediate of the production path for one stream, computed by the
                  Stage 2 functions themselves (``standardize.stream_from_record``, ``m4_detect``, ``m4_e2e``,
                  ``m4_online``, ``final_infer``, ``m1_features``) and checked against ``m4_production.predict_stream``;
* ``fixtures``  - synthetic streams (``motionlab.synth``) and real recordings with their reference outputs.

Privacy: nothing here uploads anything; real recordings and anything derived from them per recording are written only
to ``analysis/data/stage3_fixtures`` (git-ignored). Persons appear as P01..P20 only; the exploratory persons are never
read.
"""
