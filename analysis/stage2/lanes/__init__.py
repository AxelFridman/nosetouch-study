"""Stage 2 lanes: one folder of modules per modelling or analysis lane, discovered by ``stage2.pipeline``.

Per lane ``<id>`` (lower case, for example ``m2`` or ``x1``):

  lanes/<id>_*.py            the lane's code (features, models, experiments)
  lanes/<id>_pipeline.py     STEPS = {'compute': fn, 'figures': fn, 'tables': fn, 'numbers': fn}, ORDER, DEPENDS
  lanes/<id>_figures.py      one function per figure, named after it, presentation defaults in a FIGURES dict
  tests/test_<id>*.py        the lane's tests (under stage2/tests)

Run one lane with ``python -m stage2.pipeline compute --lanes <id>`` (then figures, tables, numbers).
The conventions and a template are in stage2/README.md, section 'Pipeline and lane conventions'.
"""
