"""Lane sec: the secondary tasks: instructed speed, instructed grip, and handedness / the dominant hand.

Steps (README section 11):
* compute: the experiments (multi-class speed and grip models on the leave-one-person-out splits, the dominant-hand
  model through cv.run_experiment, its label-permutation baseline) and every derived file (``sec_compute``);
  current results are skipped unless ``--force``; ``--only`` takes experiment names (``sec_speed_*``,
  ``sec_dom_null``);
* figures: SEC_NN_* from the saved files only (``sec_figures``);
* tables: CSV + booktabs .tex (``sec_report``);
* numbers: results/sec_numbers.json, then sec_SUMMARY.md and results/finalists_sec.json (an empty list: no hand
  model comes from this lane).

    python -m stage2.pipeline compute --lanes sec
    python -m stage2.pipeline figures --lanes sec
    python -m stage2.pipeline tables --lanes sec
    python -m stage2.pipeline numbers --lanes sec

The compute step re-runs the app's segmentation on every stored stream once (``sec_reseg``: label-independent
windows, cached; ``--only sec_reseg --force`` repeats it).
"""
from stage2.pipeline import Lane, limit_threads

limit_threads(2)                                   # the lane's thread budget, BEFORE numpy loads

L = Lane('sec')
ORDER = 920
DEPENDS = ()


def compute(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import sec_compute
    sec_compute.compute(force, only)


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import sec_figures
    sec_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import sec_report
    sec_report.tables(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import sec_report
    sec_report.numbers()
    sec_report.write_summary()
    L.write_finalists([])


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
