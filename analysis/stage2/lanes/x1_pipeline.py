"""Lane x1: exploratory, descriptive case study of one participant with Parkinson's disease (PD01); no model.

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes x1            # analyses whose outputs are stale
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes x1 --force    # everything (rebuilds PD01's cache too)
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes x1 [--only X1_03_spectra]
    .venv/Scripts/python.exe -m stage2.pipeline tables --lanes x1
    .venv/Scripts/python.exe -m stage2.pipeline numbers --lanes x1            # numbers, finalists ([]) and the summary

Steps: compute (``x1_compute.run``: the exploratory cache, per-trial characteristics, matched comparisons, spectra,
traces, yaw curves, multivariate map, task behaviour), figures (``x1_figures``: from saved files only), tables
(``x1_tables``: CSV + booktabs), numbers (``x1_report``: results/x1_numbers.json, results/finalists_x1.json - empty,
this lane has no model - and reports/stage2/x1_SUMMARY.md).
"""
from stage2.pipeline import limit_threads

limit_threads(1)                    # lane x1's thread budget, before numpy loads

ORDER = 500                         # exploratory lanes run after the model lanes
DEPENDS = ()


def compute(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import x1_compute
    x1_compute.run(force, only)


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import x1_figures
    x1_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import x1_tables
    x1_tables.write(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import x1_report
    x1_report.write_all(only)


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
