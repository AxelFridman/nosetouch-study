# analysis/reports/

Output of the scripts in `analysis/scripts/`, one folder per run (git-ignored except this README:
reports can contain participant-level numbers).

| Folder (default name) | Written by | Contents |
|---|---|---|
| `qa_<data>/` | `qa_report.py <data>` | `qa_report.md`, `tables/*.csv`, `summary.json`, `sampling.png`, `counts.png`, `durations.png` |
| `baseline_<data>/` | `train_baseline.py <data>` | `report.md`, `metrics.csv`, `subgroups.csv`, `mirror_consistency.csv`, `oof_<scheme>.csv`, `participant_accuracy.csv`, `exclusions.csv`, `features.csv`, `roc.png`, `subgroups.png`, `chiral_by_hand*.png` |
| `trials/` | `plot_trial.py <data> --trial <id>` | `trial_<trial_id>.png` |

`--out DIR` writes anywhere else. Everything here can be regenerated from the data, so delete freely.
In the CSV files, text that a spreadsheet would run as a formula (`=`, `+`, `-`, `@`, tab, line break
first) starts with a single quote: the strings come from records posted to a public endpoint.
When a result is reported, keep the report together with the data snapshot name, the git commit and
`pip freeze` output that produced it.
