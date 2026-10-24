# analysis/data/

Local data for the analysis kit. **Never commit participant data.** Git ignores everything in this
folder except this README (`analysis/data/*` in the root `.gitignore`); keep it that way, and do not
copy data into other tracked folders, notebooks or issue trackers.

Suggested layout:

```
data/
  synthetic/                regenerated on demand, never edited:  cd app && npm run synth -- --participants 12 --preset street --out ../analysis/data/synthetic
  synthetic_v2/             the same for protocol 2.0.0 visits (--preset street)
  live/                     the Heroku database, replaced by every pull:  python analysis/scripts/pull_live.py
  real/
    2026-11-20/             a dated, read-only snapshot (copy data/live/ before analysing it)
    exports/                export bundles participants shared from the app (nosetouch_*.json / .json.txt)
    lab/                    copies of server/received/ from lab sessions
```

Every loader accepts any mix of these (folder trees `<study>/<participant>/<session>/{session.json,
trial_*.json}`, export bundles, zip archives); duplicates across sources are dropped by record id.

Rules (docs/02 sections 9-10, docs/06):

* Snapshots are read-only: never edit raw records. Keep manual corrections (exclusions, relabels) in a
  separate versioned table keyed by `trial_id`.
* Synthetic records carry `simulated: true` and the study code `synthetic-v1` (the app) or
  `synthetic-py` (`motionlab.synth`). Filter them out of every real-data analysis
  (`--simulated exclude`).
* Deletion on request: remove `<study_code>/<participant_id>/` from every snapshot and every derived copy
  (and from the database).
* From protocol 2.0.0 on the raw records carry the participant's full name (`participant.nickname`). The
  tables of `motionlab` never include it; never share raw records beyond the lab, and share derived
  tables only without names and without `device.user_agent`.
