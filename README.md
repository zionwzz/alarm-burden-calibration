# alarm-burden-calibration

Code for *Alarm burden beyond threshold replay: linked-reward calibration of recorded clinical
alarms* (Zihao Wang, Michele Ceccarelli and Min Lu; Division of Biostatistics and Bioinformatics,
University of Miami).

Retrospective alarm studies estimate the effect of changing a bedside-monitor alarm limit by
replaying the candidate threshold on stored vital-sign measurements. A threshold excursion is not
necessarily annunciated, so replay counts excursions the monitor never alarmed on. Linked-reward
calibration anchors replay to the alarms the monitor recorded: every replayed excursion is kept,
including those with no linked alarm, which contribute zero linked seconds, and the baseline linked
alarm time is standardised over the excursions the candidate limit generates under explicit
transport and overlap conditions. The estimator, its patient-cluster bootstrap, the scan of the
telemetry release that produces its inputs, the simulation study and the figure and table generators
are all here.

## What is here, and what is not

The repository holds the analysis and simulation code, the synthetic tests and the figure-generation
sources. It does not hold the ALOTT records, any derived patient- or
admission-level dataset, the study's cohort table or split record, the fitted calibration maps, the
aggregate analysis results the tables and figures were generated from, or the saved simulation
replicates. The ALOTT database (version 1.0.0) is distributed by PhysioNet to credentialed users under
its Restricted Health Data Use Agreement and License; a reanalysis needs that access together with the
study-specific cohort and split records. The simulation, its tables and its figures regenerate from
the code alone.

## Layout

| Directory | Contents |
|---|---|
| `alarmreplay/` | The package: the linked-reward estimator and bootstrap (`linked_reward.py`), the conventions shared by every scanner (`scan_common.py`), the lag-truncation routines (`truncation.py`), the cohort-restriction hook (`cohort.py`), the figure style (`figstyle_lrc.py`), the simulation scenario lettering (`scenarios.py`) and the filesystem locations (`paths.py`). |
| `scan/` | The eight scans of the telemetry release: alarm epidemiology, delay-model fitting grid, crossing-to-annunciation lead, linkage, linked features, replay at candidate limits (marginal and joint with overshoot) and the measurement-convention scan. Each is resumable and writes one row file per stripe. |
| `analysis/` | The downstream analyses: delay-model selection and its held-out adjudication, the tuning-set feature map, the linked-reward analysis over the limit-shift grid, transport and support diagnostics, the stochastic-order check, stability and influence checks, and the oncology strata. |
| `simulation/` | The simulation study: data-generating laws and estimators (`simulate_linked_reward.py`), the submitted runs with their seeds (`run_submission_simulations.py`) and the generator of the simulation tables and findings paragraphs (`make_sim_table.py`). |
| `figures/` | One script per data figure and per simulation figure, and the page geometry every figure is built to (`figpage.py`). |
| `tables/` | The generators of the manuscript's data tables. |
| `tests/`, `tools/test_*.py` | Synthetic tests: every scanner end to end on a synthetic release, the shared conventions, the cohort hook, the estimator, and numerical checks of the algebra and the linkage conventions. |
| `tools/` | The scan driver, the completeness and joint-cell identity checks, the remover of duplicated row blocks after an interrupted scan, the tuning-map builder, the plot-data exporter, the comparison of the prespecified and in-sample maps, and the cohort-restriction tools. |

`generated/` (tables, figures and plot data written by the generators), `work/` (a working tree),
`data/` and `simulation/results/` are created on demand and are not tracked.

## Without the ALOTT release

Python 3.11 with `pip install -r requirements.txt` (NumPy, SciPy, Matplotlib), and a TeX
distribution with pdfLaTeX, BibTeX and TikZ for the manuscript and the framework diagram.

```sh
make test                 # estimator, algebra and linkage checks; the scanners on a synthetic release
make simulation           # the submitted simulation runs, then the simulation tables and findings text
make simulation-figures   # Figures 2, S6 and S7 from the simulation results
```

`make simulation` is computationally intensive: four million pseudo-patients per scenario define the
population truths, each of the nine scenarios uses 500 replicates with 200 patient-cluster bootstrap
resamples in the first 400, and the patient-count sweeps use 400 replicates per size. Every seed is
fixed in `run_submission_simulations.py`; replicate-level estimates and intervals are saved so that
coverage can be rescored without regenerating data. The simulation keys its scenarios by the letters
of the code (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `I`, `J`), which the seeds are tied to; the manuscript
letters the same scenarios `A` to `I` in the order of its design table, and `alarmreplay/scenarios.py`
holds the correspondence that the tables and figures print through. The simulation tables and
findings paragraphs in the manuscript were generated by `simulation/make_sim_table.py` and then edited
for the final text; the numbers are the generated ones.

## With authorised ALOTT access

The scripts locate their inputs through environment variables, documented in `alarmreplay/paths.py`:
`ALARM_TELEMETRY` (the release's telemetry directory), `ALARM_COHORT` (the study cohort table, one
row per admission with `aMRN`, `aCSN` and the diagnosis-derived tier), `ALARM_DIAGNOSES`,
`ALARM_COVERAGE` and `ALARM_CANCER_SITES` (used only by the cancer-group derivation and the stability
checks), and `ALARM_WORK`, the working tree the scans and analyses write to. The design directory of
the working tree holds the patient-level split assignment and the design record, which are fixed
before any model is fitted.

The scans run through the driver, which plans by default and executes in bounded, resumable rounds:

```sh
python3 tools/run_scans.py --work <work> --telemetry <telemetry> --cohort <cohort table> \
    --design-dir <design> --stripes 4 --execute
python3 tools/cohort_completeness.py --work <work> --cohort <cohort table>
python3 tools/assert_joint_consistency.py --work <work> --cohort <cohort table> --baseline-tag FIT4p0
```

The second and third commands check that every job covered its intended admissions and that the
duration-by-overshoot cell counts of the joint replay agree with the feature scan. The downstream
analyses then run in this order, with `ALARM_WORK` set:

```sh
python3 analysis/aggregate_alarm_epidemiology.py
python3 analysis/select_delay_model.py
python3 analysis/adjudicate_test.py                    # the held-out split, read once
python3 analysis/aggregate_measurement_conventions.py
make tuningmap analysis                                # the tuning-set feature map, then the linked-reward
                                                       # analysis and the transport diagnostics under it
python3 analysis/st_order_check.py
python3 analysis/cancer_groups.py && python3 analysis/oncology_strata.py
python3 analysis/stability_checks.py 1 && python3 analysis/stability_checks.py 2
make plotdata tables data-figures                      # generated/plot_data, generated/tables, generated/figures
```

The linked-reward analysis takes the tuning-set feature map through `ALARM_FEATURE_MAP`; run without
it, in a working tree of its own, it selects the map in sample, which is the sensitivity the
manuscript reports (`ALARM_INSAMPLE_LINK_ANALYSIS` names that run's output for the tables).
`analysis/link_analysis.py` also accepts `ALARM_COHORT_SUBSET`, a file naming admissions by `aCSN`,
to restrict an arm to a subgroup, with `tools/make_cohort_subset.py` and
`tools/matched_size_control.py` beside it; the submitted analysis uses every admission of the analysis
subsample.

The table and figure generators read a completed working tree, or a copy of its analysis records in the
same layout, and write under `generated/`; the manuscript's captions and notes are editorial, and the
numbers are the generated ones.

## Data and ethics

ALOTT: Lawrence J, Rayo M, Huerta T. ALarms, Outcomes Telemetry with Timing (ALOTT): a Bedside-EMR
Database (version 1.0.0). PhysioNet, 2025. https://doi.org/10.13026/sbq5-dy17. PhysioNet: Goldberger
et al., *Circulation* 2000; Pollard et al., *Nature Health* 2026, https://doi.org/10.1038/s44360-026-00096-z.
The study analysed the de-identified release under its data-use agreement, with no patient contact and
no attempt at re-identification, and nothing in this repository identifies a patient, an admission or
a clinician. Users of the code who hold ALOTT access remain bound by their own agreement with
PhysioNet; in particular, no derived dataset that carries the release's identifiers may be shared.

## Citation and license

Please cite the manuscript when using this code; `CITATION.cff` carries the reference. The code is
released under the MIT License.
