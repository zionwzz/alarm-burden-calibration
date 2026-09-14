"""Filesystem locations, resolved from the environment.

Inputs of the credentialed analysis (none is distributed with this code):

ALARM_TELEMETRY  directory of the telemetry release (one subdirectory per patient,
                 then per admission, holding ALARM*.csv.gz and MEASUREMENT*.csv.gz)
ALARM_COHORT     cohort table (one row per admission: aMRN, aCSN, tier, ...)
ALARM_DIAGNOSES  directory of per-admission diagnosis tables, used only to derive cancer groups
ALARM_COVERAGE   reference-stream coverage table, used only by the stability checks
ALARM_CANCER_SITES  an earlier set of cancer-site labels (aCSN, cancer_site), used only to report
                 their agreement with the diagnosis-derived groups
ALARM_WORK       working tree for intermediate scan rows and analysis outputs; defaults to ./work
                 under the repository root. The table and figure generators read the analysis
                 records from this tree (policy/, delay_model/, comparison/, sensitivity/,
                 diagnostics/, alarm_epidemiology/ and design/), so a copy of a completed run's
                 de-identified records in the same layout serves them as well as the run itself.

Outputs of the generators (created on demand, ignored by version control):

ALARM_PLOTDATA   the figure aggregates tools/export_plot_data.py writes and the data figures read;
                 defaults to generated/plot_data
ALARM_FIGOUT     directory the figure scripts write to; defaults to generated/figures
ALARM_TABLES     directory the table generators write to; defaults to generated/tables

The working tree keeps the subdirectory names the analysis scripts use, so a run can
be resumed or inspected without a database.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TELEMETRY = os.environ.get("ALARM_TELEMETRY", os.path.join(REPO, "data", "telemetry-data"))
COHORT = os.environ.get("ALARM_COHORT", os.path.join(REPO, "data", "cohort.csv"))
DIAGNOSES = os.environ.get("ALARM_DIAGNOSES", os.path.join(REPO, "data", "admission-diagnoses"))
COVERAGE = os.environ.get("ALARM_COVERAGE", os.path.join(REPO, "data", "reference_stream_coverage.csv"))
CANCER_SITES = os.environ.get("ALARM_CANCER_SITES", os.path.join(REPO, "data", "admission_subgroups.csv"))   # earlier cancer-site labels, for the agreement check only
WORK = os.environ.get("ALARM_WORK", os.path.join(REPO, "work"))

GENERATED = os.path.join(REPO, "generated")
PLOTDATA = os.environ.get("ALARM_PLOTDATA", os.path.join(GENERATED, "plot_data"))
FIGOUT = os.environ.get("ALARM_FIGOUT", os.path.join(GENERATED, "figures"))
TABLES = os.environ.get("ALARM_TABLES", os.path.join(GENERATED, "tables"))

DESIGN = os.path.join(WORK, "design")     # the split assignment, the design record and the tuning map
EPIDEMIOLOGY = os.path.join(WORK, "alarm_epidemiology")
DELAY_MODEL = os.path.join(WORK, "delay_model")
COMPARISON = os.path.join(WORK, "comparison")
REPLAY = os.path.join(WORK, "replay")
SENSITIVITY = os.path.join(WORK, "sensitivity")
POLICY = os.path.join(WORK, "policy")
SIMULATION = os.path.join(WORK, "simulation")
DIAGNOSTICS = os.path.join(WORK, "diagnostics")
CACHE = os.path.join(WORK, "cache")

SPLIT_ASSIGNMENT = os.path.join(DESIGN, "SPLIT_ASSIGNMENT.csv")


def ensure(*paths):
    """Create the given directories if they do not exist and return them."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return paths if len(paths) > 1 else paths[0]
