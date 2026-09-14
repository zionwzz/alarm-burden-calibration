PYTHON ?= python3
export PYTHONPATH := $(CURDIR)
FEATURE_MAP ?= $${ALARM_WORK:-$(CURDIR)/work}/design/FEATURE_MAP_TUNE.json
FIGOUT ?= $${ALARM_FIGOUT:-$(CURDIR)/generated/figures}

.PHONY: help test simulation simulation-figures manuscript tuningmap analysis plotdata tables data-figures figures clean

help:
	@echo "Without the credentialed release:  make test | make simulation | make simulation-figures | make manuscript"
	@echo "With a credentialed working tree (ALARM_WORK): make tuningmap analysis plotdata tables data-figures"

# ---------------------------------------------------------------- no protected data needed
test:
	$(PYTHON) tools/test_linked_reward.py
	$(PYTHON) tools/test_final_theory.py
	$(PYTHON) -W ignore -m unittest tests.test_scanners tests.test_cohort_subset

# Computationally intensive: four million pseudo-patients per scenario for the population truths,
# 500 replicates per scenario, 400 per patient count. Writes simulation/results/ (not distributed)
# and then the simulation tables and findings paragraphs into generated/tables/.
simulation:
	$(PYTHON) simulation/run_submission_simulations.py
	$(PYTHON) simulation/make_sim_table.py

# Figures 2, S6 and S7 are drawn from simulation/results/ (run `make simulation` first).
simulation-figures:
	$(PYTHON) figures/fig2_simulation.py
	$(PYTHON) figures/figS6_simulation_laws.py
	$(PYTHON) figures/figS7_simulation_replicates.py

# The manuscript as submitted, from manuscript/ (pdfLaTeX and BibTeX).
manuscript:
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex && bibtex supplement && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

# ---------------------------------------------------------------- credentialed working tree needed
# The scans themselves: tools/run_scans.py --execute (see README.md). The targets below
# read a completed working tree (ALARM_WORK) and write under generated/.
tuningmap:
	$(PYTHON) tools/make_tuning_map.py
analysis: tuningmap
	ALARM_FEATURE_MAP="$(FEATURE_MAP)" $(PYTHON) analysis/link_analysis.py
	ALARM_FEATURE_MAP="$(FEATURE_MAP)" $(PYTHON) analysis/transport_diagnostics.py
plotdata:
	$(PYTHON) tools/export_plot_data.py
tables:
	$(PYTHON) tables/tables_cohort.py
	$(PYTHON) tables/tables_policy.py
	$(PYTHON) tables/tables_diagnostics.py
	$(PYTHON) tables/table_crossing.py
	$(PYTHON) tables/table_represent.py
data-figures:
	mkdir -p "$(FIGOUT)"
	cd manuscript/figures && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$(FIGOUT)" Figure1_framework.tex
	$(PYTHON) figures/fig3_holdout.py
	$(PYTHON) figures/fig4_combined.py
	$(PYTHON) figures/fig5_policy.py
	$(PYTHON) figures/figS1_study_flow.py
	$(PYTHON) figures/figS2_fit_grid.py
	$(PYTHON) figures/figS3_conventions.py
	$(PYTHON) figures/figS4_lag.py
	$(PYTHON) figures/figS5_grid.py
figures: simulation-figures data-figures

clean:
	find manuscript generated -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' -o -name '*.toc' \) -delete 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
