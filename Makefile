# Convenience targets for paper figure generation and build.
#
# Usage:
#   make plot          # generate every paper figure; placeholder for missing data
#   make plot-real     # only render figures whose data is on disk (no placeholders)
#   make paper         # plot + latexmk in paper/ (assumes pdflatex installed)
#   make paper-clean   # remove paper/build/

POETRY := poetry
PYTHON := $(POETRY) run python
PLOT_SCRIPT := plot_sweep.py
PAPER_DIR := paper
FIGS_DIR := $(PAPER_DIR)/figs

# Envs that the paper expects figures for; matches plot_sweep._PAPER_EXPERT.
PAPER_ENVS := Plane3DCircle CheetahRun Reactor FourTank BarkourJoystick

.PHONY: plot plot-real paper paper-clean help

help:
	@echo "Targets:"
	@echo "  make plot       - generate every paper figure (data pending -> placeholder)"
	@echo "  make plot-real  - render only figures whose data is on disk (no placeholders)"
	@echo "  make paper      - plot + latexmk in $(PAPER_DIR)/"
	@echo "  make paper-clean- rm -rf $(PAPER_DIR)/build/"

plot:
	@echo "==> generating paper figures with placeholders for missing data"
	@mkdir -p $(FIGS_DIR)
	@$(PYTHON) $(PLOT_SCRIPT) --paper --with-placeholders
	@echo "==> done. Figures in $(FIGS_DIR)/"

plot-real:
	@echo "==> generating paper figures (real data only; missing figures skipped)"
	@mkdir -p $(FIGS_DIR)
	@$(PYTHON) $(PLOT_SCRIPT) --paper
	@echo "==> done. Figures in $(FIGS_DIR)/"

paper: plot
	@echo "==> building $(PAPER_DIR)/build/paper.pdf"
	@cd $(PAPER_DIR) && latexmk -pdflatex
	@echo "==> $(PAPER_DIR)/build/paper.pdf"

paper-clean:
	@rm -rf $(PAPER_DIR)/build/
	@echo "==> $(PAPER_DIR)/build/ cleaned"
