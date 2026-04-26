# latexmk config for the EGE paper.
#
# - forces classic pdflatex (avoids luaotfload dependency on systems without
#   texlive-luatex installed)
# - only builds paper.tex, skipping standalone fragments like
#   appendix_hyperparam_search.tex which are meant to be \input'd, not compiled
# - runs biber/bibtex and reruns pdflatex automatically when needed

$pdf_mode = 1;                         # produce PDF
$pdflatex = 'pdflatex %O %S';          # classic engine
$bibtex_use = 2;                       # run bibtex when cited entries exist
@default_files = ('paper.tex');        # explicit target (ignore appendix_*.tex)

# Output directory for build artifacts — keeps the source tree clean.
$aux_dir = 'build';
$out_dir = 'build';
