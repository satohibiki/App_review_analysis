.SUFFIXES: .tex .dvi .pdf .xbb .png .jpg
# compile option
TEX     = platex --kanji=utf8
BIBTEX  = pbibtex --kanji=utf8
DVIPDFM = dvipdfmx
DVIPDFMFLAGS = -p a4
EBB       = extractbb
EBB_SUFFIX= .xbb
DVICONCAT = dviconcat

# font
MAP = "ipa.map"

# Dirs
IMAGE_DIR=images
KEIO_MARK=$(IMAGE_DIR)/keio.pdf
KEIO_BB=$(addsuffix $(EBB_SUFFIX), $(basename $(KEIO_MARK)))

# Targets
THESIS=thesis

# using document 
TEX_FILES   = $(wildcard */*.tex) 
STY_FILES   = $(wildcard *.sty)
IMAGE_FILES = $(wildcard $(IMAGE_DIR)/*.png) $(wildcard $(IMAGE_DIR)/*.jpg) $(wildcard $(IMAGE_DIR)/*.pdf)  
BB_FILES    = $(addsuffix $(EBB_SUFFIX), $(basename $(IMAGE_FILES)))

# image suffix rule
.pdf.xbb:
	$(EBB) $< 
.jpg.xbb:
	$(EBB) $< 
.png.xbb:
	$(EBB) $< 

# latex compile suffix rule
.tex.dvi:
	$(TEX) $<
	$(TEX) $<
.dvi.pdf:
	$(DVIPDFM) $(DVIPDFMFLAGS) $<

$(THESIS).dvi : $(THESIS).tex abstractContents.tex $(TEX_FILES) $(STY_FILES) $(IMAGE_FILES) $(BB_FILES)
	$(TEX) $(THESIS).tex
	$(BIBTEX) $(THESIS)
	$(TEX) $(THESIS).tex
	$(TEX) $(THESIS).tex

.PHONY: thesis thsis-embed clean distclean
.DEFAULT_GOAL := all

thesis: $(THESIS).pdf
thesis-embed: $(THESIS).dvi $(MAP)
	$(DVIPDFM) $(DVIPDFMFLAGS) -f $(MAP) $<
all: thesis
clean:
	@rm -rf *.toc *.log *.dvi *.fls *.aux *.maf *.mtc *.bbl *.blg *.out *.lof *.lot *.fdb_latexmk *.synctex.gz *.ps
distclean: clean
	@rm -rf $(BB_FILES) $(THESIS).pdf
