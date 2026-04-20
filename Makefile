PY ?= python3
TUNE_LIMIT ?= 200

SUMMARY := classical/outputs/tuning/tune_k_summary.json
IQ_RESULTS := classical/outputs/eval_iq/results.csv
OCR_RESULTS := classical/outputs/eval_ocr/results.csv

.PHONY: all tune eval-iq eval-ocr eval clean help

all: tune eval

tune: $(SUMMARY)

$(SUMMARY):
	$(PY) classical/tune_k.py --limit $(TUNE_LIMIT)

eval-iq: $(IQ_RESULTS)

$(IQ_RESULTS): $(SUMMARY)
	$(PY) classical/eval_iq.py

eval-ocr: $(OCR_RESULTS)

$(OCR_RESULTS): $(SUMMARY)
	$(PY) classical/eval_ocr.py

eval: eval-iq eval-ocr

clean:
	rm -rf classical/outputs/tuning classical/outputs/eval_iq classical/outputs/eval_ocr

help:
	@echo "Targets:"
	@echo "  make all        tune k, then run both evaluations"
	@echo "  make tune       run tune_k.py (writes $(SUMMARY))"
	@echo "  make eval-iq    run eval_iq.py (requires tune)"
	@echo "  make eval-ocr   run eval_ocr.py (requires tune)"
	@echo "  make eval       run both evaluations (requires tune)"
	@echo "  make clean      remove tuning/ and eval_*/ outputs"
	@echo ""
	@echo "Override variables:"
	@echo "  TUNE_LIMIT=50 make tune    sweep k on 50 images instead of 200"
	@echo "  PY=python make all         use a different python interpreter"
