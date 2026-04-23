PY ?= python3
TUNE_LIMIT ?= 200

# Classical Wiener outputs
SUMMARY := classical/outputs/tuning/tune_k_summary.json
IQ_RESULTS := classical/outputs/eval_iq/results.csv
OCR_RESULTS := classical/outputs/eval_ocr/results.csv

# Restormer outputs
R_CKPT := restormer/outputs/checkpoints/best.pt
R_IQ_RESULTS := restormer/outputs/eval_iq/results.csv
R_OCR_RESULTS := restormer/outputs/eval_ocr/results.csv
COMPARE_DIR := restormer/outputs/compare

# Restormer training hyper-params (override on the command line)
R_VARIANT ?= small
R_EPOCHS ?= 30
R_BATCH ?= 8
R_PATCH ?= 128
R_LR ?= 3e-4
R_DEVICE ?= auto

.PHONY: all tune eval-iq eval-ocr eval \
        restormer-train restormer-eval-iq restormer-eval-ocr restormer-eval \
        compare full clean help

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

# ------------------------------- Restormer -------------------------------

restormer-train: $(R_CKPT)

$(R_CKPT):
	$(PY) restormer/train.py \
	    --variant $(R_VARIANT) \
	    --epochs $(R_EPOCHS) \
	    --batch-size $(R_BATCH) \
	    --patch-size $(R_PATCH) \
	    --lr $(R_LR) \
	    --device $(R_DEVICE)

restormer-eval-iq: $(R_IQ_RESULTS)

$(R_IQ_RESULTS): $(R_CKPT)
	$(PY) restormer/eval_iq.py --ckpt $(R_CKPT) --device $(R_DEVICE)

restormer-eval-ocr: $(R_OCR_RESULTS)

$(R_OCR_RESULTS): $(R_CKPT)
	$(PY) restormer/eval_ocr.py --ckpt $(R_CKPT) --device $(R_DEVICE)

restormer-eval: restormer-eval-iq restormer-eval-ocr

compare: $(IQ_RESULTS) $(OCR_RESULTS) $(R_IQ_RESULTS) $(R_OCR_RESULTS)
	$(PY) restormer/compare.py --output-dir $(COMPARE_DIR)

full: all restormer-train restormer-eval compare

clean:
	rm -rf classical/outputs/tuning classical/outputs/eval_iq classical/outputs/eval_ocr \
	       restormer/outputs/checkpoints restormer/outputs/eval_iq restormer/outputs/eval_ocr \
	       $(COMPARE_DIR) restormer/outputs/train_log.csv restormer/outputs/train_summary.json

help:
	@echo "Classical Wiener targets:"
	@echo "  make all              tune k, then run both Wiener evaluations"
	@echo "  make tune             run tune_k.py (writes $(SUMMARY))"
	@echo "  make eval-iq          run classical/eval_iq.py"
	@echo "  make eval-ocr         run classical/eval_ocr.py"
	@echo "  make eval             run both Wiener evaluations"
	@echo ""
	@echo "Restormer targets:"
	@echo "  make restormer-train      train Restormer on BMVC_image_data"
	@echo "  make restormer-eval-iq    PSNR eval with Restormer"
	@echo "  make restormer-eval-ocr   CER/WER eval with Restormer"
	@echo "  make restormer-eval       both Restormer evals"
	@echo "  make compare              overlay Wiener vs Restormer curves"
	@echo "  make full                 everything: wiener + restormer + compare"
	@echo "  make clean                remove all generated outputs"
	@echo ""
	@echo "Override variables:"
	@echo "  R_VARIANT=base R_EPOCHS=200 R_BATCH=16 make restormer-train"
	@echo "  R_DEVICE=cuda make restormer-eval"
	@echo "  TUNE_LIMIT=50 make tune"
