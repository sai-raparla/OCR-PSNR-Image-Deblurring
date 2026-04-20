from wienerfiltering import run_all

run_all(
    "data/raw/BMVC_OCR_test_data/n_00",
    "data/raw/BMVC_OCR_test_data/psf",
    "classical/outputs/wiener_ocr_n00",
    10,
    0.01
)
