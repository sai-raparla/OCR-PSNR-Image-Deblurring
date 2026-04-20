from wienerfiltering import run_all

run_all(
    "data/raw/BMVC_image_data/blur",
    "data/raw/BMVC_image_data/psf",
    "classical/outputs/wiener_test",
    5,
    0.01
)
