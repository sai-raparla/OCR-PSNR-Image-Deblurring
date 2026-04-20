from richardsonlucy import run_all

run_all(
    "data/raw/BMVC_image_data/blur",
    "data/raw/BMVC_image_data/psf",
    "classical/outputs/rl_t3",
    50,
    3
)
