from pathlib import Path

from PIL import Image
import numpy as np

HELPERS_DIR = Path(__file__).resolve().parent
CLASSICAL_DIR = HELPERS_DIR.parent
REPO_ROOT = CLASSICAL_DIR.parent

img_id = "0000000"

blur_path = REPO_ROOT / "data" / "BMVC_image_data" / "blur" / f"{img_id}_blur.png"
restored_path = HELPERS_DIR / "wiener_test" / f"{img_id}_blur.png"

a = np.array(Image.open(blur_path).convert("L"))
b = np.array(Image.open(restored_path).convert("L"))

print("blur shape:", a.shape)
print("restored shape:", b.shape)
print("blur min/max:", a.min(), a.max())
print("restored min/max:", b.min(), b.max())
print("arrays equal?:", np.array_equal(a, b))
