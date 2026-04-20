import sys
from pathlib import Path

from PIL import Image, ImageDraw

HELPERS_DIR = Path(__file__).resolve().parent
CLASSICAL_DIR = HELPERS_DIR.parent
REPO_ROOT = CLASSICAL_DIR.parent

sys.path.insert(0, str(CLASSICAL_DIR))
from wienerfiltering import open_pic, save_pic, do_wiener

PREVIEWS_DIR = HELPERS_DIR / "previews"
WIENER_DIR = HELPERS_DIR / "wiener_test"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
WIENER_DIR.mkdir(parents=True, exist_ok=True)

img_id = "0009059"

data_dir = REPO_ROOT / "data" / "BMVC_image_data"
blur_path = data_dir / "blur" / f"{img_id}_blur.png"
orig_path = data_dir / "orig" / f"{img_id}_orig.png"
psf_path = data_dir / "psf" / f"{img_id}_psf.png"
restored_path = WIENER_DIR / f"{img_id}_blur.png"

if not restored_path.exists():
    print("Restored image not found, computing Wiener for", img_id)
    blur_arr = open_pic(str(blur_path))
    psf_arr = open_pic(str(psf_path))
    out_arr = do_wiener(blur_arr, psf_arr, 0.01)
    save_pic(out_arr, str(restored_path))

blur = Image.open(blur_path).convert("L")
orig = Image.open(orig_path).convert("L")
restored = Image.open(restored_path).convert("L")

blur = blur.resize((300, 300))
orig = orig.resize((300, 300))
restored = restored.resize((300, 300))

top = 40
img = Image.new("L", (900, 340), color=255)

img.paste(blur, (0, top))
img.paste(orig, (300, top))
img.paste(restored, (600, top))

d = ImageDraw.Draw(img)
d.text((120, 10), "Blur", fill=0)
d.text((420, 10), "Original", fill=0)
d.text((690, 10), "Wiener Restored", fill=0)

out_path = PREVIEWS_DIR / f"{img_id}_compare.png"
img.save(out_path)
print("saved", out_path)
