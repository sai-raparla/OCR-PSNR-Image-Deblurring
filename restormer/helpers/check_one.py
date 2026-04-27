import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

HELPERS_DIR = Path(__file__).resolve().parent
RESTORMER_DIR = HELPERS_DIR.parent
REPO_ROOT = RESTORMER_DIR.parent
CLASSICAL_DIR = REPO_ROOT / "classical"

sys.path.insert(0, str(RESTORMER_DIR))
sys.path.insert(0, str(CLASSICAL_DIR))

from wienerfiltering import open_pic, save_pic  # noqa: E402
from inference import load_checkpoint, pick_device, restore_np  # noqa: E402

PREVIEWS_DIR = HELPERS_DIR / "previews"
RESTORMER_CACHE = HELPERS_DIR / "restormer_test"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
RESTORMER_CACHE.mkdir(parents=True, exist_ok=True)

img_id = sys.argv[1] if len(sys.argv) > 1 else "0000000"
ckpt_path = (
    Path(sys.argv[2]) if len(sys.argv) > 2
    else RESTORMER_DIR / "outputs" / "checkpoints" / "best.pt"
)

data_dir = REPO_ROOT / "data" / "BMVC_image_data"
blur_path = data_dir / "blur" / f"{img_id}_blur.png"
restored_path = RESTORMER_CACHE / f"{img_id}_blur.png"

if not blur_path.exists():
    raise FileNotFoundError(f"Blurred image not found: {blur_path}")

if not restored_path.exists():
    print(f"Restored image not found, running Restormer for {img_id}...")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = pick_device("auto")
    print(f"Loading Restormer from {ckpt_path} on {device}")
    model = load_checkpoint(str(ckpt_path), device)
    blur_arr = open_pic(str(blur_path))
    out_arr = restore_np(model, blur_arr, device)
    save_pic(out_arr, str(restored_path))

blur = np.array(Image.open(blur_path).convert("L"))
restored = np.array(Image.open(restored_path).convert("L"))

if blur.shape != restored.shape:
    raise ValueError(
        f"Shape mismatch: blur {blur.shape} vs restored {restored.shape}"
    )

print(f"image id:         {img_id}")
print(f"shape:            {blur.shape}")
print(f"blur min/max:     {blur.min()} / {blur.max()}")
print(f"restored min/max: {restored.min()} / {restored.max()}")
print(f"identical?:       {np.array_equal(blur, restored)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(blur, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Before (blurry)")
axes[0].axis("off")

axes[1].imshow(restored, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("After (Restormer restored)")
axes[1].axis("off")

fig.suptitle(f"Image {img_id}: before vs after deblurring (Restormer)")
fig.tight_layout()

out_path = PREVIEWS_DIR / f"{img_id}_diff.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved comparison: {out_path}")

plt.show()
