import argparse
import os
import re
import numpy as np
from PIL import Image


def extract_image_id(filename):
    match = re.match(r"(\d+)", filename)
    if match:
        return match.group(1)
    return None


def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_grayscale_image(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def psf_to_otf(psf, shape):
    h, w = shape
    ph, pw = psf.shape

    padded = np.zeros((h, w), dtype=np.float32)
    padded[:ph, :pw] = psf

    padded = np.roll(padded, -ph // 2, axis=0)
    padded = np.roll(padded, -pw // 2, axis=1)

    otf = np.fft.fft2(padded)
    return otf


def wiener_deblur(blurred, psf, k=0.01):
    psf_sum = psf.sum()
    if psf_sum != 0:
        psf = psf / psf_sum

    G = np.fft.fft2(blurred)
    H = psf_to_otf(psf, blurred.shape)
    H_conj = np.conj(H)

    F_hat = (H_conj / (np.abs(H) ** 2 + k)) * G
    restored = np.real(np.fft.ifft2(F_hat))

    restored = np.clip(restored, 0.0, 1.0)
    return restored


def build_file_map(folder):
    file_map = {}

    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            image_id = extract_image_id(filename)
            if image_id is not None:
                file_map[image_id] = filename

    return file_map


def process_images(blur_dir, psf_dir, output_dir, limit=None, k=0.01):
    blur_map = build_file_map(blur_dir)
    psf_map = build_file_map(psf_dir)

    common_ids = sorted(set(blur_map.keys()) & set(psf_map.keys()))

    if limit is not None:
        common_ids = common_ids[:limit]

    print(f"Found {len(common_ids)} matching blur/psf pairs")

    if len(common_ids) == 0:
        print("No matching files found.")
        return

    for i, image_id in enumerate(common_ids, start=1):
        blur_name = blur_map[image_id]
        psf_name = psf_map[image_id]

        blur_path = os.path.join(blur_dir, blur_name)
        psf_path = os.path.join(psf_dir, psf_name)
        output_path = os.path.join(output_dir, blur_name)

        try:
            blurred = load_grayscale_image(blur_path)
            psf = load_grayscale_image(psf_path)

            restored = wiener_deblur(blurred, psf, k=k)
            save_grayscale_image(restored, output_path)

            print(f"[{i}/{len(common_ids)}] saved {output_path}")

        except Exception as e:
            print(f"Skipping {blur_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Apply Wiener filtering to blurred images using PSF images")
    parser.add_argument("--blur-dir", required=True, help="Folder with blurred images")
    parser.add_argument("--psf-dir", required=True, help="Folder with PSF images")
    parser.add_argument("--output-dir", required=True, help="Folder to save restored images")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N images")
    parser.add_argument("--k", type=float, default=0.01, help="Wiener filter noise constant")

    args = parser.parse_args()

    process_images(
        blur_dir=args.blur_dir,
        psf_dir=args.psf_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        k=args.k,
    )


if __name__ == "__main__":
    main()
