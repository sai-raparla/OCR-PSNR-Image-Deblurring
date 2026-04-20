from PIL import Image, ImageDraw
import os

os.makedirs("classical/outputs/previews", exist_ok=True)

blur_path = "data/raw/BMVC_OCR_test_data/n_00/0000000_blur.png"
orig_path = "data/raw/BMVC_OCR_test_data/orig/0000000_orig.png"
restored_path = "classical/outputs/wiener_ocr_n00/0000000_blur.png"

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

img.save("classical/outputs/previews/ocr_test_0000000_compare.png")
print("saved classical/outputs/previews/ocr_test_0000000_compare.png")
