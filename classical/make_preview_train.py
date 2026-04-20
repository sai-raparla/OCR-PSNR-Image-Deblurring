from PIL import Image, ImageDraw
import os

from wienerfiltering import open_pic, save_pic, do_wiener

os.makedirs("classical/outputs/previews", exist_ok=True)
os.makedirs("classical/outputs/wiener_test", exist_ok=True)

img_id = "0009059"

blur_path = "data/BMVC_image_data/blur/" + img_id + "_blur.png"
orig_path = "data/BMVC_image_data/orig/" + img_id + "_orig.png"
psf_path = "data/BMVC_image_data/psf/" + img_id + "_psf.png"
restored_path = "classical/outputs/wiener_test/" + img_id + "_blur.png"

if not os.path.exists(restored_path):
    print("Restored image not found, computing Wiener for", img_id)
    blur_arr = open_pic(blur_path)
    psf_arr = open_pic(psf_path)
    out_arr = do_wiener(blur_arr, psf_arr, 0.01)
    save_pic(out_arr, restored_path)

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

out_path = "classical/outputs/previews/" + img_id + "_compare.png"
img.save(out_path)
print("saved", out_path)
