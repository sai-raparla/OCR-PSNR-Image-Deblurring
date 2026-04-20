from PIL import Image
import numpy as np

a = np.array(Image.open("data/raw/BMVC_image_data/blur/0000000_blur.png").convert("L"))
b = np.array(Image.open("classical/outputs/wiener_test/0000000_blur.png").convert("L"))

print("blur shape:", a.shape)
print("restored shape:", b.shape)
print("blur min/max:", a.min(), a.max())
print("restored min/max:", b.min(), b.max())
print("arrays equal?:", np.array_equal(a, b))
