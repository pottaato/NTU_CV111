#CV HW2 part (a): generate binary with threshold at 128 - 蕭恩慈 / B07902095
from PIL import Image

ori = Image.open("lena.bmp")
w, h = ori.size

threshold = 128

bin = Image.new("1", ori.size)

for c in range(w):
    for r in range(h):
        x = ori.getpixel((c, r))
        x = 1 if x >= threshold else 0
        bin.putpixel((c, r), x)

bin.save("binary2-lena.jpg")
