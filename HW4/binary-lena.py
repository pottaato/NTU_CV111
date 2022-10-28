from PIL import Image

threshold = 128

ori = Image.open("lena.bmp")

w, h = ori.size

bin = Image.new("1", ori.size)

for c in range(w):
    for r in range(h):
        # Get pixel from original image.
        value = ori.getpixel((c, r))
        if (value >= threshold):
            value = 1
        else:
            value = 0
        # Put pixel to binary image.
        bin.putpixel((c, r), value)

# Save image.
bin.save('binary-lena.bmp')