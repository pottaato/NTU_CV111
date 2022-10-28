from PIL import Image

ori = Image.open("lena.bmp")

w, h = ori.size

usd = Image.new("L", ori.size)
rsl = Image.new("L", ori.size)
dia = Image.new("L", ori.size)

for c in range(w):
    for r in range(h):
        #upside-down
        x = ori.getpixel((c, h - 1 - r))
        usd.putpixel((c, r), x)

        #right-side-left
        x = ori.getpixel((w - 1 - c, r))
        rsl.putpixel((c, r), x)

        #diagonal flip
        x = ori.getpixel((r, c))
        dia.putpixel((c, r), x)

usd.save("upside-down-lena.jpg")
rsl.save("right-side-left-lena.jpg")
dia.save("flipped-lena.jpg")
        