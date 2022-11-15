#CV HW6: write a program which counts the Yokoi connectivity number on a downsampled - 蕭恩慈 / B07902095
from PIL import Image
import numpy as np

#get binary img
def binary(ori):
    w, h = ori.size
    bin = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            value = ori.getpixel((c, r))
            if (value >= 128): #threshold= 128
                value = 1
            else:
                value = 0
            bin.putpixel((c, r), value)

    return bin

#downsampling
def downsampling(ori):
    w = int(ori.size[0] / 8)
    h = int(ori.size[1] / 8)
    downsampled = Image.new("1", (w, h))
    for c in range(downsampled.size[0]):
        for r in range(downsampled.size[1]):
            ori_px = ori.getpixel((c * 8, r * 8))
            downsampled.putpixel((c, r), ori_px)

    return downsampled

#get neighborhood px
def neighbor(ori, position): 
    neighborPixel = np.zeros(9)
    x, y = position
    for dx in range(3):
        for dy in range(3):
            target_x = x + (dx - 1)
            target_y = y + (dy - 1)
            if ((0 <= target_x < ori.size[0]) and (0 <= target_y < ori.size[1])):
                neighborPixel[3 * dy + dx] = ori.getpixel((target_x, target_y))
            else:
                neighborPixel[3 * dy + dx] = 0

    neighborPixel = [ 
        neighborPixel[4], neighborPixel[5], neighborPixel[1],
        neighborPixel[3], neighborPixel[7], neighborPixel[8],
        neighborPixel[2], neighborPixel[0], neighborPixel[6]
    ]

    return neighborPixel

def h(b, c, d, e):
    if ((b == c) and (b != d or b != e)):
        return "q"
    if ((b == c) and (b == d and b == e)):
        return "r"
    if (b != c):
        return "s"

def f(a1, a2, a3, a4):
    if ([a1, a2, a3, a4].count("r") == 4): #if a1 = a2 = a3 = a4 = r
        return 5
    else:
        return [a1, a2, a3, a4].count("q") #else count q

def yokoi(ori):
    yokoi = np.full(downsampled.size, " ")
    for c in range(ori.size[0]):
        for r in range(ori.size[1]):
            if (ori.getpixel((c, r)) != 0):
                neighborPixel = neighbor(ori, (c, r))
                yokoi[c, r] = f(
                    h(neighborPixel[0], neighborPixel[1], neighborPixel[6], neighborPixel[2]),
                    h(neighborPixel[0], neighborPixel[2], neighborPixel[7], neighborPixel[3]),
                    h(neighborPixel[0], neighborPixel[3], neighborPixel[8], neighborPixel[4]),
                    h(neighborPixel[0], neighborPixel[4], neighborPixel[5], neighborPixel[1])
                )
            else:
                yokoi[c, r] = " "
    
    return yokoi


if __name__ == "__main__":
    ori = Image.open("lena.bmp")

    bin = binary(ori)
    bin.save("binary-lena.bmp")

    downsampled = downsampling(bin)
    downsampled.save("downsampled-lena.bmp")

    yokoi = yokoi(downsampled)
    np.savetxt("yokoi-lena.txt", yokoi.T, fmt = "%s", delimiter = "")
