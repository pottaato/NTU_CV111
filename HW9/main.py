#CV HW9: implement following edge detectors with their thresholds - 蕭恩慈 / B07902095

import numpy as np
from PIL import Image
import math

ori = Image.open("lena.bmp")
w = ori.size[0]
h = ori.size[1]


# def BoxFilter(ori, boxW, boxH):
#     ctrKernel = (boxW // 2, boxH // 2)
#     boxImg = ori.copy()

#     for c in range(w):
#         for r in range(h):
#             boxPx = []
#             for x in range(boxW):
#                 for y in range(boxH):
#                     targetX = c + (x - ctrKernel[0])
#                     targetY = r + (y - ctrKernel[1])
#                     if((0 <= targetX < w) and (0 <= targetY < h)):
#                         oriPx = ori.getpixel((targetX, targetY))
#                         boxPx.append(oriPx)
#             boxImg.putpixel((c, r), int(sum(boxPx) / len(boxPx)))
#     return boxImg

def Roberts(ori, th):
    ber = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = c
            y0 = r
            x1 = min(c + 1, w - 1)
            y1 = min(r + 1, h - 1)

            r1 = -ori.getpixel((x0, y0)) + ori.getpixel((x1, y1))
            r2 = -ori.getpixel((x1, y0)) + ori.getpixel((x0, y1))

            mag = int(math.sqrt(r1 ** 2 + r2 ** 2))

            if(mag >= th):
                ber.putpixel((c, r), 0)
            else:
                ber.putpixel((c, r), 1)
    return ber

def Prewitt(ori, th):
    prw = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 1, 0)
            y0 = max(r - 1, 0)
            x1 = c
            y1 = r
            x2 = min(c + 1, w - 1)
            y2 = min(r + 1, h - 1)
            
            p1 = -ori.getpixel((x0, y0)) - ori.getpixel((x1, y0)) - ori.getpixel((x2, y0)) + ori.getpixel((x0, y2)) + ori.getpixel((x1, y2)) + ori.getpixel((x2, y2))

            p2 = -ori.getpixel((x0, y0)) - ori.getpixel((x0, y1)) - ori.getpixel((x0, y2)) + ori.getpixel((x2, y0)) + ori.getpixel((x2, y1)) + ori.getpixel((x2, y2))
            
            mag = int(math.sqrt(p1 ** 2 + p2 ** 2))

            if(mag >= th):
                prw.putpixel((c, r), 0)
            else:
                prw.putpixel((c, r), 1)
    return prw

def Sobel(ori, th):
    sob = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 1, 0)
            y0 = max(r - 1, 0)
            x1 = c
            y1 = r
            x2 = min(c + 1, w - 1)
            y2 = min(r + 1, h - 1)

            p1 = -ori.getpixel((x0, y0)) - 2 * ori.getpixel((x1, y0)) - ori.getpixel((x2, y0)) \
                + ori.getpixel((x0, y2)) + 2 * ori.getpixel((x1, y2)) + ori.getpixel((x2, y2))
            p2 = -ori.getpixel((x0, y0)) - 2 * ori.getpixel((x0, y1)) - ori.getpixel((x0, y2)) \
                + ori.getpixel((x2, y0)) + 2 * ori.getpixel((x2, y1)) + ori.getpixel((x2, y2))

            mag = int(math.sqrt(p1 ** 2 + p2 ** 2))

            if(mag >= th):
                sob.putpixel((c, r), 0)
            else:
                sob.putpixel((c, r), 1)
    return sob

def FreiChen(ori, th):
    fre = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 1, 0)
            y0 = max(r - 1, 0)
            x1 = c
            y1 = r
            x2 = min(c + 1, w - 1)
            y2 = min(r + 1, h - 1)

            p1 = -ori.getpixel((x0, y0)) - math.sqrt(2) * ori.getpixel((x1, y0)) - ori.getpixel((x2, y0)) \
                + ori.getpixel((x0, y2)) + math.sqrt(2) * ori.getpixel((x1, y2)) + ori.getpixel((x2, y2))
            p2 = -ori.getpixel((x0, y0)) - math.sqrt(2) * ori.getpixel((x0, y1)) - ori.getpixel((x0, y2)) \
                + ori.getpixel((x2, y0)) + math.sqrt(2) * ori.getpixel((x2, y1)) + ori.getpixel((x2, y2))

            mag = int(math.sqrt(p1 ** 2 + p2 ** 2))

            if(mag >= th):
                fre.putpixel((c, r), 0)
            else:
                fre.putpixel((c, r), 1)
    return fre


def Kirsch(ori, th):
    kir = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 1, 0)
            y0 = max(r - 1, 0)
            x1 = c
            y1 = r
            x2 = min(c + 1, w - 1)
            y2 = min(r + 1, h - 1)

            k = np.zeros(8)
            k[0] = -3 * ori.getpixel((x0, y0)) - 3 * ori.getpixel((x1, y0)) + 5 * ori.getpixel((x2, y0)) - 3 * ori.getpixel((x0, y1)) \
                + 5 * ori.getpixel((x2, y1)) - 3 * ori.getpixel((x0, y2)) - 3 * ori.getpixel((x1, y2)) + 5 * ori.getpixel((x2, y2))
            k[1] = -3 * ori.getpixel((x0, y0)) + 5 * ori.getpixel((x1, y0)) + 5 * ori.getpixel((x2, y0)) - 3 * ori.getpixel((x0, y1)) \
                + 5 * ori.getpixel((x2, y1)) - 3 * ori.getpixel((x0, y2)) - 3 * ori.getpixel((x1, y2)) - 3 * ori.getpixel((x2, y2))
            k[2] = 5 * ori.getpixel((x0, y0)) + 5 * ori.getpixel((x1, y0)) + 5 * ori.getpixel((x2, y0)) - 3 * ori.getpixel((x0, y1)) \
                - 3 * ori.getpixel((x2, y1)) - 3 * ori.getpixel((x0, y2)) - 3 * ori.getpixel((x1, y2)) - 3 * ori.getpixel((x2, y2))
            k[3] = 5 * ori.getpixel((x0, y0)) + 5 * ori.getpixel((x1, y0)) - 3 * ori.getpixel((x2, y0)) + 5 * ori.getpixel((x0, y1)) \
                - 3 * ori.getpixel((x2, y1)) - 3 * ori.getpixel((x0, y2)) - 3 * ori.getpixel((x1, y2)) - 3 * ori.getpixel((x2, y2))
            k[4] = 5 * ori.getpixel((x0, y0)) - 3 * ori.getpixel((x1, y0)) - 3 * ori.getpixel((x2, y0)) + 5 * ori.getpixel((x0, y1)) \
                - 3 * ori.getpixel((x2, y1)) + 5 * ori.getpixel((x0, y2)) - 3 * ori.getpixel((x1, y2)) - 3 * ori.getpixel((x2, y2))
            k[5] = -3 * ori.getpixel((x0, y0)) - 3 * ori.getpixel((x1, y0)) - 3 * ori.getpixel((x2, y0)) + 5 * ori.getpixel((x0, y1)) \
                - 3 * ori.getpixel((x2, y1)) + 5 * ori.getpixel((x0, y2)) + 5 * ori.getpixel((x1, y2)) - 3 * ori.getpixel((x2, y2))
            k[6] = -3 * ori.getpixel((x0, y0)) - 3 * ori.getpixel((x1, y0)) - 3 * ori.getpixel((x2, y0)) - 3 * ori.getpixel((x0, y1)) \
                - 3 * ori.getpixel((x2, y1)) + 5 * ori.getpixel((x0, y2)) + 5 * ori.getpixel((x1, y2)) + 5 * ori.getpixel((x2, y2))
            k[7] = -3 * ori.getpixel((x0, y0)) - 3 * ori.getpixel((x1, y0)) - 3 * ori.getpixel((x2, y0)) - 3 * ori.getpixel((x0, y1)) \
                + 5 * ori.getpixel((x2, y1)) - 3 * ori.getpixel((x0, y2)) + 5 * ori.getpixel((x1, y2)) + 5 * ori.getpixel((x2, y2))

            mag = max(k)

            if(mag >= th):
                kir.putpixel((c, r), 0)
            else:
                kir.putpixel((c, r), 1)
    return kir

def Robinson(ori, th):
    rob = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 1, 0)
            y0 = max(r - 1, 0)
            x1 = c
            y1 = r
            x2 = min(c + 1, w - 1)
            y2 = min(r + 1, h - 1)

            k = np.zeros(8)
            k[0] = -1 * ori.getpixel((x0, y0)) - 2 * ori.getpixel((x0, y1)) - 1 * ori.getpixel((x0, y2)) \
                + 1 * ori.getpixel((x2, y0)) + 2 * ori.getpixel((x2, y1)) + 1 * ori.getpixel((x2, y2))
            k[1] = -1 * ori.getpixel((x0, y1)) - 2 * ori.getpixel((x0, y2)) - 1 * ori.getpixel((x1, y2)) \
                + 1 * ori.getpixel((x1, y0)) + 2 * ori.getpixel((x2, y0)) + 1 * ori.getpixel((x2, y1))
            k[2] = -1 * ori.getpixel((x0, y2)) - 2 * ori.getpixel((x1, y2)) - 1 * ori.getpixel((x2, y2)) \
                + 1 * ori.getpixel((x0, y0)) + 2 * ori.getpixel((x1, y0)) + 1 * ori.getpixel((x2, y0))
            k[3] = -1 * ori.getpixel((x1, y2)) - 2 * ori.getpixel((x2, y2)) - 1 * ori.getpixel((x2, y1)) \
                + 1 * ori.getpixel((x0, y1)) + 2 * ori.getpixel((x0, y0)) + 1 * ori.getpixel((x1, y0))
            k[4] = -1 * ori.getpixel((x2, y0)) - 2 * ori.getpixel((x2, y1)) - 1 * ori.getpixel((x2, y2)) \
                + 1 * ori.getpixel((x0, y0)) + 2 * ori.getpixel((x0, y1)) + 1 * ori.getpixel((x0, y2))
            k[5] = -1 * ori.getpixel((x1, y0)) - 2 * ori.getpixel((x2, y0)) - 1 * ori.getpixel((x2, y1)) \
                + 1 * ori.getpixel((x0, y1)) + 2 * ori.getpixel((x0, y2)) + 1 * ori.getpixel((x1, y2))
            k[6] = -1 * ori.getpixel((x0, y0)) - 2 * ori.getpixel((x1, y0)) - 1 * ori.getpixel((x2, y0)) \
                + 1 * ori.getpixel((x0, y2)) + 2 * ori.getpixel((x1, y2)) + 1 * ori.getpixel((x2, y2))
            k[7] = -1 * ori.getpixel((x0, y1)) - 2 * ori.getpixel((x0, y0)) - 1 * ori.getpixel((x1, y0)) \
                + 1 * ori.getpixel((x1, y2)) + 2 * ori.getpixel((x2, y2)) + 1 * ori.getpixel((x2, y1))

            mag = max(k)
            if(mag >= th):
                rob.putpixel((c, r), 0)
            else:
                rob.putpixel((c, r), 1)
    return rob

def NevatiaBabu(ori, th):
    nev = Image.new("1", ori.size)
    for c in range(w):
        for r in range(h):
            x0 = max(c - 2, 0)
            y0 = max(r - 2, 0)
            x1 = max(c - 1, 0)
            y1 = max(r - 1, 0)
            x2 = c
            y2 = r
            x3 = min(c + 1, w - 1)
            y3 = min(r + 1, h - 1)
            x4 = min(c + 2, w - 1)
            y4 = min(r + 2, h - 1)

            neighbors = [
                ori.getpixel((x0, y0)), ori.getpixel((x1, y0)), ori.getpixel((x2, y0)), ori.getpixel((x3, y0)), ori.getpixel((x4, y0)), 
                ori.getpixel((x0, y1)), ori.getpixel((x1, y1)), ori.getpixel((x2, y1)), ori.getpixel((x3, y1)), ori.getpixel((x4, y1)), 
                ori.getpixel((x0, y2)), ori.getpixel((x1, y2)), ori.getpixel((x2, y2)), ori.getpixel((x3, y2)), ori.getpixel((x4, y2)), 
                ori.getpixel((x0, y3)), ori.getpixel((x1, y3)), ori.getpixel((x2, y3)), ori.getpixel((x3, y3)), ori.getpixel((x4, y3)),
                ori.getpixel((x0, y4)), ori.getpixel((x1, y4)), ori.getpixel((x2, y4)), ori.getpixel((x3, y4)), ori.getpixel((x4, y4))]

            k = np.zeros(6)
            k[0] = (100) * neighbors[0] + (100) * neighbors[1] + (100) * neighbors[2] + (100) * neighbors[3] + (100) * neighbors[4] + \
                    (100) * neighbors[5] + (100) * neighbors[6] + (100) * neighbors[7] + (100) * neighbors[8] + (100) * neighbors[9] + \
                    (0) * neighbors[10] + (0) * neighbors[11] + (0) * neighbors[12] + (0) * neighbors[13] + (0) * neighbors[14] + \
                    (-100) * neighbors[15] + (-100) * neighbors[16] + (-100) * neighbors[17] + (-100) * neighbors[18] + (-100) * neighbors[19] + \
                    (-100) * neighbors[20] + (-100) * neighbors[21] + (-100) * neighbors[22] + (-100) * neighbors[23] + (-100) * neighbors[24]
            k[1] = (100) * neighbors[0] + (100) * neighbors[1] + (100) * neighbors[2] + (100) * neighbors[3] + (100) * neighbors[4] + \
                    (100) * neighbors[5] + (100) * neighbors[6] + (100) * neighbors[7] + (78) * neighbors[8] + (-32) * neighbors[9] + \
                    (100) * neighbors[10] + (92) * neighbors[11] + (0) * neighbors[12] + (-92) * neighbors[13] + (-100) * neighbors[14] + \
                    (32) * neighbors[15] + (-78) * neighbors[16] + (-100) * neighbors[17] + (-100) * neighbors[18] + (-100) * neighbors[19] + \
                    (-100) * neighbors[20] + (-100) * neighbors[21] + (-100) * neighbors[22] + (-100) * neighbors[23] + (-100) * neighbors[24]
            k[2] = (100) * neighbors[0] + (100) * neighbors[1] + (100) * neighbors[2] + (32) * neighbors[3] + (-100) * neighbors[4] + \
                    (100) * neighbors[5] + (100) * neighbors[6] + (92) * neighbors[7] + (-78) * neighbors[8] + (-100) * neighbors[9] + \
                    (100) * neighbors[10] + (100) * neighbors[11] + (0) * neighbors[12] + (-100) * neighbors[13] + (-100) * neighbors[14] + \
                    (100) * neighbors[15] + (78) * neighbors[16] + (-92) * neighbors[17] + (-100) * neighbors[18] + (-100) * neighbors[19] + \
                    (100) * neighbors[20] + (-32) * neighbors[21] + (-100) * neighbors[22] + (-100) * neighbors[23] + (-100) * neighbors[24]
            k[3] = (-100) * neighbors[0] + (-100) * neighbors[1] + (0) * neighbors[2] + (100) * neighbors[3] + (100) * neighbors[4] + \
                    (-100) * neighbors[5] + (-100) * neighbors[6] + (0) * neighbors[7] + (100) * neighbors[8] + (100) * neighbors[9] + \
                    (-100) * neighbors[10] + (-100) * neighbors[11] + (0) * neighbors[12] + (100) * neighbors[13] + (100) * neighbors[14] + \
                    (-100) * neighbors[15] + (-100) * neighbors[16] + (0) * neighbors[17] + (100) * neighbors[18] + (100) * neighbors[19] + \
                    (-100) * neighbors[20] + (-100) * neighbors[21] + (0) * neighbors[22] + (100) * neighbors[23] + (100) * neighbors[24]
            k[4] = (-100) * neighbors[0] + (32) * neighbors[1] + (100) * neighbors[2] + (100) * neighbors[3] + (100) * neighbors[4] + \
                    (-100) * neighbors[5] + (-78) * neighbors[6] + (92) * neighbors[7] + (100) * neighbors[8] + (100) * neighbors[9] + \
                    (-100) * neighbors[10] + (-100) * neighbors[11] + (0) * neighbors[12] + (100) * neighbors[13] + (100) * neighbors[14] + \
                    (-100) * neighbors[15] + (-100) * neighbors[16] + (-92) * neighbors[17] + (78) * neighbors[18] + (100) * neighbors[19] + \
                    (-100) * neighbors[20] + (-100) * neighbors[21] + (-100) * neighbors[22] + (-32) * neighbors[23] + (100) * neighbors[24]
            k[5] = (100) * neighbors[0] + (100) * neighbors[1] + (100) * neighbors[2] + (100) * neighbors[3] + (100) * neighbors[4] + \
                    (-32) * neighbors[5] + (78) * neighbors[6] + (100) * neighbors[7] + (100) * neighbors[8] + (100) * neighbors[9] + \
                    (-100) * neighbors[10] + (-92) * neighbors[11] + (0) * neighbors[12] + (92) * neighbors[13] + (100) * neighbors[14] + \
                    (-100) * neighbors[15] + (-100) * neighbors[16] + (-100) * neighbors[17] + (-78) * neighbors[18] + (32) * neighbors[19] + \
                    (-100) * neighbors[20] + (-100) * neighbors[21] + (-100) * neighbors[22] + (-100) * neighbors[23] + (-100) * neighbors[24]

            mag = max(k)
            if(mag >= th):
                nev.putpixel((c, r), 0)
            else:
                nev.putpixel((c, r), 1)
    return nev

def main():
    ber = Roberts(ori, 12) #checked
    ber.save("roberts-lena.bmp")

    prw = Prewitt(ori, 24) #checked
    prw.save("prewitt-lena.bmp")

    sb = Sobel(ori, 38) #checked
    sb.save("sobel-lena.bmp")

    fr = FreiChen(ori, 30) #checked
    fr.save("frei-chen-lena.bmp")

    kr = Kirsch(ori, 135) #checked
    kr.save("kirsch-lena.bmp")

    rb = Robinson(ori, 43) #checked
    rb.save("robinson-lena.bmp")

    nv = NevatiaBabu(ori, 12500) #checked
    nv.save("nevatia-babu-lena.bmp")

    #for report
    # ber.save("roberts-lena.jpg")
    # prw.save("prewitt-lena.jpg")
    # sb.save("sobel-lena.jpg")
    # fr.save("frei-chen-lena.jpg")
    # kr.save("kirsch-lena.jpg")
    # rb.save("robinson-lena.jpg")
    # nv.save("nevatia-babu-lena.jpg")

if __name__ == "__main__":
    main()