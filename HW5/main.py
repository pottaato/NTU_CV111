#CV HW5: write programs which do gray-scale morphology on a gray-scale image - 蕭恩慈 / B07902095
import numpy as np 
from PIL import Image

def dilation(ori, kernel, ctr_kernel):
    dil = Image.new("L", ori.size)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            px_local_max = 0
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if(kernel[x, y] == 1):
                        target_x = r + (x - ctr_kernel[0])
                        target_y = c + (y - ctr_kernel[1])
                        if((0 <= target_x < ori.size[0]) and (0 <= target_y < ori.size[1])):
                            px_ori = ori.getpixel((target_x, target_y))
                            px_local_max = max(px_ori, px_local_max) 

            dil.putpixel((r, c), px_local_max)    

    return dil

def erosion(ori, kernel, ctr_kernel):
    eros = Image.new("L", ori.size)
    #print(ori)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            px_local_min = 255
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if(kernel[x, y] == 1):
                        target_x = r + (x - ctr_kernel[0])
                        target_y = c + (y - ctr_kernel[1])
                        if((0 <= target_x < ori.size[0]) and (0 <= target_y < ori.size[1])):
                            px_ori = ori.getpixel((target_x, target_y))
                            px_local_min = min(px_ori, px_local_min)

            eros.putpixel((r, c), px_local_min)
                           
    return eros

def opening(ori, kernel, ctr_kernel):
    return dilation(erosion(ori, kernel, ctr_kernel), kernel, ctr_kernel)

def closing(ori, kernel, ctr_kernel):
     return erosion(dilation(ori, kernel, ctr_kernel), kernel, ctr_kernel)

def main():
    ctr_kernel = (2, 2)
    kernel = np.array([
                [0, 1, 1, 1, 0], 
                [1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1], 
                [0, 1, 1, 1, 0]])
    
    ori = Image.open("lena.bmp")
    
    dil = dilation(ori, kernel, ctr_kernel)
    eros = erosion(ori, kernel, ctr_kernel)
    op = opening(ori, kernel, ctr_kernel)
    cl = closing(ori, kernel, ctr_kernel)

    dil.save("gray-dilated-lena.bmp")
    eros.save("gray-erosion-lena.bmp")
    op.save("gray-opening-lena.bmp")
    cl.save("gray-closing-lena.bmp")

    #for report
    # dil.save("gray-dilated-lena.jpg")
    # eros.save("gray-erosion-lena.jpg")
    # op.save("gray-opening-lena.jpg")
    # cl.save("gray-closing-lena.jpg")

if __name__ == "__main__":
    main()