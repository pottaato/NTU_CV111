#CV HW4: write programs which do binary morphology on a binary image - 蕭恩慈 / B07902095
#the binary image of the original image (binary-lena.bmp) will be used as "original image" of this program
import numpy as np 
from PIL import Image

def dilation(ori, kernel):
    ctr_kernel = tuple([x // 2 for x in kernel.shape])
    dil = Image.new("1", ori.size)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            px_ori = ori.getpixel((r, c))
            if(px_ori != 0):
                for x in range(kernel.shape[0]):
                    for y in range(kernel.shape[1]):
                        if(kernel[x, y] == 1):
                            target_x = r + (x - ctr_kernel[0])
                            target_y = c + (y - ctr_kernel[1])
                            if((0 <= target_x < ori.size[0]) and (0 <= target_y < ori.size[1])):
                                dil.putpixel((target_x, target_y), 1)                  
    return dil

def erosion(ori, kernel):
    ctr_kernel = tuple([x // 2 for x in kernel.shape])
    eros = Image.new("1", ori.size)
    #print(ori)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            match = True
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if(kernel[x, y] == 1):
                        target_x = r + (x - ctr_kernel[0])
                        target_y = c + (y - ctr_kernel[1])
                        if((0 <= target_x < ori.size[0]) and (0 <= target_y < ori.size[1])):
                            if(ori.getpixel((target_x, target_y)) == 0):
                                match = False
                                break
                        else:
                            match = False
                            break
            if (match):
                eros.putpixel((r, c), 1)
                           
    return eros

def opening(ori, kernel):
    return dilation(erosion(ori, kernel), kernel)

def closing(ori, kernel):
    return erosion(dilation(ori, kernel), kernel)

def complement(ori):
    comp_ori = Image.new("1", ori.size)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            if(ori.getpixel((r, c)) == 0):
                comp_ori.putpixel((r, c), 1)
            else:
                comp_ori.putpixel((r, c), 0)
                
    return comp_ori
    
def intersection(ori, comp):
    intersect = Image.new("1", ori.size)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            px_ori = ori.getpixel((r, c))
            px_comp = comp.getpixel((r, c))
            if(px_ori != 0 and px_comp != 0):
                intersect.putpixel((r, c), 1)
            else:
                intersect.putpixel((r, c), 0)
                
    return intersect

def erosionCenter(ori, kernel, ctr_kernel):
    eros = Image.new("1", ori.size)
    for r in range(ori.size[0]):
        for c in range(ori.size[1]):
            exist = True
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if(kernel[x, y] == 1):
                        target_x = r + (x - ctr_kernel[0])
                        target_y = c + (y - ctr_kernel[1])
                        # print(ctr_kernel)
                        # print(ori.size)
                        # print(target_x)
                        # print(target_y)
                        if(target_x < ori.size[0]):
                            if (target_y < ori.size[1]):
                                if(ori.getpixel((target_x, target_y)) == 0):
                                    exist = False
                                    break
                        else:
                            exist = False
                            break
            if (exist):
                eros.putpixel((r, c), 1)
                           
    return eros
   
def hitAndMiss(ori, J_kernel, J_center, K_kernel, K_center):
    return intersection(erosionCenter(ori, J_kernel, J_center), 
                        erosionCenter(complement(ori), K_kernel, K_center))

def main():
    kernel = np.array([
                [0, 1, 1, 1, 0], 
                [1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1], 
                [0, 1, 1, 1, 0]])
    
    ori = Image.open("binary-lena.bmp")
    
    dil = dilation(ori, kernel)
    eros = erosion(ori, kernel)
    op = opening(ori, kernel)
    cl = opening(ori, kernel)
    
    J_kernel = np.array([[1, 1], [0, 1]])
    K_kernel = np.array([[1, 1], [0, 1]])
    J_center = (1, 0)
    K_center = (0, 1)
    hitMiss = hitAndMiss(ori, J_kernel, J_center, K_kernel, K_center) 
    
    #print("==========")

    dil.save("dilated-lena.bmp")
    eros.save("erosion-lena.bmp")
    op.save("opening-lena.bmp")
    cl.save("closing-lena.bmp")
    hitMiss.save("hit-and-miss-lena.bmp")

if __name__ == "__main__":
    main()