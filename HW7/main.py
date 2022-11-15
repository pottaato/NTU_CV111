#CV HW7: write a program which does thinning on a downsampled image - 蕭恩慈 / B07902095
import cv2
import numpy as np
import copy


def findInteriorBorder(ori):
    def h(c, d):
        if c == d:
            return c
        return "b"
    interior_border = np.zeros(ori.shape, np.int)
    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            #background px
            if ori[i][j] > 0:
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    if j == 0:
                        x1, x4 = ori[i][j + 1], ori[i + 1][j]

                    elif j == (ori.shape[1]) - 1:
                        x3, x4 = ori[i][j - 1], ori[i + 1][j]

                    else:
                        x1, x3, x4 = ori[i][j + 1], ori[i][j - 1], ori[i + 1][j]

                elif i == (ori.shape[0]) - 1:
                    if j == 0:
                        x1, x2 = ori[i][j + 1], ori[i - 1][j]

                    elif j == (ori.shape[1]) - 1:
                        x2, x3 = ori[i - 1][j], ori[i][j - 1]

                    else:
                        x1, x2, x3 = ori[i][j + 1], ori[i - 1][j], ori[i][j - 1]

                else:
                    if j == 0:
                        x1, x2, x4 = ori[i][j + 1], ori[i - 1][j], ori[i + 1][j]

                    elif j == (ori.shape[1]) - 1:
                        x2, x3, x4 = ori[i - 1][j], ori[i][j - 1], ori[i + 1][j]

                    else:
                        x1, x2, x3, x4 = ori[i][j + 1], ori[i - 1][j], ori[i][j - 1], ori[i + 1][j]

                x1 = x1 / 255
                x2 /= 255
                x3 /= 255
                x4 /= 255
                a1 = h(1, x1)
                a2 = h(a1, x2)
                a3 = h(a2, x3)
                a4 = h(a3, x4)

                if a4 == "b":
                    interior_border[i][j] = 2 #border px
                
                else:
                    interior_border[i][j] = 1 #interior px
    
    return interior_border

def findPairRelationship(interior_border):
    def h(a, m):
        if a == m:
            return 1
        return 0
    
    pair_relationship = np.zeros(interior_border.shape, np.int)
    for i in range(interior_border.shape[0]):
        for j in range(interior_border.shape[1]):
            if interior_border[i][j] > 0:
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    if j == 0:
                        x1, x4 = interior_border[i][j + 1], interior_border[i + 1][j]

                    elif j == (interior_border.shape[1]) - 1:
                        x3, x4 = interior_border[i][j - 1], interior_border[i + 1][j]

                    else:
                        x1, x3, x4 = interior_border[i][j + 1], interior_border[i][j - 1], interior_border[i + 1][j]

                elif i == (interior_border.shape[0]) - 1:
                    if j == 0:
                        x1, x2 = interior_border[i][j + 1], interior_border[i - 1][j]

                    elif j == (interior_border.shape[1]) - 1:
                        x2, x3 = interior_border[i - 1][j], interior_border[i][j - 1]

                    else:
                        x1, x2, x3 = interior_border[i][j + 1], interior_border[i - 1][j], interior_border[i][j - 1]

                else:
                    if j == 0:
                        x1, x2, x4 = interior_border[i][j + 1], interior_border[i - 1][j], interior_border[i + 1][j]

                    elif j == (interior_border.shape[1]) - 1:
                        x2, x3, x4 = interior_border[i - 1][j], interior_border[i][j - 1], interior_border[i + 1][j]

                    else:
                        x1, x2, x3, x4 = interior_border[i][j + 1], interior_border[i - 1][j], interior_border[i][j - 1], interior_border[i + 1][j]

                if ((h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1)) >= 1) and (interior_border[i][j] == 2):
                    pair_relationship[i][j] = 1

                else:
                    pair_relationship[i][j] = 2

    return pair_relationship

def yokoi(ori):
    def h(b, c, d, e):
        if ((b == c) and (b != d or b != e)):
            return "q"

        if ((b == c) and (b == d and b == e)):
            return "r"
        
        return "s"

    yokoi_map = np.zeros(ori.shape, np.int)
    for i in range(ori.shape[0]):
        for j in range(ori.shape[1]):
            if ori[i][j] > 0:
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = 0, ori[i + 1][j], ori[i + 1][j + 1]

                    elif j == (ori.shape[1]) - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], 0
                        x8, x4, x5 = ori[i + 1][j - 1], ori[i + 1][j], 0

                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = ori[i + 1][j - 1], ori[i + 1][j], ori[i + 1][j + 1]

                elif i == (ori.shape[0]) - 1:
                    if j == 0:
                        x7, x2, x6 = 0, ori[i - 1][j], ori[i - 1][j + 1]
                        x3, x0, x1 = 0, ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = 0, 0, 0

                    elif j == (ori.shape[1]) - 1:
                        x7, x2, x6 = ori[i - 1][j - 1], ori[i - 1][j], 0
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], 0
                        x8, x4, x5 = 0, 0, 0

                    else:
                        x7, x2, x6 = ori[i - 1][j - 1], ori[i - 1][j], ori[i - 1][j + 1]
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = 0, 0, 0

                else:
                    if j == 0:
                        x7, x2, x6 = 0, ori[i - 1][j], ori[i - 1][j + 1]
                        x3, x0, x1 = 0, ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = 0, ori[i + 1][j], ori[i + 1][j + 1]

                    elif j == (ori.shape[1]) - 1:
                        x7, x2, x6 = ori[i - 1][j - 1], ori[i - 1][j], 0
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], 0
                        x8, x4, x5 = ori[i + 1][j - 1], ori[i + 1][j], 0

                    else:
                        x7, x2, x6 = ori[i - 1][j - 1], ori[i - 1][j], ori[i - 1][j + 1]
                        x3, x0, x1 = ori[i][j - 1], ori[i][j], ori[i][j + 1]
                        x8, x4, x5 = ori[i + 1][j - 1], ori[i + 1][j], ori[i + 1][j + 1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)

                if (a1 == "r") and (a2 == "r") and (a3 == "r") and (a4 == "r"):
                    res = 5
                
                else:
                    res = 0
                    for a_i in [a1, a2, a3, a4]:
                        if a_i == "q":
                            res += 1
                
                yokoi_map[i][j] = res
        
    return yokoi_map

def main():

    ori = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]])

    interior_border = findInteriorBorder(ori * 255)
    #print(interior_border)

    interior_border = np.array ([
        [2, 2, 2, 2, 2, 2, 0],
        [2, 1, 1, 1, 1, 2, 0],
        [2, 2, 2, 2, 1, 2, 2],
        [2, 0, 0, 2, 2, 1, 2],
        [2, 0, 0, 0, 2, 2, 2],
        [2, 0, 0, 0, 0, 0, 2]])

    pair_relationship = findPairRelationship(interior_border)
    #print(pair_relationship)

    ori = cv2.imread("lena.bmp", 0)

    #binarize 
    bin = np.zeros(ori.shape, np.int)
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            if ori[c][r] >= 128: #threshold = 128
                bin[c][r] = 255

    #downsampling 8x8
    down = np.zeros((64, 64), np.int)
    for c in range(down.shape[0]):
        for r in range(down.shape[1]):
            down[c][r] = bin[c * 8][r * 8]

    #thinning
    thin = down
    while True:
        cp_thin = copy.deepcopy(thin)
        interior_border = findInteriorBorder(thin)
        pair_relationship = findPairRelationship(interior_border)

        yokoi_map = yokoi(thin)
        rm_map = (yokoi_map == 1) * 1

        for c in range(pair_relationship.shape[0]):
            for r in range(pair_relationship.shape[1]):
                if (rm_map[c][r] == 1) and (pair_relationship[c][r] == 1): #p
                    thin[c][r] = 0

        if (np.sum(thin == cp_thin)) == (thin.shape[0] * thin.shape[1]):
            break

    cv2.imwrite("thin-lena.bmp", thin)

    #for report
    # cv2.imwrite("thin-lena.jpg", thin)
    # cv2.imwrite("lena.jpg", ori) 

if __name__ == "__main__":
    main()
