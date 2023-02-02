#CV HW8: write a program which does all the task description - 蕭恩慈 / B07902095
import cv2
import numpy as np
import math
import random
import os

kernel = np.ones((5, 5), dtype=np.uint8)
kernel[4, 4] = kernel[4, 0] = kernel[0, 4] = kernel[0, 0] = 0

def Gaussian(ori, amp):
    gauss = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            gauss[c, r] = ori[c, r] + random.gauss(0, 1) * amp
    return gauss

def SaltnPepper(ori, prob):
    snp = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            if (random.uniform(0, 1) < prob):
                snp[c, r] = 0
            elif (random.uniform(0, 1) > (1 - prob)):
                snp[c, r] = 255
            else:
                snp[c, r] = ori[c, r]
    return snp

#size ==> filter size
def BoxFilter(ori, size):
    boxed = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    ctr = size // 2
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            total, cnt = 0, 0
            for x in range(size):
                for y in range(size):
                    if ((0 <= (c + x - ctr) < ori.shape[0]) and (0 <= (r + y - ctr) < ori.shape[1])):
                        total += ori[(c + x - ctr), (r + y - ctr)]
                        cnt += 1
            boxed[c, r] = total // cnt
    return boxed

def MedianFilter(ori, size):
    med = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    ctr = size // 2
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            px = []
            for x in range(size):
                for y in range(size):
                    if ((0 <= (c + x - ctr) < ori.shape[0]) and (0 <= (r + y - ctr) < ori.shape[1])):
                        px.append(ori[(c + x - ctr), (r + y - ctr)])
            px.sort()
            cnt = len(px)
            if (cnt % 2 == 1):
                med[c, r] = px[cnt // 2]
            else:
                tmp = px[(cnt - 1) // 2] / 2 + px[cnt // 2] / 2
                med[c, r] = tmp
    return med

def dilation(ori, kernel):
    dil = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            px = -1
            for x in range(5):
                for y in range(5):
                    if kernel[x, y] == 1:
                        if ((0 <= c + x - 2 < ori.shape[0]) and (0 <= r + y - 2 < ori.shape[1])):
                            if ori[(c + x - 2), (r + y - 2)] > px:
                                px = ori[(c + x - 2), (r + y - 2)]
            dil[c, r] = px

    return dil

def erosion(ori, kernel):
    eros = np.zeros((ori.shape[0], ori.shape[1]), dtype=np.uint8)
    for c in range(ori.shape[0]):
        for r in range(ori.shape[1]):
            px = 256
            for x in range(5):
                for y in range(5):
                    if kernel[x, y] == 1:
                        if ((0 <= c + x - 2 < ori.shape[0]) and (0 <= r + y - 2 < ori.shape[1])):
                            if ori[(c + x - 2), (r + y - 2)] < px:
                                px = ori[(c + x - 2), (r + y - 2)]
            eros[c, r] = px

    return eros

def opening(ori, kernel):
    return dilation(erosion(ori, kernel), kernel)

def closing(ori, kernel):
    return erosion(dilation(ori, kernel), kernel)

def SNR(signal, noise):
    
    avg_signal = 0
    var_signal = 0
    avg_noise = 0
    var_noise = 0

    for c in range(signal.shape[0]):
        for r in range(signal.shape[1]):
            avg_signal += signal[c, r]
            if (noise[c, r] >= signal[c, r]):
                avg_noise += (noise[c, r] - signal[c, r])
            else:
                avg_noise -= (signal[c, r] - noise[c, r])
    
    avg_signal = avg_signal / (signal.shape[0] * signal.shape[1])
    avg_noise = avg_noise / (signal.shape[0] * signal.shape[1])

    for c in range(signal.shape[0]):
        for r in range(signal.shape[1]):
            var_signal += math.pow((signal[c, r] - avg_signal), 2)
            diff = 0
            if (noise[c, r] >= (signal[c, r] + avg_noise)):
                diff = noise[c, r] - signal[c, r] - avg_noise
            else:
                diff = signal[c, r] + avg_noise - noise[c, r]
            var_noise += math.pow(diff, 2)
    
    var_signal = var_signal / (signal.shape[0] * signal.shape[1])
    var_noise = var_noise / (signal.shape[0] * signal.shape[1])

    return math.log(math.sqrt(var_signal) / math.sqrt(var_noise), 10) * 20

def main():
    ori = cv2.imread("lena.bmp", 0)

    if not os.path.exists("gauss-10"):
        os.makedirs("gauss-10")
    if not os.path.exists("gauss-30"):
        os.makedirs("gauss-30")
    if not os.path.exists("snp-01"):
        os.makedirs("snp-01")
    if not os.path.exists("snp-005"):
        os.makedirs("snp-005")

    #1. Gaussian noise w/ amplitude 10 : gauss-10
    gauss_10 = Gaussian(ori, 10)
    cv2.imwrite("lena-gauss-10.bmp", gauss_10)
    #2. Gaussian noise w/ amplitude 30 : gauss-30
    gauss_30 = Gaussian(ori, 30)
    cv2.imwrite("lena-gauss-30.bmp", gauss_30)
    #3. Salt and pepper w/ threshold .1 : snp-01
    snp_01 = SaltnPepper(ori, 0.1)
    cv2.imwrite("lena-snp-01.bmp", snp_01)
    #4. Salt and pepper w/ threshold 0.05 : snp-005
    snp_005 = SaltnPepper(ori, 0.05)
    cv2.imwrite("lena-snp-005.bmp", snp_005)


    #gauss-10
    #1.1. 3x3 box filter : gauss-10-box-3
    gauss10box3 = BoxFilter(gauss_10, 3)
    cv2.imwrite("gauss-10/lena-gauss-10-box-3.bmp", gauss10box3)
    #1.2. 5x5 box filter : gauss-10-box-5
    gauss10box5 = BoxFilter(gauss_10, 5)
    cv2.imwrite("gauss-10/lena-gauss-10-box-5.bmp", gauss10box5)
    #1.3. 3x3 median filter : gauss-10-med-3
    gauss10med3 = MedianFilter(gauss_10, 3)
    cv2.imwrite("gauss-10/lena-gauss-10-med-3.bmp", gauss10med3)
    #1.4. 5x5 median filter: gauss-10-med-5
    gauss10med5 = MedianFilter(gauss_10, 5)
    cv2.imwrite("gauss-10/lena-gauss-10-med-5.bmp", gauss10med5)
    #1.5. opening then closing : gauss-10-open-close
    gauss10_open_close = closing(opening(gauss_10, kernel), kernel)
    cv2.imwrite("gauss-10/lena-gauss-10-open-close.bmp", gauss10_open_close)
    #1.6. closing then opening : gauss-10-close-open
    gauss10_close_open = opening(closing(gauss_10, kernel), kernel)
    cv2.imwrite("gauss-10/lena-gauss-10-close-open.bmp", gauss10_close_open)

    
    #gauss-30    
    #2.1. 3x3 box filter : gauss-30-box-3
    gauss30box3 = BoxFilter(gauss_30, 3)
    cv2.imwrite("gauss-30/lena-gauss-30-box-3.bmp", gauss30box3)
    #2.2. 5x5 box filter : gauss-30-box-5
    gauss30box5 = BoxFilter(gauss_30, 5)
    cv2.imwrite("gauss-30/lena-gauss-30-box-5.bmp", gauss30box5)
    #2.3. 3x3 median filter : gauss-30-med-3
    gauss30med3 = MedianFilter(gauss_30, 3)
    cv2.imwrite("gauss-30/lena-gauss-30-med-3.bmp", gauss30med3)
    #2.4. 5x5 median filter: gauss-30-med-5
    gauss30med5 = MedianFilter(gauss_30, 5)
    cv2.imwrite("gauss-30/lena-gauss-30-med-5.bmp", gauss30med5)
    #2.5. opening then closing : gauss-30-open-close
    gauss30_open_close = closing(opening(gauss_30, kernel), kernel)
    cv2.imwrite("gauss-30/lena-gauss-30-open-close.bmp", gauss30_open_close)
    #2.6. closing then opening : gauss-30-close-open
    gauss30_close_open = opening(closing(gauss_30, kernel), kernel)
    cv2.imwrite("gauss-30/lena-gauss-30-close-open.bmp", gauss30_close_open)


    #snp-01
    #3.1. 3x3 box filter : snp-01-box-3
    snp01box3 = BoxFilter(snp_01, 3)
    cv2.imwrite("snp-01/lena-snp-01-box-3.bmp", snp01box3)
    #3.2. 5x5 box filter : snp-01-box-5
    snp01box5 = BoxFilter(snp_01, 5)
    cv2.imwrite("snp-01/lena-snp-01-box-5.bmp", snp01box5)
    #3.3. 3x3 median filter : snp-01-med-3
    snp01med3 = MedianFilter(snp_01, 3)
    cv2.imwrite("snp-01/lena-snp-01-med-3.bmp", snp01med3)
    #3.4. 5x5 median filter : snp-01-med-5
    snp01med5 = MedianFilter(snp_01, 5)
    cv2.imwrite("snp-01/lena-snp-01-med-5.bmp", snp01med5)
    #3.5. opening then closing : snp-01-open-close
    snp01_open_close = closing(opening(snp_01, kernel), kernel)
    cv2.imwrite("snp-01/lena-snp-01-open-close.bmp", snp01_open_close)
    #3.6. closing then opening : snp-01-close-open
    snp01_close_open = opening(closing(snp_01, kernel), kernel)
    cv2.imwrite("snp-01/lena-snp-01-close-open.bmp", snp01_close_open)

    #snp-005
    #4.1. 3x3 box filter : snp-005-box-3
    snp005box3 = BoxFilter(snp_005, 3)
    cv2.imwrite("snp-005/lena-snp-005-box-3.bmp", snp005box3)
    #4.2. 5x5 box filter : snp-005-box-5
    snp005box5 = BoxFilter(snp_005, 5)
    cv2.imwrite("snp-005/lena-snp-005-box-5.bmp", snp005box5)
    #4.3. 3x3 median filter : snp-005-med-3
    snp005med3 = MedianFilter(snp_005, 3)
    cv2.imwrite("snp-005/lena-snp-005-med-3.bmp", snp005med3)
    #4.4. 5x5 median filter : snp-005-med-5
    snp005med5 = MedianFilter(snp_005, 5)
    cv2.imwrite("snp-005/lena-snp-005-med-5.bmp", snp005med5)
    #4.5. opening then closing : snp-005-open-close
    snp005_open_close = closing(opening(snp_005, kernel), kernel)
    cv2.imwrite("snp-005/lena-snp-005-open-close.bmp", snp005_open_close)
    #4.6. closing then opening : snp-005-close-open
    snp005_close_open = opening(closing(snp_005, kernel), kernel)
    cv2.imwrite("snp-005/lena-snp-005-close-open.bmp", snp005_close_open)


    #SNR
    f = open("SNR.txt", "w")
    f.write("SNR of each image: \n\n")
    #1
    gauss_10_snr = SNR(ori, gauss_10)
    f.write("Gaussian 10: " + str(gauss_10_snr) + "\n")
    #2
    gauss_30_snr = SNR(ori, gauss_30)
    f.write("Gaussian 30: " + str(gauss_30_snr) + "\n")
    #3
    snp_01_snr = SNR(ori, snp_01)
    f.write("Salt and Pepper 0.1: " + str(snp_01_snr) + "\n")
    #4
    snp_005_snr = SNR(ori, snp_005)
    f.write("Salt and Pepper 0.05: " + str(snp_005_snr) + "\n")
    
    f.write("\n")

    #1.1
    gauss10box3_snr = SNR(ori, gauss10box3)
    f.write("Gaussian 10 + 3x3 box filter: " + str(gauss10box3_snr) + "\n")
    #1.2
    gauss10box5_snr = SNR(ori, gauss10box5)
    f.write("Gaussian 10 + 5x5 box filter: " + str(gauss10box5_snr) + "\n")
    #1.3
    gauss10med3_snr = SNR(ori, gauss10med3)
    f.write("Gaussian 10 + 3x3 median filter: " + str(gauss10med3_snr) + "\n")
    #1.4
    gauss10med5_snr = SNR(ori, gauss10med5)
    f.write("Gaussian 10 + 5x5 median filter: " + str(gauss10med5_snr) + "\n")
    #1.5
    gauss10_open_close_snr = SNR(ori, gauss10_open_close)
    f.write("Gaussian 10 + opening then closing filter: " + str(gauss10_open_close_snr) + "\n")
    #1.6
    gauss10_close_open_snr = SNR(ori, gauss10_close_open)
    f.write("Gaussian 10 + closing then opening filter: " + str(gauss10_close_open_snr) + "\n")

    f.write("\n")

    #2.1
    gauss30box3_snr = SNR(ori, gauss30box3)
    f.write("Gaussian 30 + 3x3 box filter: " + str(gauss30box3_snr) + "\n")
    #2.2
    gauss30box5_snr = SNR(ori, gauss30box5)
    f.write("Gaussian 30 + 5x5 box filter: " + str(gauss30box5_snr) + "\n")
    #2.3
    gauss30med3_snr = SNR(ori, gauss30med3)
    f.write("Gaussian 30 + 3x3 median filter: " + str(gauss30med3_snr) + "\n")
    #2.4
    gauss30med5_snr = SNR(ori, gauss30med5)
    f.write("Gaussian 30 + 5x5 median filter: " + str(gauss30med5_snr) + "\n")
    #2.5
    gauss30_open_close_snr = SNR(ori, gauss30_open_close)
    f.write("Gaussian 30 + opening then closing filter: " + str(gauss30_open_close_snr) + "\n")
    #2.6
    gauss30_close_open_snr = SNR(ori, gauss30_close_open)
    f.write("Gaussian 30 + closing then opening filter: " + str(gauss30_close_open_snr) + "\n")

    f.write("\n")

    #3.1
    snp01box3_snr = SNR(ori, snp01box3)
    f.write("Salt and Pepper 0.1 + 3x3 box filter: " + str(snp01box3_snr) + "\n")
    #3.2
    snp01box5_snr = SNR(ori, snp01box5)
    f.write("Salt and Pepper 0.1 + 5x5 box filter: " + str(snp01box5_snr) + "\n")
    #3.3
    snp01med3_snr = SNR(ori, snp01med3)
    f.write("Salt and Pepper 0.1 + 3x3 median filter: " + str(snp01med3_snr) + "\n")
    #3.4
    snp01med5_snr = SNR(ori, snp01med5)
    f.write("Salt and Pepper 0.1 + 5x5 median filter: " + str(snp01med5_snr) + "\n")
    #3.5
    snp01_open_close_snr = SNR(ori, snp01_open_close)
    f.write("Salt and Pepper 0.1 + opening then closing filter: " + str(snp01_open_close_snr) + "\n")
    #3.6
    snp01_close_open_snr = SNR(ori, snp01_close_open)
    f.write("Salt and Pepper 0.1 + closing then opening filter: " + str(snp01_close_open_snr) + "\n")

    f.write("\n")

    #4.1
    snp005box3_snr = SNR(ori, snp005box3)
    f.write("Salt and Pepper 0.05 + 3x3 box filter: " + str(snp005box3_snr) + "\n")
    #4.2
    snp005box5_snr = SNR(ori, snp005box5)
    f.write("Salt and Pepper 0.05 + 5x5 box filter: " + str(snp005box5_snr) + "\n")
    #4.3
    snp005med3_snr = SNR(ori, snp005med3)
    f.write("Salt and Pepper 0.05 + 3x3 median filter: " + str(snp005med3_snr) + "\n")
    #4.4
    snp005med5_snr = SNR(ori, snp005med5)
    f.write("Salt and Pepper 0.05 + 5x5 median filter: " + str(snp005med5_snr) + "\n")
    #4.5
    snp005_open_close_snr = SNR(ori, snp005_open_close)
    f.write("Salt and Pepper 0.05 + opening then closing filter: " + str(snp005_open_close_snr) + "\n")
    #4.6
    snp005_close_open_snr = SNR(ori, snp005_close_open)
    f.write("Salt and Pepper 0.05 + closing then opening filter: " + str(snp005_close_open_snr) + "\n")

    f.close()

if __name__ == "__main__":
    main()
