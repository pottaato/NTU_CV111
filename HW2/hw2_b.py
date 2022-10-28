#CV HW2 part (b): generate image's histogram - 蕭恩慈 / B07902095
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

ori = Image.open("lena.bmp")
w, h = ori.size

histogram = np.zeros(256)

for c in range(w): 
    for r in range(h): 
        x = ori.getpixel((c, r))
        histogram[x] += 1

with open("histogram-lena.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(histogram)

plt.bar(range(len(histogram)), histogram)
plt.savefig("histogram-lena.jpg")
plt.show()
