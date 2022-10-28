#CV HW3: generate image(with specification) and its histogram - 蕭恩慈 / B07902095
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv

ori = Image.open("lena.bmp")
w, h = ori.size

#part(a): original
ori.save("original-lena.bmp")
#ori.save("original-lena.jpg")

ori_hist = np.zeros(256)

for c in range(w): 
    for r in range(h): 
        x = ori.getpixel((c, r))
        ori_hist[x] += 1

with open("ori_hist.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(ori_hist)

plt.bar(range(len(ori_hist)), ori_hist)
plt.savefig("original-lena-histogram.jpg")

#part(b): 1/3 intensified
intense = Image.new("L", ori.size)

for c in range(w):
    for r in range(h):
        x = ori.getpixel((c, r))
        intense.putpixel((c, r), x // 3)

intense.save("intensify-lena.bmp")
#intense.save("intensify-lena.jpg")

intense_hist = np.zeros(256)

for c in range(w):
    for r in range(h):
        x = intense.getpixel((c, r))
        intense_hist[x] += 1

with open("intese_hist.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(intense_hist)

plt.clf()
plt.bar(range(len(intense_hist)), intense_hist)
plt.savefig("intensify-lena-histogram.jpg")
#plt.show()

#part(C): equalization to b
cdf = np.zeros(256)

for i in range(len(cdf)):
    cdf[i] = 255 * np.sum(intense_hist[0:i + 1]) / w / h
    #print(cdf[i])

equal = Image.new("L", ori.size)

for c in range(w):
    for r in range(h):
        x = intense.getpixel((c, r))
        equal.putpixel((c, r), int(cdf[x]))

equal.save("equalized-lena.bmp")
#equal.save("equalized-lena.jpg")

equal_hist = np.zeros(256)

for c in range(w):
    for r in range(h):
        x = equal.getpixel((c, r))
        equal_hist[x] += 1

with open("equal_hist.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(equal_hist)

plt.clf()
plt.bar(range(len(equal_hist)), equal_hist)
plt.savefig("equalized-lena-histogram.jpg")
#plt.show()
