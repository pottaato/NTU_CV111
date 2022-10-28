#CV HW2 part (c): generate connected components - 蕭恩慈 / B07902095
from PIL import Image, ImageDraw
import numpy as np

class Stack:
    def __init__(self):
        self.list = []
    
    def push(self, item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def empty(self):
        return len(self.list) == 0

ori = Image.open("lena.bmp")
bin = Image.open("binary2-lena.bmp")

w, h = ori.size
threshold = 500


visited = np.zeros((w, h))
labels = np.zeros((w, h))
cnt_labels = np.zeros(w * h)
id = 1

for c in range(w):
    for r in range(h):
        #check if havent been visited, then tagged as visited
        if bin.getpixel((c, r)) == 0:
            visited[c, r] = 1
        #else if have been visited
        elif visited[c, r] == 0:
            stack = Stack()
            stack.push((c, r))

            while not stack.empty():
                col, row = stack.pop()
                if visited[col, row] == 1: continue
                visited[col, row] = 1
                labels[col, row] = id
                cnt_labels[id] += 1

                for x in [(col - 1), col, (col + 1)]:
                    for y in [(row - 1), row, (row + 1)]:
                        if (0 <= x < w) and (0 <= y < h):
                            if (bin.getpixel((x, y)) != 0) and (visited[x, y] == 0):
                                stack.push((x, y))

            id += 1

recs = Stack()

for bound_reg, n in enumerate(cnt_labels):
    if (n >= threshold): 
        r_left = w
        r_top = h
        r_right = 0
        r_bottom = 0

        for x in range(w):
            for y in range(h):
                if (labels[x, y] == bound_reg):
                    if (x < r_left): r_left = x
                    if (x > r_right): r_right = x
                    if (y < r_top): r_top = y
                    if (y > r_bottom): r_bottom = y

        recs.push((r_left, r_top, r_right, r_bottom))

conn = Image.new("RGB", ori.size)
conn_arr = conn.load()

for c in range(w):
    for r in range(h):
        conn_arr[c, r] = (0, 0, 0) if (bin.getpixel((c, r)) == 0) else (255, 255, 255)

while not recs.empty():
    r_left, r_top, r_right, r_bottom = recs.pop()

    d = ImageDraw.Draw(conn)
    d.rectangle(((r_left, r_top), (r_right, r_bottom)), outline = "blue", width = 5)

    r_x = (r_left + r_right) / 2
    r_y = (r_top + r_bottom) / 2

    d.line(((r_x - 10, r_y), (r_x + 10, r_y)), fill = "red", width = 4)
    d.line(((r_x, r_y - 10), (r_x, r_y + 10)), fill = "red", width = 4)

conn.save("bounded-lena.jpg")



