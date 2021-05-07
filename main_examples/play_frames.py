import cv2
import os

img_array = []
frames = sorted(os.listdir('frames'))

for filename in (frames):
    img = cv2.imread(f'frames/{filename}')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('hexapod.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
