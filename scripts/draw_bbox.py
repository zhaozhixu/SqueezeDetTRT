#! /usr/bin/python

import cv2
import sys
import numpy as np

img_name = sys.argv[1]
res_name = sys.argv[2]
img = cv2.imread(img_name)
# cv2.imshow("img", img)
res_file = open(res_name, "r")
res_lines = res_file.readlines()
res_file.close()
for line in res_lines:
    fields = line.split()
    if len(fields) == 0:
        break
    klass = fields[0]
    cords = fields[4:8]
    prob = fields[-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (int(np.round(float(cords[0]))), int(np.round(float(cords[1])))), (int(np.round(float(cords[2]))), int(np.round(float(cords[3])))), (0, 255, 0))
    # cv2.rectangle(img, cords[0:2], cords[2:4], (0, 255, 0), 4)
    # cv2.putText(img, klass+": "+prob, (int(np.round(float(cords[0]))), int(np.round(float(cords[1])))), font, 1, (0, 255, 0))

# cv2.imshow("bbox", img)
cv2.imwrite(img_name+"_bbox.jpg", img)
# cv2.waitKey(0)
