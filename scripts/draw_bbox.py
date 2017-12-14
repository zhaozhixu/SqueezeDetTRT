#! /usr/bin/python

import cv2
import sys
import os
import re
import numpy as np

plot_prob_thresh = 0.4

def draw_bboxes(img_dir, res_dir, bbox_dir, subfix):
    if not os.path.isdir(res_dir):
        print (res_dir + "is not a valid directory")
        exit()
    if not os.path.isdir(img_dir):
        print (img_dir + "is not a valid directory")
        exit()
    if not os.path.exists(bbox_dir):
        os.makedirs(bbox_dir)

    res_file_list = os.listdir(res_dir)
    for i in range(len(res_file_list)):
        res_name = res_file_list[i]
        img_name = re.sub(r'\.txt', '.'+subfix, res_name)
        img_path = os.path.join(img_dir, img_name)
        res_path = os.path.join(res_dir, res_name)
        bbox_name = re.sub(r'\.txt', '_bbox.'+subfix, res_name)
        bbox_path = os.path.join(bbox_dir, bbox_name)

        sys.stdout.write("\r({:d}/{:d}) image: {:s}".format(i+1, len(res_file_list), img_name))
        sys.stdout.flush()
        img = cv2.imread(img_path)
        res_file = open(res_path, "r")
        res_lines = res_file.readlines()
        res_file.close()
        for line in res_lines:
            fields = line.split()
            if len(fields) == 0:
                break
            prob = fields[-1]
            if float(prob) < plot_prob_thresh:
                continue
            klass = fields[0]
            cords = fields[4:8]
            for i in range(len(cords)):
                cords[i] = int(np.round(float(cords[i])))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0))
            cv2.putText(img, klass+": "+prob, (cords[0], cords[1]), font, 0.5, (0, 255, 0))

        cv2.imwrite(bbox_path, img)
    sys.stdout.write("\n")

def main():
    usage = "usage: " + sys.argv[0] + " IMAGE_DIR RESULT_DIR BBOX_DIR IMAGE_SUBFIX"
    if len(sys.argv) < 5:
        print (usage)
        exit()
    img_dir = sys.argv[1]
    res_dir = sys.argv[2]
    bbox_dir = sys.argv[3]
    subfix = sys.argv[4]
    draw_bboxes(img_dir, res_dir, bbox_dir, subfix)

if __name__ == '__main__':
    main()
