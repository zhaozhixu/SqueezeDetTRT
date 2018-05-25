#! /usr/bin/python

import cv2
import sys
import os
import re
import numpy as np

def area(xmin, ymin, xmax, ymax):
    if xmax < xmin or ymax < ymin:
        return 0;
    else:
        return (xmax - xmin) * (ymax - ymin)

def compute_iou(xmin, ymin, xmax, ymax, xmin_gt, ymin_gt, xmax_gt, ymax_gt):
    xmin_inter = max(xmin, xmin_gt)
    ymin_inter = max(ymin, ymin_gt)
    xmax_inter = min(xmax, xmax_gt)
    ymax_inter = min(ymax, ymax_gt)
    area_det = area(xmin, ymin, xmax, ymax)
    area_gt = area(xmin_gt, ymin_gt, xmax_gt, ymax_gt)
    area_inter = area(xmin_inter, ymin_inter, xmax_inter, ymax_inter)
    area_union = area_det + area_gt - area_inter
    iou = area_inter / area_union * 1.0
    print("xmin = %d, ymin = %d, xmax = %d, ymax = %d" % (xmin, ymin, xmax, ymax))
    print("xmin_gt = %d, ymin_gt = %d, xmax_gt = %d, ymax_gt = %d" % (xmin_gt, ymin_gt, xmax_gt, ymax_gt))
    print("xmin_inter = %d, ymin_inter = %d, xmax_inter = %d, ymax_inter = %d" % (xmin_inter, ymin_inter, xmax_inter, ymax_inter))
    print("area_inter = %d, area_union = %d" % (area_inter, area_union))
    print("iou = %.2f" % iou)
    return iou

def draw_bboxes(img_dir, bbox_dir, gt_dir, index):
    if not os.path.isdir(img_dir):
        print (img_dir + "is not a valid directory")
        exit()
    if not os.path.isdir(bbox_dir):
        print (bbox_dir + "is not a valid directory")
        exit()
    if not os.path.isdir(gt_dir):
        print (gt_dir + "is not a valid directory")
        exit()

    img_file = os.path.join(img_dir, index+".jpg")
    bbox_file = os.path.join(bbox_dir, index+".xml")
    gt_file = os.path.join(gt_dir, index+".txt")

    bbox_fh = open(bbox_file, "r")
    bbox_str = bbox_fh.read()
    bbox_fh.close()
    m = re.search(r'<xmin>(.+)</xmin>', bbox_str)
    if m:
        xmin_bbox = int(np.round(float(m.group(1))))
    m = re.search(r'<xmax>(.+)</xmax>', bbox_str)
    if m:
        xmax_bbox = int(np.round(float(m.group(1))))
    m = re.search(r'<ymin>(.+)</ymin>', bbox_str)
    if m:
        ymin_bbox = int(np.round(float(m.group(1))))
    m = re.search(r'<ymax>(.+)</ymax>', bbox_str)
    if m:
        ymax_bbox = int(np.round(float(m.group(1))))

    gt_fh = open(gt_file, "r")
    gt_str = gt_fh.read()
    gt_fh.close()
    fields = gt_str.split()
    klass = fields[0]
    xmin_gt = int(np.round(float(fields[4])))
    ymin_gt = int(np.round(float(fields[5])))
    xmax_gt = int(np.round(float(fields[6])))
    ymax_gt = int(np.round(float(fields[7])))

    img = cv2.imread(img_file)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (xmin_gt, ymin_gt), (xmax_gt, ymax_gt), (0, 0, 255))
    cv2.rectangle(img, (xmin_bbox, ymin_bbox), (xmax_bbox, ymax_bbox), (0, 255, 0))
    cv2.putText(img, klass, (xmin_gt, ymin_gt), font, 0.5, (0, 0, 255))
    cv2.imshow("detection", img)
    key = cv2.waitKey(0)

    iou = compute_iou(xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox, xmin_gt, ymin_gt, xmax_gt, ymax_gt)
    print(iou)

def main():
    usage = "usage: " + sys.argv[0] + " DAC_DIR INDEX"
    if len(sys.argv) < 3:
        print (usage)
        exit()
    img_dir = os.path.join(sys.argv[1], "images")
    bbox_dir = os.path.join(sys.argv[1], "result/xml/XJTU-IAIR-Falcon")
    gt_dir = os.path.join(sys.argv[1], "labels")
    index = sys.argv[2]
    draw_bboxes(img_dir, bbox_dir, gt_dir, index)

if __name__ == '__main__':
    main()
