#! /usr/bin/python3

import os
import cv2
import re
from ctypes import *

init_flag = False
libsqdtrt = None

def detect_init():
    global init_flag, libsqdtrt
    if init_flag:
        return
    ret = os.system("date >> make.log; make libso | tee -a make.log") >> 8
    if ret != 0:
        print("Oops! Make failed. exit " % ret)
        exit(1)
    libsqdtrt = CDLL("libsqdtrt.so")
    wts_str = create_string_buffer("data/sqdtrt_small_concat_v4.wts")
    libsqdtrt.sdt_init(wts_str)
    init_flag = True

def detect_detect(data, height, width, x_shift, y_shift):
    global init_flag, libsqdtrt
    if not init_flag:
        detect_init()
    # print ("hello")
    # res_str = create_string_buffer('\000' * 6400)
    res_str = create_string_buffer(6400)
    libsqdtrt.sdt_detect(data, height, width, x_shift, y_shift, res_str, None, None)
    libsqdtrt.sdt_get_time_detect.restype = c_float
    libsqdtrt.sdt_get_time_misc.restype = c_float
    time_detect = libsqdtrt.sdt_get_time_detect()
    time_misc = libsqdtrt.sdt_get_time_misc()

    strre = r'(\w+) -1 -1 0\.0 ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+) 0\.0 0\.0 0\.0 0\.0 0\.0 0\.0 0\.0 ([0-9.]+)'
    lines = res_str.value.decode().split("\n")
    result = []
    i = 0
    # print ("new img %d results" % len(lines))
    for line in lines:
        # print line
        m = re.search(strre, line)
        if m:
            result.append([])
            result[i].append(m.group(1)) # class
            result[i].append(float(m.group(2))) # x_min
            result[i].append(float(m.group(3))) # y_min
            result[i].append(float(m.group(4))) # x_max
            result[i].append(float(m.group(5))) # y_max
            result[i].append(float(m.group(6))) # prob
            # print("detect: %.2fms misc: %.2fms class: %s x_min: %.2f y_min: %.2f x_max: %.2f y_max: %.2f prob: %.3f" % (time_detect, time_misc, result[i][0], result[i][1], result[i][2], result[i][3], result[i][4], result[i][5]))
            i = i + 1

    return result

def detect_cleanup():
    global init_flag, libsqdtrt
    libsqdtrt.sdt_cleanup()
    init_flag = False
