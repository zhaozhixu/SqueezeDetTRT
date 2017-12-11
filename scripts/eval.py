#! /usr/bin/python

import sys

usage = "usage: " + sys.argv[0] + "IMAGE_DIR EVAL_LIST_FILE RESULT_DIR"

if sys.argc < 3:
    print(usage)
    exit()
