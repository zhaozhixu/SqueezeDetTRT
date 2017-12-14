#! /usr/bin/python

import sys
import os

usage = "usage: " + sys.argv[0] + "IMAGE_DIR EVAL_LIST_FILE RESULT_DIR"

def main():
    if len(sys.argv) < 3:
        print(usage)
        exit()

if __name__ == '__main__':
    main()
