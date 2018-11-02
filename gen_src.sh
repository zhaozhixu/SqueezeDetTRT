#! /bin/sh

wts=$1
mkdir XJTU-IAIR-Falcon
cp *.h *.cpp *.c *.cu *.py Makefile *.sh scripts/iou.pl XJTU-IAIR-Falcon
mkdir XJTU-IAIR-Falcon/data
cp $wts XJTU-IAIR-Falcon/data
