#! /bin/sh

if [ $# -eq 0 ]; then
    echo "usage: $0 <weight_file>"
    exit 0
fi

weight_file=$1

mkdir XJTU-IAIR-Falcon
cp *.h *.cpp *.c *.cu *.py Makefile *.sh XJTU-IAIR-Falcon
mkdir XJTU-IAIR-Falcon/data
# cp data/sqdtrt_dji.wts XJTU-IAIR-Falcon/data
cp $weight_file XJTU-IAIR-Falcon/data
