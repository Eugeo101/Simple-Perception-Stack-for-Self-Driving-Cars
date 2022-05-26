#!/bin/bash
for arg in "$@"
do
index=$(echo $arg | cut -f1 -d=)
val=$(echo $arg | cut -f2 -d=)
case $index in
debug_mode) x=$val;;
*)
esac
done

python3 -W ignore pipeline.py $1 $2 $3 $x