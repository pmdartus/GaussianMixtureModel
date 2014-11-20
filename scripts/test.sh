#!/bin/bash

TRAINING_FILE=$1
TEST_FILE=$2
OUTPUT_FILE="data/output.txt"

python main.py $TRAINING_FILE $TEST_FILE $OUTPUT_FILE

LINE_COUNT=`cat $TEST_FILE | wc -l`
LINE_DIFF=`diff -y --suppress-common-lines $OUTPUT_FILE $TEST_FILE | grep '^' | wc -l`
DIV="($LINE_DIFF/$LINE_COUNT)*100"
echo `bc -l <<< $DIV`
