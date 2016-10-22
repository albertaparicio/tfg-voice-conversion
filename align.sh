#!/bin/bash -
#===============================================================================
#
#          FILE: align.sh
#
#         USAGE: ./align.sh
#
#   DESCRIPTION: to convert Ahocoder data into ASCII files, use:
#                $ x2x +fa INFILE > OUTFILE.dat
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 10/20/2016 12:29
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
# Start time of execution
START=$(date +%s.%N)

DIR_REF=test_source
DIR_TST=test_target
DIR_VOC=vocoded
DIR_FRM=frames

# FILENAME=SF1_TF1_200001

# Initialize vocoded and warped frames directories
mkdir -p $DIR_VOC/$DIR_REF
mkdir -p $DIR_VOC/$DIR_TST

mkdir -p $DIR_FRM

# Get list of files to align
ls $DIR_REF | perl -pe 's/.wav//' > basenames.list

# Perform alignment on each file in basenames.list
while read FILENAME <&3; do

    ## Encode vocoder parameters with Ahocoder
    # Encode source sample
    ahocoder16_64 $DIR_REF/$FILENAME.wav $DIR_VOC/$DIR_REF/$FILENAME.f0 $DIR_VOC/$DIR_REF/$FILENAME.mfcc $DIR_VOC/$DIR_REF/$FILENAME.vf

    # Encode target sample
    ahocoder16_64 $DIR_TST/$FILENAME.wav $DIR_VOC/$DIR_TST/$FILENAME.f0 $DIR_VOC/$DIR_TST/$FILENAME.mfcc $DIR_VOC/$DIR_TST/$FILENAME.vf

    # Apply dynamic time warping
    dtw -l 40 -v $DIR_FRM/$FILENAME.frames $DIR_VOC/$DIR_TST/$FILENAME.mfcc < $DIR_VOC/$DIR_REF/$FILENAME.mfcc > /dev/null

    # Convert frames file to ASCII format
    x2x +ia $DIR_FRM/$FILENAME.frames | do_columns.pl -c 2 > $DIR_FRM/$FILENAME.txt

    # TODO remove .frames file
    rm $DIR_FRM/$FILENAME.frames
done 3< basenames.list

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"
