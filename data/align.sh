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
    ahocoder16_64 $DIR_REF/$FILENAME.wav $DIR_VOC/$DIR_REF/$FILENAME.lf0 $DIR_VOC/$DIR_REF/$FILENAME.mcp $DIR_VOC/$DIR_REF/$FILENAME.vf
    # Convert source vocoder parameter files to ASCII format
    x2x +fa $DIR_VOC/$DIR_REF/$FILENAME.lf0 > $DIR_VOC/$DIR_REF/$FILENAME.lf0.dat
    x2x +fa $DIR_VOC/$DIR_REF/$FILENAME.mcp | do_columns.pl -c 40 > $DIR_VOC/$DIR_REF/$FILENAME.mcp.dat
    x2x +fa $DIR_VOC/$DIR_REF/$FILENAME.vf > $DIR_VOC/$DIR_REF/$FILENAME.vf.dat

    # Encode target sample
    ahocoder16_64 $DIR_TST/$FILENAME.wav $DIR_VOC/$DIR_TST/$FILENAME.lf0 $DIR_VOC/$DIR_TST/$FILENAME.mcp $DIR_VOC/$DIR_TST/$FILENAME.vf
    # Convert source vocoder parameter files to ASCII format
    x2x +fa $DIR_VOC/$DIR_TST/$FILENAME.lf0 > $DIR_VOC/$DIR_TST/$FILENAME.lf0.dat
    x2x +fa $DIR_VOC/$DIR_TST/$FILENAME.mcp | do_columns.pl -c 40 > $DIR_VOC/$DIR_TST/$FILENAME.mcp.dat
    x2x +fa $DIR_VOC/$DIR_TST/$FILENAME.vf > $DIR_VOC/$DIR_TST/$FILENAME.vf.dat

    # Apply dynamic time warping
    dtw -l 40 -v $DIR_FRM/$FILENAME.frames $DIR_VOC/$DIR_TST/$FILENAME.mcp < $DIR_VOC/$DIR_REF/$FILENAME.mcp > /dev/null

    # Convert frames file to ASCII format
    x2x +ia $DIR_FRM/$FILENAME.frames | do_columns.pl -c 2 > $DIR_FRM/$FILENAME.frames.txt

    # Remove binary files
    rm $DIR_VOC/$DIR_REF/$FILENAME.lf0
    rm $DIR_VOC/$DIR_REF/$FILENAME.mcp
    rm $DIR_VOC/$DIR_REF/$FILENAME.vf

    rm $DIR_VOC/$DIR_TST/$FILENAME.lf0
    rm $DIR_VOC/$DIR_TST/$FILENAME.mcp
    rm $DIR_VOC/$DIR_TST/$FILENAME.vf

    rm $DIR_FRM/$FILENAME.frames
done 3< basenames.list

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"