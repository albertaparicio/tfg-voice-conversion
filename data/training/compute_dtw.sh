#!/bin/bash -
#===============================================================================
#
#          FILE: extract_features.sh
#
#         USAGE: ./extract_features.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 18/11/16 14:08
#      REVISION:  ---
#===============================================================================
# TODO Code cleanup
set -o nounset                              # Treat unset variables as an error
# Start time of execution
START=$(date +%s.%N)

#DIR_REF=test_source
#DIR_TST=test_target
DIR_REF=SF1
DIR_TST=TF1
DIR_VOC=vocoded
DIR_FRM=frames

# FILENAME=SF1_TF1_200001

# Initialize vocoded and warped frames directories
# mkdir -p ${DIR_VOC}/${DIR_REF}
# mkdir -p ${DIR_VOC}/${DIR_TST}
# 
# mkdir -p ${DIR_FRM}

# Get list of files to align. Only pick common files between source and target directory
#ls ${DIR_REF} | perl -pe 's/.wav//' > basenames.list
diff ${DIR_REF} ${DIR_TST} | grep 'and' | cut -d '/' -f 2 | cut -d '.' -f 1 > basenames.list

# Initialize silence sample file
sox -n -r 16000 -c 1  -e signed-integer -b 16 silence.wav  trim 0 0.025

# Perform alignment on each file in basenames.list
while read FILENAME <&3; do
    echo "Processing ${FILENAME}..."

    # Join silence sample with file to process
    # This is done to compensate for the zero-padding
    # that Ahocoder uses in the beginning of the sound files
    sox silence.wav ${DIR_REF}/${FILENAME}.wav ${DIR_REF}/${FILENAME}_sil.wav
    sox silence.wav ${DIR_TST}/${FILENAME}.wav ${DIR_TST}/${FILENAME}_sil.wav

    # Apply UPC's DTW
    WINDOW_SIZE=30; # 30ms window size
    FRAME_RATE=5; # 5ms window shift
    PRM_NAME="mMFPC,ed1E"
    PRM_OPT="-q 2,1 -e 0.1 -l $WINDOW_SIZE -d $FRAME_RATE -v HAMMING -o 20 -c 16"
    ZASKA="Zaska  -P $PRM_NAME $PRM_OPT"

    # Compute mfcc $DIR_REF/${FILENAME}.wav $DIR_TST/${FILENAME}.wav => mfcc/$DIR_REF/${FILENAME}.prm mfcc/$DIR_TST/${FILENAME}.prm
    ${ZASKA} -t RAW -x wav=msw -n . -p mfcc -F ${DIR_REF}/${FILENAME}_sil ${DIR_TST}/${FILENAME}_sil

    # Align: mfcc/${DIR_REF}/${FILENAME}.prm, mfcc/${DIR_TST/${FILENAME}.prm => dtw/${DIR_REF}-${DIR_TST}/${FILENAME}.dtw
    b=2
    dtw -b -${b} -t mfcc/${DIR_REF} -r mfcc/${DIR_TST} -a dtw/beam${b} -w -B -f -F ${FILENAME}

    # Remove files with silence attached to them
    rm ${DIR_REF}/${FILENAME}_sil.wav ${DIR_TST}/${FILENAME}_sil.wav

done 3< basenames.list

# Remove auxiliar parameters
rm -r mfcc/ silence.wav

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"

