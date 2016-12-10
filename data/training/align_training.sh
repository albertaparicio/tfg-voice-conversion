#!/bin/bash -
#===============================================================================
#
#          FILE: align_training.sh
#
#         USAGE: ./align_training.sh
#
#   DESCRIPTION:
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

DIR_REF=SF1
DIR_TST=TF1
DIR_VOC=vocoded

# Initialize vocoded and warped frames directories
mkdir -p ${DIR_VOC}/${DIR_REF}
mkdir -p ${DIR_VOC}/${DIR_TST}

# # Get list of files to align. Only pick common files between source and target directory
# diff ${DIR_REF} ${DIR_TST} | grep 'and' | cut -d '/' -f 2 | cut -d '.' -f 1 > basenames.list
# Get list of files to align
ls $DIR_REF | cut -d '.' -f 1 > basenames.list

# Perform alignment on each file in basenames.list
while read FILENAME <&3; do
    echo "Processing $FILENAME..."
    ## Encode vocoder parameters with Ahocoder
    # Encode source sample
    ahocoder16_64 ${DIR_REF}/${FILENAME}.wav ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0 ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf
    # Convert source vocoder parameter files to ASCII format
    x2x +fa ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0 > ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0.dat
    x2x +fa ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp | do_columns.pl -c 40 > ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp.dat
    x2x +fa ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf > ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf.dat

    # Encode target sample
    ahocoder16_64 ${DIR_TST}/${FILENAME}.wav ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0 ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf
    # Convert source vocoder parameter files to ASCII format
    x2x +fa ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0 > ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0.dat
    x2x +fa ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp | do_columns.pl -c 40 > ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp.dat
    x2x +fa ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf > ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf.dat

    # Interpolate lfo and vf data
    # Source
    python $(which interpolate.py) --f0_file ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0.dat --vf_file ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf.dat --no-uv
    # Target
    python $(which interpolate.py) --f0_file ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0.dat --vf_file ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf.dat --no-uv

    # Remove binary files
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf

    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0
    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp
    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf

done 3< basenames.list

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"
