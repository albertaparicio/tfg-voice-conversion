#!/bin/bash -
#===============================================================================
#
#          FILE: seq2seq_align_training.sh
#
#         USAGE: ./seq2seq_align_training.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: interpolate.py from santi-pdp/ahoproc_tools @ GitHub
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

DIR_VOC=vocoded_s2s

# Iterate over speakers list
while read DIR_SPK <&3; do
    echo "Processing files from speaker ${DIR_SPK}"

    # Initialize vocoded parameters directory
    mkdir -p $DIR_VOC/$DIR_SPK

    # Get list of files to align
    ls $DIR_SPK | cut -d '.' -f 1 > seq2seq_basenames.list

    # Extract parameters from files in seq2seq_basenames.list
    while read FILENAME <&4; do
        echo "Processing $DIR_SPK/$FILENAME"

        # Encode vocoder parameters with Ahocoder
        ahocoder16_64 ${DIR_SPK}/${FILENAME}.wav \
            ${DIR_VOC}/${DIR_SPK}/${FILENAME}.lf0 \
            ${DIR_VOC}/${DIR_SPK}/${FILENAME}.mcp \
            ${DIR_VOC}/${DIR_SPK}/${FILENAME}.vf

        # Convert vocoder parameter files to ASCII format
        x2x +fa ${DIR_VOC}/${DIR_SPK}/${FILENAME}.lf0 > \
            ${DIR_VOC}/${DIR_SPK}/${FILENAME}.lf0.dat
        x2x +fa ${DIR_VOC}/${DIR_SPK}/${FILENAME}.mcp | \
            do_columns.pl -c 40 > ${DIR_VOC}/${DIR_SPK}/${FILENAME}.mcp.dat
        x2x +fa ${DIR_VOC}/${DIR_SPK}/${FILENAME}.vf > \
            ${DIR_VOC}/${DIR_SPK}/${FILENAME}.vf.dat

        # Interpolate lfo and vf data
        python $(which interpolate.py) \
            --f0_file ${DIR_VOC}/${DIR_SPK}/${FILENAME}.lf0.dat \
            --vf_file ${DIR_VOC}/${DIR_SPK}/${FILENAME}.vf.dat \
            --no-uv

        # Remove binary files
        rm ${DIR_VOC}/${DIR_SPK}/${FILENAME}.lf0
        rm ${DIR_VOC}/${DIR_SPK}/${FILENAME}.mcp
        rm ${DIR_VOC}/${DIR_SPK}/${FILENAME}.vf

    done 4< seq2seq_basenames.list
done 3< speakers.list

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"

exit 0
