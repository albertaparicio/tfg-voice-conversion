#!/bin/bash -
#===============================================================================
#
#          FILE: align.sh
#
#         USAGE: ./align.sh
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

#DIR_REF=test_source
#DIR_TST=test_target
DIR_REF=SF1
DIR_TST=TF1
DIR_VOC=vocoded
DIR_FRM=frames

# FILENAME=SF1_TF1_200001

# Initialize vocoded and warped frames directories
mkdir -p ${DIR_VOC}/${DIR_REF}
mkdir -p ${DIR_VOC}/${DIR_TST}

mkdir -p ${DIR_FRM}

# Get list of files to align. Only pick common files between source and target directory
#ls ${DIR_REF} | perl -pe 's/.wav//' > basenames.list
diff ${DIR_REF} ${DIR_TST} | grep 'and' | cut -d '/' -f 2 | cut -d '.' -f 1 > basenames.list

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
    interpolate.py --f0_file ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0.dat --vf_file ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf.dat --no-uv
    # Target
    interpolate.py --f0_file ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0.dat --vf_file ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf.dat --no-uv

    # Apply dynamic time warping
    dtw -l 40 -s ${DIR_FRM}/${FILENAME}.score -v ${DIR_FRM}/${FILENAME}.frames ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp < ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp > /dev/null

    # Convert frames file to ASCII format
    x2x +ia ${DIR_FRM}/${FILENAME}.frames | do_columns.pl -c 2 > ${DIR_FRM}/${FILENAME}.frames.txt
    x2x +fa ${DIR_FRM}/${FILENAME}.score > ${DIR_FRM}/${FILENAME}.dtw_score

    # Remove binary files
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.lf0
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.mcp
    rm ${DIR_VOC}/${DIR_REF}/${FILENAME}.vf

    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.lf0
    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.mcp
    rm ${DIR_VOC}/${DIR_TST}/${FILENAME}.vf

    rm ${DIR_FRM}/${FILENAME}.frames
    rm ${DIR_FRM}/${FILENAME}.score
done 3< basenames.list

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"
