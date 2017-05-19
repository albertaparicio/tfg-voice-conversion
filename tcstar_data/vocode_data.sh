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
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 10/20/2016 12:29
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

vocode_sequence()
{
#     echo "Processing $2/$3"

    # Encode vocoder parameters with Ahocoder
    ahocoder16_64 "$2"/"$3".wav \
        "$1"/"$2"/"$3".lf0 \
        "$1"/"$2"/"$3".mcp \
        "$1"/"$2"/"$3".vf

    # Convert vocoder parameter files to ASCII format
    x2x +fa "$1"/"$2"/"$3".lf0 > \
        "$1"/"$2"/"$3".lf0.dat
    x2x +fa "$1"/"$2"/"$3".mcp | \
        do_columns.pl -c 40 > "$1"/"$2"/"$3".mcp.dat
    x2x +fa "$1"/"$2"/"$3".vf > \
        "$1"/"$2"/"$3".vf.dat

    # Interpolate lfo and vf data
    python $(which interpolate.py) \
        --f0_file "$1"/"$2"/"$3".lf0.dat \
        --vf_file "$1"/"$2"/"$3".vf.dat \
        --no-uv

    # Remove binary files
    rm "$1"/"$2"/"$3".lf0
    rm "$1"/"$2"/"$3".mcp
    rm "$1"/"$2"/"$3".vf
}
export -f vocode_sequence

# Start time of execution
START=$(date +%s.%N)

DIR_VOC=tcstar_vocoded
ls 75 | cut -d'.' -f1 > tcstar_basenames.list

while read line; do mkdir -p "$DIR_VOC/${line%/*}"; done < speakers.list

parallel --bar vocode_sequence {1} {2} {3} ::: "$DIR_VOC" ::: $(cat speakers.list) ::: $(cat tcstar_basenames.list)

# # Iterate over speakers list
# while read DIR_SPK <&3; do
#     echo "Processing files from speaker ${DIR_SPK}"
#
#     # Initialize vocoded parameters directory
#     mkdir -p $DIR_VOC/$DIR_SPK
#
#     # Get list of files to align
#     ls $DIR_SPK | cut -d '.' -f 1 > tcstar_basenames.list
#
#     # Extract parameters from files in seq2seq_basenames.list
#     while read FILENAME <&4; do
#
#
#     done 4<
# done 3<

# Set End time of execution
END=$(date +%s.%N)

# Compute and echo the time of execution
DIFF=$(echo "$END-$START" | bc)

echo "Execution time: $DIFF seconds"

exit 0
