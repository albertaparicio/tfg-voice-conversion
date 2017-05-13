#!/bin/bash -
#===============================================================================
#
#          FILE: trim.sh
#
#         USAGE: ./trim.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 13/05/17 17:19
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

trim_sequence()
{
# $1 - Original dataset location
# $2 - Destination dataset location
# $3 - Speaker name
# $4 - Filename (no extension)

# Construct filenames from arguments
ORIG="$1/$3_wav_unamed/$4.wav"
OUTNAME="$2/training/$3/$4.wav"
#ORIG=75/0.wav
#OUTNAME=0.wav

PHS2=$(cat $1/$3_mar_unamed/$4.mar | grep PHS | sed '2q;d' | cut -d ' ' -f2 | cut -d',' -f1)
PHSL1=$(cat $1/$3_mar_unamed/$4.mar | grep PHS | sed '$!d' | cut -d ' ' -f2 | cut -d',' -f1)
PHSL3=$(cat $1/$3_mar_unamed/$4.mar | grep PHS | sed '$!d' | cut -d ' ' -f2 | cut -d',' -f3)

BEG=$(python -c "print(max($PHS2,0.1)-0.1)")
END_SIL=$(python -c "print(min($PHSL3 - $PHSL1,0.1))")
END=$(echo "$PHSL1 - $BEG + $END_SIL" | bc -l)

sox "$ORIG" "$OUTNAME" trim "$BEG" "$END"
}
export -f trim_sequence

# Start time of execution
START=$(date +%s.%N)

ORIG_PATH="$1"
DEST_PATH="$2"

# Make speakers and filenames lists
ls "$ORIG_PATH" | grep wav_unamed | cut -d'_' -f1 > "$DEST_PATH"/speakers.list
ls "$ORIG_PATH"/$(ls "$ORIG_PATH" | grep wav_unamed | cut -d'_' -f1 | sed q)_wav_unamed | cut -d'.' -f1 > "$DEST_PATH"/basenames.list

# Make directories of speakers
while read line; do mkdir -p "$DEST_PATH/training/$line"; done < "$DEST_PATH"/speakers.list

parallel --bar trim_sequence {1} {2} {3} {4} ::: "$ORIG_PATH" ::: "$DEST_PATH" ::: $(cat "$DEST_PATH"/speakers.list) ::: $(cat "$DEST_PATH"/basenames.list)