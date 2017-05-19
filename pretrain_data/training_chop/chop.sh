#!/bin/bash - 
#===============================================================================
#
#          FILE: chop.sh
# 
#         USAGE: ./chop.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 05/03/17 17:03
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
DIR='../../training'
SPK=72

FILE_LIST=$(ls $DIR/$SPK | grep .cc | head | cut -f1 -d'.' 2>&1)
# FILE_LIST='T6B72110009'

# echo "$FILE_LIST"

# Iterate over files
for FILE in $(echo "$FILE_LIST"); do
    echo "Processing $FILE"
    FRAMES=$(cat "$DIR/$SPK/$FILE.lf0" | wc -l)
#     echo $FRAMES
    let CHOPS=($FRAMES+500-1)/500
#     echo 'foo'
#     CHOPS=$(echo "$FRAMES / 500" | bc -l)
#     echo $CHOPS

    for (( i=0 ; i<$CHOPS ; i++)); do
#         echo $i
        cat "$DIR/$SPK/$FILE.cc" | tail -n $(($FRAMES-$i*500)) | head -n 500 > ./"$FILE"_"$i".cc
        cat "$DIR/$SPK/$FILE.lf0_log" | tail -n $(($FRAMES-$i*500)) | head -n 500 > ./"$FILE"_"$i".lf0_log
        cat "$DIR/$SPK/$FILE.i.fv" | tail -n $(($FRAMES-$i*500)) | head -n 500 > ./"$FILE"_"$i".i.fv
        cat "$DIR/$SPK/$FILE.lf0_log.uv_mask" | tail -n $(($FRAMES-$i*500)) | head -n 500 > ./"$FILE"_"$i".lf0_log.uv_mask
    done
done

