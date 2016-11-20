#!/bin/bash


DIR_REF='SF1'
DIR_TST='TF1'

#FILENAME=t1
FILENAME=100020

# TODO Cleanup or delete
# #-----------------------------------------------------------------
#
# WINDOW_SIZE=20;	FRAME_RATE=4;
# PRM_NAME="mMFPC,ed1E"
# PRM_OPT="-q 2,1 -e 0.1 -l $WINDOW_SIZE -d $FRAME_RATE -v HAMMING -o 20 -c 16"
# ZASKA="Zaska  -P $PRM_NAME $PRM_OPT"
#
#
# # Compute mfcc $DIR_REF/$FILENAME.wav $DIR_TST/$FILENAME.wav => mfcc/$DIR_REF/$FILENAME.prm mfcc/$DIR_TST/$FILENAME.prm
# $ZASKA -t RAW -x wav=msw -n . -p mfcc -F $DIR_REF/$FILENAME $DIR_TST/$FILENAME
#
# # Align: mfcc/$DIR_REF/$FILENAME.prm, mfcc/$DIR_TST/$FILENAME.prm => dtw/${DIR_REF}-$DIR_TST/$FILENAME.dtw
# upc_dtw -t mfcc/$DIR_REF -r mfcc/$DIR_TST -a dtw/${DIR_REF}-$DIR_TST  -w -F $FILENAME
#
# #-----------------------------------------------------------------

# Check: create transcription labels, for each 0.1 seconds in reference:

step=0.1
dur=$(soxi -D $DIR_REF/$FILENAME.wav)

perl -e '
   print "#\n";
   $t = 0;
   $n = 1;
   while ($t <= '$dur') {
     print "$t\t121\tT$n\n";
     $t += '$step';
     $n ++;
   }
' > $DIR_REF/$FILENAME.ts


echo '#' > $DIR_TST/$FILENAME.ts

cat $DIR_REF/$FILENAME.ts | grep -v '#' | cut -f 1 |\
    dtw_project -p frames/$FILENAME.frames.dtw - |\
    perl -ne '
       BEGIN {$n=1}
       chomp;
       print "$_\t121\tR$n\n";
       $n++' >> $DIR_TST/$FILENAME.ts

