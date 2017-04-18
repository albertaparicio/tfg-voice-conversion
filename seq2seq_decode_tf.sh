#!/bin/bash -
#===============================================================================
#
#          FILE: decode_aho.sh
#
#         USAGE: ./decode_aho.sh params_dir basename
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 09/11/16 18:14
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
# Run predictions
#python seq2seq_decode_prediction.py


#PARAMS_DIR="$1"
# PARAMS_DIR='data/test/predicted/SF1-TF1'
PARAMS_DIR='data/test/tf_predicted/'

while read TRG_SPK <&5; do
    while read SRC_SPK <&4; do
#        while read BASENAME <&3; do
        for file in ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/*.dat; do
            BASENAME=$(basename "${file}")

            echo $BASENAME >> ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}.list

        done

        cat ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}.list | cut -d'.' -f1 | sort | uniq > ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}_uniq.list

        while read BASENAME <&3; do
            # Convert parameters to float data
            echo 'Convert parameters to float data'
            x2x +af ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.vf.dat > ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.vf
            x2x +af ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.lf0.dat > ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.lf0
            cat ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.mcp.dat | do_columns.pl -c 1 | x2x +af > ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.mcp

            mkdir -p ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/wav

            # Decode parameters
            echo 'Decode parameters'
            ahodecoder16_64 ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.lf0 ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.mcp ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/${BASENAME}.vf ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}/wav/${BASENAME}.wav

            echo "Decoded ${BASENAME}"
        done 3< ${PARAMS_DIR}${SRC_SPK}-${TRG_SPK}_uniq.list
    done 4< 'data/test/speakers.list'
done 5< 'data/test/speakers.list'
