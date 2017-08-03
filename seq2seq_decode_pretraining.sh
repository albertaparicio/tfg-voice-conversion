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
# python seq2seq_decode_prediction.py


#PARAMS_DIR="$1"
# PARAMS_DIR='data/test/predicted/SF1-TF1'
PARAMS_DIR='predicted_pretrain_data/training_chop/'

while read BASENAME <&3; do
    #BASENAME="$2"

    # Convert parameters to float data
    echo 'Convert parameters to float data'
    x2x +af predicted_${BASENAME}.vf.dat > predicted_${BASENAME}.vf
    x2x +af predicted_${BASENAME}.lf0.dat > predicted_${BASENAME}.lf0
    cat predicted_${BASENAME}.mcp.dat | do_columns.pl -c 1 | x2x +af > predicted_${BASENAME}.mcp

    WAV_DIR=$(echo predicted_${BASENAME} | grep -o '^\(\w*\/\)\{3\}')wav
    mkdir -p ${WAV_DIR}

    # Decode parameters
    echo 'Decode parameters'
    ahodecoder16_64 predicted_${BASENAME}.lf0 predicted_${BASENAME}.mcp predicted_${BASENAME}.vf ${WAV_DIR}/$(basename predicted_${BASENAME}).wav

    echo "Finished predicted_${BASENAME}"
done 3< 'pretrain_data/training_chop/pretrain_basenames.list'
