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
PARAMS_DIR="$1"
BASENAME="$2"

# Convert parameters to float data
echo 'Convert parameters to float data'
x2x +af ${PARAMS_DIR}/${BASENAME}.vf.dat > ${PARAMS_DIR}/${BASENAME}.vf
x2x +af ${PARAMS_DIR}/${BASENAME}.lf0.dat > ${PARAMS_DIR}/${BASENAME}.lf0
cat ${PARAMS_DIR}/${BASENAME}.mcp.dat | do_columns.pl -c 1 | x2x +af > ${PARAMS_DIR}/${BASENAME}.mcp

mkdir -p ${PARAMS_DIR}/wav

# Decode parameters
echo 'Decode parameters'
ahodecoder16_64 ${PARAMS_DIR}/${BASENAME}.lf0 ${PARAMS_DIR}/${BASENAME}.mcp ${PARAMS_DIR}/${BASENAME}.vf ${PARAMS_DIR}/wav/${BASENAME}.wav

echo "Finished ${BASENAME}"

