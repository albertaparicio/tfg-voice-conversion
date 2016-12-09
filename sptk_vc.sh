#!/bin/bash -
#===============================================================================
#
#          FILE: sptk_vc.sh
#
#         USAGE: ./sptk_vc.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 10/11/16 18:09
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
TRAIN_DIR='data/training'
TEST_DIR='data/test'

GMM_DIR=${TRAIN_DIR}/gmm_vc

SRC_DIR='SF1'
TRG_DIR='TF1'

# TRAIN_FILENAME='100007'
TEST_FILENAME='200007'

mkdir -p ${GMM_DIR}

# Iterate over training files
while read TRAIN_FILENAME <&3; do
    # Convert source and target training files from .wav to .raw
    sox ${TRAIN_DIR}/${SRC_DIR}/${TRAIN_FILENAME}.wav ${GMM_DIR}/${TRAIN_FILENAME}.src.raw
    sox ${TRAIN_DIR}/${TRG_DIR}/${TRAIN_FILENAME}.wav ${GMM_DIR}/${TRAIN_FILENAME}.trg.raw

    # Convert source and target training files from .wav to .raw
    sox ${TEST_DIR}/${SRC_DIR}/${TEST_FILENAME}.wav ${GMM_DIR}/${TEST_FILENAME}.test.src.raw

    # Copy log(f0) from the test target speaker
    x2x +af ${TEST_DIR}/vocoded/${TRG_DIR}/${TEST_FILENAME}.vf.dat > ${GMM_DIR}/${TEST_FILENAME}.test.trg.lf0

    # Compute MCEP of source file
    x2x +sf < ${GMM_DIR}/${TRAIN_FILENAME}.src.raw | frame -l 400 -p 80 | \
        window -l 400 -L 1024 -w 0 | \
        mcep -l 1024 -m 24 -a 0.42 | \
        delta -m 24 -r 1 1 > ${GMM_DIR}/${TRAIN_FILENAME}.src.mcep.delta

    # Compute MCEP of target file
    x2x +sf < ${GMM_DIR}/${TRAIN_FILENAME}.trg.raw | frame -l 400 -p 80 | \
        window -l 400 -L 1024 -w 0 | \
        mcep -l 1024 -m 24 -a 0.42 | \
        delta -m 24 -r 1 1 > ${GMM_DIR}/${TRAIN_FILENAME}.trg.mcep.delta

    # DTW align source and target files
    # TODO concatenar resultats del DTW, i entrar-los tots al gmm
    # TODO Canviar dades usades per a fer servir la sortida del ahocoder enlloc de paràmetres calculats amb SPTK
    dtw -l 50 -p 5 -n 2 ${GMM_DIR}/${TRAIN_FILENAME}.trg.mcep.delta < ${GMM_DIR}/${TRAIN_FILENAME}.src.mcep.delta >> ${GMM_DIR}/train_data.dtw
    echo "Finished ${TRAIN_FILENAME}"

done 3< "${TRAIN_DIR}/basenames.list"

echo "Computing GMM"
gmm -l 100 -m 32 -f ${GMM_DIR}/train_data.dtw > ${GMM_DIR}/train_data.src_trg.gmm

# Perform GMM-based voice conversion
x2x +sf < ${GMM_DIR}/${TEST_FILENAME}.test.src.raw | frame -l 400 -p 80 | \
    window -l 400 -L 1024 -w 0 | \
    mcep -l 1024 -m 24 -a 0.42 | \
    vc -l 25 -m 2 -r 1 1 ${GMM_DIR}/train_data.src_trg.gmm \
    > ${GMM_DIR}/${TEST_FILENAME}.test.trg.mcep

# Synthesise waveform
excite -p 80 ${GMM_DIR}/${TEST_FILENAME}.test.trg.lf0 | \
    mlsadf -m 24 -p 80 -a 0.42 -P 5 ${GMM_DIR}/${TEST_FILENAME}.test.trg.mcep | \
    x2x +fs -o > ${GMM_DIR}/${TEST_FILENAME}.test.trg.raw

# Convert waveform to .wav
echo "Synthesizing waveform"
# sox -r 16000 ${GMM_DIR}/${TEST_FILENAME}.test.trg.raw ${GMM_DIR}/${TEST_FILENAME}.test.trg.wav
rawtowav 16000 16 ${GMM_DIR}/${TEST_FILENAME}.test.trg.raw ${GMM_DIR}/${TEST_FILENAME}.test.trg.wav
echo "Finished!"

