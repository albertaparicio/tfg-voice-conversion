#!/bin/bash -
#===============================================================================
#
#          FILE: tf_job.sh
#
#         USAGE: ./tf_job.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 23/03/17 10:37
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

# SBATCH -p veu             # Partition to submit to
# SBATCH --mem=32G      # Max CPU Memory
# SBATCH --gres=gpu:1

PARTITION='veu'
CPU_CORES=16
MEM='64G'
# BATCH_SIZE=10

# echo "srun -p $PARTITION -c$CPU_CORES --mem=$MEM --gres=gpu:1 python seq2seq_tf_main.py $*"
srun -p $PARTITION -c$CPU_CORES --mem=$MEM --gres=gpu:1 python seq2seq_tf_main.py --no-train --save-h5
