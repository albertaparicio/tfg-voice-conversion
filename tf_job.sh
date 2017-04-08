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
# 
# python seq2seq_tf_main.py --no-test

srun -p veu -c16 --mem=32G --gres=gpu:1 python seq2seq_tf_main.py --no-test
