#!/bin/bash - 
#===============================================================================
#
#          FILE: fv_interpolate.sh
# 
#         USAGE: ./fv_interpolate.sh
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: readlink version 8.25 or later
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 03/02/17 19:47
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

find $(readlink -f ".") -name "*" | grep --regexp=.fv$ > fv_guia.list
python `which interpolate.py` --vf_guia=fv_guia.list --no-uv
