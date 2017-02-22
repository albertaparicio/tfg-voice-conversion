echo $(cat `ls ./**/*.i.fv` | wc | cut -f2 -d' ') '* 0.005 / 3600' | bc -l
