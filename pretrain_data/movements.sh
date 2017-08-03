ls | grep .cc | head | cut -f1 -d'.' 2>&1 | awk '{ print $0 ".*"; fflush() }' | xargs -I@ sh -c 'cp -v @ -t ../training/'
