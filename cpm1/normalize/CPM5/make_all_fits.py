#!/usr/bin/env python

# Make all the normalization fits

import os

member=1

sDir = os.path.dirname(os.path.realpath(__file__))

def is_done(member, day, variable):
    fn = "/home/h03/hadsx/extremes/ML/pb1/DCVAE_Climate_sjb1/CPM5/daily/1day/%s/normalised/shape_member_%02d_%d.nc" % (
        variable,
        member,
        day,
    )
    if os.path.exists(fn):
        return True
    return False


count = 0
for variable in (
    "tas",
    "psl",
    "uas",
    "vas",
):
    # for day in range(0, 900):
    # for day in range(0, 10):
    for month in range(6, 7):
        # if is_done(member, month, variable):
        #     continue
        cmd = "%s/fit_for_day.py --member=%d --month=%d --variable=%s" % (
            sDir,
            member,
            month,
            variable,
        )
        print(cmd)
