#!/usr/bin/env python

# Make raw data tensors for normalization

import os
import argparse
import time

sDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--member", help="Member", type=int, required=True)
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--year", help="Year", type=int, required=True)
# parser.add_argument("--day", help="Integer day", type=int, required=True)
args = parser.parse_args()
# print(args)
# print()

def is_done(member, variable, year, day):
    fn = ("/scratch/hadsx/cpm/5km/daily/1day/%s/raw_tensors/member_%02d_%04d_%d.tfd") % (
        args.variable,
        args.member,
        year,
        day,
    )
    # print(fn)
    if os.path.exists(fn):
        # print("file done")
        # print(fn)
        return True
    # print("file not done")
    # print(fn)
    return False


count = 0
# for year in range(1980, 1981):
year = args.year
for day in range(0, 900):
# for day in range(0, 2):
    # print(year, day)

# for day in range(0, 11):
    # if is_done(args.member, args.variable, year, day):
    #     continue
    cmd = "%s/make_training_tensor.py --member=%d --variable=%s --year=%04d --day=%d" % (
        sDir,
        args.member,
        args.variable,
        year,
        day,
    )
    print(cmd)

# time.sleep(20)

# count = 0
# for year in range(1980, 1981):
#     for day in range(999, 2000):

#     # for day in range(0, 11):
#         if is_done(args.member, args.variable, year, day):
#             continue
#         cmd = "%s/make_training_tensor.py --member=%d --variable=%s --year=%04d --day=%d" % (
#             sDir,
#             args.member,
#             args.variable,
#             year,
#             day,
#         )
#         print(cmd)

# # time.sleep(20)

# count = 0
# for year in range(1980, 1981):
#     for day in range(1999, 3000):

#     # for day in range(0, 11):
#         if is_done(args.member, args.variable, year, day):
#             continue
#         cmd = "%s/make_training_tensor.py --member=%d --variable=%s --year=%04d --day=%d" % (
#             sDir,
#             args.member,
#             args.variable,
#             year,
#             day,
#         )
#         print(cmd)

# # time.sleep(20)

# count = 0
# for year in range(1980, 1981):
#     for day in range(2999, 3600):

#     # for day in range(0, 11):
#         if is_done(args.member, args.variable, year, day):
#             continue
#         cmd = "%s/make_training_tensor.py --member=%d --variable=%s --year=%04d --day=%d" % (
#             sDir,
#             args.member,
#             args.variable,
#             year,
#             day,
#         )
#         print(cmd)
