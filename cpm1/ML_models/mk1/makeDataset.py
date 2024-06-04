# Make tf.data.Datasets from CPM5 monthly averages

# This is a generic script to make a TensorFlow Dataset
# Follow the instructions in autoencoder.py to use it.

import os
import sys
import random
import tensorflow as tf
import numpy as np
import random


# Load a pre-standardised tensor from a list of files
def load_tensor(file_names):
    sict = tf.io.read_file(file_names[0])
    imt = tf.io.parse_tensor(sict, np.float32)
    ima = tf.reshape(imt, [244, 180, 1])
    for fni in range(1, len(file_names)):
        sict = tf.io.read_file(file_names[fni])
        imt = tf.io.parse_tensor(sict, np.float32)
        imt = tf.reshape(imt, [244, 180, 1])
        ima = tf.concat([ima, imt], 2)
    return ima


# Find out how many tensors available for each year from a source
def getDataAvailability(source):
    # dir = "%s/DCVAE-Climate/normalized_datasets/%s" % (os.getenv("SCRATCH"), source)
    # dir = "/scratch/hadsx/cpm/5km/daily/1day/%s/norm_tensors" % (source) # args.variable
    dir = "%s/%s/norm_tensors" % (os.getenv("MLSCRATCH"), source) # args.variable

    aFiles = os.listdir(dir)
    firstYr = 3000
    lastYr = 0
    maxCount = 0
    filesYM = {}
    for fN in aFiles:
        # fN: member_01_1980_99.tfd
        year = int(fN[10:14]) # int(fN[:4])
        # month = int(fN[5:7])
        idot    = fN.find('.')
        day     = int(fN[15:(idot-2)]) # int(fN[5:7])
        if year < firstYr:
            firstYr = year
        if year > lastYr:
            lastYr = year
        key = "%04d%02d" % (year, day)
        if key not in filesYM:
            filesYM[key] = []
        filesYM[key].append("%s/%s" % (dir, fN))
        if len(filesYM[key]) > maxCount:
            maxCount = len(filesYM[key])
    return (firstYr, lastYr, maxCount, filesYM)


# Make a set of input filenames
def getFileNamesOld(
    sources,
    purpose,
    firstYr,
    lastYr,
    testSplit,
    maxTrainingMonths,
    maxTestMonths,
    correlatedEnsembles,
    maxEnsembleCombinations,):
    avail = {}
    maxCount = 1
    for source in sources:
        avail[source] = getDataAvailability(source)
        if firstYr is None or avail[source][0] > firstYr:
            firstYr = avail[source][0]
        if lastYr is None or avail[source][1] < lastYr:
            lastYr = avail[source][1]
        if correlatedEnsembles:
            maxCount = avail[source][2]  # always the same
        else:
            maxCount *= avail[source][2]

    # Make file name lists for available years - repeating if there are multiple ensemble members
    aMonths = []
    fNames = {}
    for rep in range(min(maxCount, maxEnsembleCombinations)):
        for year in range(firstYr, lastYr + 1):
            for month in range(1, 13):
                mnth = "%04d%02d" % (year, month)
                smnth = []
                bad = False
                for source in sources:
                    if mnth in avail[source][3]:
                        if correlatedEnsembles:
                            smnth.append(avail[source][3][mnth][rep])
                        else:
                            smnth.append(random.sample(avail[source][3][mnth], 1)[0])
                    else:
                        bad = True
                        break
                if bad:
                    continue
                mnth += "%05d" % rep
                aMonths.append(mnth)
                fNames[mnth] = smnth

    # Test/Train split
    if purpose is not None:
        test_ns = list(range(0, len(aMonths), testSplit))
        if purpose == "Train":
            aMonths = [aMonths[x] for x in range(len(aMonths)) if x not in test_ns]
        elif purpose == "Test":
            aMonths = [aMonths[x] for x in range(len(aMonths)) if x in test_ns]
        else:
            raise Exception("Unsupported purpose " + purpose)

    aMonths.sort()  # Months in time order (validation plots)

    # Limit maximum data size
    if purpose == "Train" and maxTrainingMonths is not None:
        if len(aMonths) >= maxTrainingMonths:
            aMonths = aMonths[0:maxTrainingMonths]
        else:
            raise ValueError(
                "Only %d months available, can't provide %d"
                % (len(aMonths), maxTrainingMonths)
            )
    if purpose == "Test" and maxTestMonths is not None:
        if len(aMonths) >= maxTestMonths:
            aMonths = aMonths[0:maxTestMonths]
        else:
            raise ValueError(
                "Only %d months available, can't provide %d"
                % (len(aMonths), maxTestMonths)
            )
    # Return a list of lists of filenames
    result = []
    for key in aMonths:
        result.append(fNames[key])
    return result

def getFileNames(a_path,vars):
    # simplified listing of file names
    # pathtodata = ("%s/%s/norm_tensors") % ( a_path, vars[0],)
    # flist1 = os.listdir(pathtodata)
    # ncols=len(vars)
    # # result = [vars for _ in flist1]
    # for var in vars:
    #     pathtodata = ("%s/%s/norm_tensors") % ( a_path, var,)
    #     flist1 = os.listdir(pathtodata)
    #     flist1.sort()
    #     result.append(flist1)

    # newreult=[[subresult[0] for subresult in result]]
    # newreult.append([subresult[1] for subresult in result])

    # my_2d_list = [flist1[i:i + ncols] for i in range(0, len(flist1), ncols)]
    # my_2d_list = [flist2[i:i + 1] for i in range(0, len(flist2), 1)]
    # my_2d_list = [my_2d_list[i:i + 1] for i in range(0, len(flist2), 1)]

    # , ['psl', 'tas', 'uas', 'vas'], ['psl', 'tas', 'uas', 'vas']]

    # pathtodata = ("%s/%s/norm_tensors") % ( a_path, vars[0],)
    # l0 = os.listdir(pathtodata)
    # l0.sort()
    # pathtodata = ("%s/%s/norm_tensors") % ( a_path, vars[1],)
    # l1 = os.listdir(pathtodata)
    # l1.sort()
    # pathtodata = ("%s/%s/norm_tensors") % ( a_path, vars[2],)
    # l2 = os.listdir(pathtodata)
    # l2.sort()
    # pathtodata = ("%s/%s/norm_tensors") % ( a_path, vars[3],)
    # l3 = os.listdir(pathtodata)
    # l3.sort()
    # # result = [vars for _ in l0]
    # result = []
    # for i, _ in enumerate(l0):
    #     aa = [l0[i],l1[i],l2[i],l3[i]]
    #     result.insert(i,aa)

    allfn = []
    for i, var in enumerate(vars):
        pathtodata = ("%s/%s/norm_tensors/") % ( a_path, var,)
        flist1 = os.listdir(pathtodata)
        flist1.sort()
        flist1 = [pathtodata + s for s in flist1]
        allfn.insert(i,flist1)

    result = []
    for i, _ in enumerate(flist1):
        aa = [allfn[0][i],allfn[1][i],allfn[2][i],allfn[3][i]]
        result.insert(i,aa)

    return result

# Get a dataset
def getDataset(specification, purpose):
    # Get a list of filename sets
    # inFiles = getFileNamesOld(
        #     specification["inputTensors"],
        #     purpose,
        #     specification["startYear"],
        #     specification["endYear"],
        #     specification["testSplit"],
        #     specification["maxTrainingMonths"],
        #     specification["maxTestMonths"],
        #     specification["correlatedEnsembles"],
        #     specification["maxEnsembleCombinations"],
    # )
    vars=['tas','psl','uas','vas']
    # inFiles = getFileNames("/scratch/hadsx/cpm/5km/daily/1day", vars)
    inFiles = getFileNames(os.getenv("MLSCRATCH"), vars)

    # Create TensorFlow Dataset object from the source file names
    tnIData = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Create Dataset from the source file contents
    tsIData = tnIData.map(load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if (
        specification["outputTensors"] is not None
    ):  # I.e. input and output are not the same
        # outFiles = getFileNames(
        #     specification["outputTensors"],
        #     purpose,
        #     specification["startYear"],
        #     specification["endYear"],
        #     specification["testSplit"],
        #     specification["maxTrainingMonths"],
        #     specification["maxTestMonths"],
        #     specification["correlatedEnsembles"],
        #     specification["maxEnsembleCombinations"],
        # )
        # outFiles = getFileNames("/scratch/hadsx/cpm/5km/daily/1day", vars)
        outFiles = getFileNames(os.getenv("MLSCRATCH"), vars)

        tnOData = tf.data.Dataset.from_tensor_slices(tf.constant(outFiles))
        tsOData = tnOData.map(
            load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Zip the data together with the filenames (so we can find the date and source of each
    #   data tensor if we need it).
    if specification["outputTensors"] is not None:
        tz_data = tf.data.Dataset.zip((tnIData, tsIData, tsOData))
    else:
        tz_data = tf.data.Dataset.zip((tnIData, tsIData))

    # Optimisation
    if (purpose == "Train" and specification["trainCache"]) or (
        purpose == "Test" and specification["testCache"]
    ):
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it

    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data
