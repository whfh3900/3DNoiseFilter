# -*- coding: UTF-8 -*-

import numpy as np
import laspy
import os
import argparse
import multiprocessing
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indata', type=str, default='./data.las', help='input data')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of procedure workers')
    parser.add_argument('--skipsplit', type=int, default=0,
                        help='put non-zero integer if skip split procedure')

    return parser.parse_args()

def Segmentation_pointID(pointID_Segment, mid_IDSplitedList, small_IDSplitedList, large_subList):

    for mid_subList in mid_IDSplitedList:

        mask_2D = np.isin(large_subList, mid_subList)
        Intersection_2D = large_subList[mask_2D]
        for small_subList in small_IDSplitedList:

            mask_3D = np.isin(Intersection_2D, small_subList)
            Intersection_3D = Intersection_2D[mask_3D]

            pointID_Segment.put(Intersection_3D)
            print("Segment added - total: "),
            print(pointID_Segment.qsize())



if __name__ == '__main__':

    eval_opt = parse_arguments()
    filepath = eval_opt.indata
    outpath = eval_opt.outdir

    points_UpperLimit = 500000
    num_of_cpus = eval_opt.workers
    skipsplit = eval_opt.skipsplit

    print("las file extracting...")
    if os.path.basename(filepath).split('.')[1] == 'las':
        inFile = laspy.file.File(os.path.dirname(filepath) + '/' + os.path.basename(filepath))
    elif os.path.basename(filepath).split('.')[1] == 'xyz':
        inFile = np.loadtxt(filepath)
    else:
        print('Please xyz or las')
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        print("Error: Creating output directory..\n" + outpath)
    if os.path.basename(filepath).split('.')[1] == 'las':
        pointID = np.array(range(inFile.x.shape[0]), dtype=np.uint64)
        originData = np.vstack((pointID, inFile.x, inFile.y, inFile.z)).transpose()
    else :
        #pointID = inFile.transpose()[0]
        pointID = np.array(range(len(inFile)), dtype=np.uint64)
        print('PointID : ',pointID,'\n',pointID.shape)
        originData = np.vstack((pointID, inFile.transpose()[0], inFile.transpose()[1], inFile.transpose()[2])).transpose()
    print("done.\n")

    x_Length = originData[:, 1].max() - originData[:, 1].min()
    y_Length = originData[:, 2].max() - originData[:, 2].min()
    z_Length = originData[:, 3].max() - originData[:, 3].min()
    num_of_points = originData.shape[0]

    # x-axis based operate
    xy_Ratio = y_Length / x_Length
    xz_Ratio = z_Length / x_Length

    x_SplitNum = 1
    y_SplitNum = 1
    z_SplitNum = 1
    if skipsplit == 0 and num_of_points >= points_UpperLimit * 4: # False skipsplit
        while num_of_points > points_UpperLimit * xy_Ratio * xz_Ratio * (x_SplitNum ** 3):
            x_SplitNum += 1

        y_SplitNum = round(x_SplitNum * xy_Ratio) if x_SplitNum * xy_Ratio > 0.5 else 1
        z_SplitNum = round(x_SplitNum * xz_Ratio) if x_SplitNum * xz_Ratio > 0.5 else 1
    else:
        print("skip procedure skip..\n")

    print("Elements sorting...")
    sortX_pointID = originData[originData[:, 1].argsort(), 0]
    sortY_pointID = originData[originData[:, 2].argsort(), 0]
    sortZ_pointID = originData[originData[:, 3].argsort(), 0]
    print("done.\n")


    print("Axises splitting...")
    print("Total segments: %d" % (int(x_SplitNum * y_SplitNum * z_SplitNum)))
    splitIdx = 0
    x_IDSplitedList = []
    while splitIdx < sortX_pointID.shape[0] - 1:
        upperIdx = min([splitIdx + round(sortX_pointID.shape[0] / x_SplitNum), sortX_pointID.shape[0] - 1])
        x_IDSplitedList.append(sortX_pointID[int(splitIdx): int(upperIdx) + 1])
        splitIdx = upperIdx + 1

    splitIdx = 0
    y_IDSplitedList = []
    while splitIdx < sortY_pointID.shape[0] - 1:
        upperIdx = min([splitIdx + round(sortY_pointID.shape[0] / y_SplitNum), sortY_pointID.shape[0] - 1])
        y_IDSplitedList.append(sortY_pointID[int(splitIdx): int(upperIdx) + 1])
        splitIdx = upperIdx + 1

    splitIdx = 0
    z_IDSplitedList = []
    while splitIdx < sortZ_pointID.shape[0] - 1:
        upperIdx = min([splitIdx + round(sortZ_pointID.shape[0] / z_SplitNum), sortZ_pointID.shape[0] - 1])
        z_IDSplitedList.append(sortZ_pointID[int(splitIdx): int(upperIdx) + 1])
        splitIdx = upperIdx + 1

    sizesort_IDSplitedList = [x_IDSplitedList, y_IDSplitedList] if len(x_IDSplitedList) > len(y_IDSplitedList) else [y_IDSplitedList, x_IDSplitedList]
    sizesort_IDSplitedList = sizesort_IDSplitedList + [z_IDSplitedList] if len(sizesort_IDSplitedList[0]) > len(z_IDSplitedList) else [z_IDSplitedList] + sizesort_IDSplitedList

    m = multiprocessing.Manager()
    pointID_Segment = m.Queue()

    func_Seg_OR = partial(Segmentation_pointID, pointID_Segment, sizesort_IDSplitedList[1], sizesort_IDSplitedList[2])

    pool = multiprocessing.Pool(processes = num_of_cpus)
    pool.map(func_Seg_OR, [subList for subList in sizesort_IDSplitedList[0]])

    pool.close()
    pool.join()
    print("done.\n")
    print("Splited files saving...")
    xyz_TotalCount = 0
    for seg_Count in range(pointID_Segment.qsize()):
        pointID_seg = pointID_Segment.get()
        if pointID_seg.size == 0:
            continue

        data_seg = originData[pointID_seg.astype(int), :]

        np.savetxt(outpath + '/' + os.path.basename(filepath).split('.')[0] + '_' + str(seg_Count) + ".xyz",
                   data_seg)
        print(os.path.basename(filepath).split('.')[0] + '_' + str(seg_Count) + ".xyz is saved... // %d points" %(data_seg.shape[0]))
        xyz_TotalCount += data_seg.shape[0]

    print("xyz_TotalCount: %d" %(xyz_TotalCount))
    print("done.\n")
