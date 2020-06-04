#-*- coding: UTF-8 -*-

import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indata', type=str, default='', help='input data')
    parser.add_argument('--outdir', type=str, default='', help='output folder')

    return parser.parse_args()

eval_opt = parse_arguments()
filepath = eval_opt.indata
outpath = eval_opt.outdir

threshold = 0.5

f_PointCloud = open(os.path.dirname(filepath) + '/' + os.path.basename(filepath), 'r')

try:
    if not os.path.exists(outpath):
        os.makedirs(outpath)
except OSError:
    print("Error: Creating output directory..\n" + outpath)

fD_Out_Results = open(outpath + '/' + os.path.basename(filepath).split('.')[0] + '_' + "OutRemoved_" + str(int(threshold * 100)) + "_D.xyz", 'w')
fB_Out_Results = open(outpath + '/' + os.path.basename(filepath).split('.')[0] + '_' + "OutRemoved_" + str(int(threshold * 100)) + "_B.xyz", 'w')
Del_Out_Results = open(outpath + '/' + os.path.basename(filepath).split('.')[0] + '_' + "OutRemoved_" + str(int(threshold * 100)) + "_R.xyz", 'w')

outlierCount = 0
while True:
    pointXYZ_PC = f_PointCloud.readline().rstrip('\n').split(' ')
    if not pointXYZ_PC or len(pointXYZ_PC) != 5:
        break

    if float(pointXYZ_PC[4]) < threshold:
        fD_Out_Results.write(pointXYZ_PC[0] + ' ' + pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + '\n')
        fB_Out_Results.write(pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + ' ' + pointXYZ_PC[0] + '\n')

    else:
        Del_Out_Results.write(pointXYZ_PC[0] + ' ' + pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + '\n')
        outlierCount += 1


print("outlierCount = %d" % outlierCount)

f_PointCloud.close()
fD_Out_Results.close()
fB_Out_Results.close()
Del_Out_Results.close()
