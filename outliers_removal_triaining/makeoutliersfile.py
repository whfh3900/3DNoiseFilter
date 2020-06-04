#-*- coding: UTF-8 -*-

import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--oridata', type=str, default='./data.xyz', help='original input data')
    parser.add_argument('--redata', type=str, default='./data.xyz', help='outliers removal input data')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder')

    return parser.parse_args()

eval_opt = parse_arguments()
oripath = eval_opt.oridata
repath = eval_opt.redata
outpath = eval_opt.outdir

original_pc = open(oripath, 'r')
removal_pc = open(repath, 'r')

try:
    if not os.path.exists(outpath):
        os.makedirs(outpath)
except OSError:
    print("Error: Creating output directory..\n" + outpath)
    
f_Out_Results = open(outpath + '/' + os.path.basename(oripath).split('.')[0] + ".outliers", 'w')



original_point = original_pc.read().split('\n')
if "" in original_point :
	original_point.remove("")

removal_point = removal_pc.read().split('\n')
if "" in removal_point :
	removal_point.remove("")


count = 0
reList = []

for reXYZ in removal_point :
        xyz = reXYZ.split(' ')
        xyz = list(map(float, xyz))
        reList.append(xyz)

print(reList)

for oriXYZ in original_point :
      xyzList = oriXYZ.split(' ')
      xyzList = xyzList[0:3]
      xyzList = list(map(float, xyzList))
      if xyzList in reList :
               f_Out_Results.write('0.000000' + '\n')

      else : 
               f_Out_Results.write('1.000000' + '\n')
    
      count += 1
      print("current/total: (%d / %d)" %(count, len(original_point)))

original_pc.close()
removal_pc.close()
f_Out_Results.close()

# f_Out_Results.close()