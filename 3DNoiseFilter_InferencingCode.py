# -*- coding: UTF-8 -*-

import os
import argparse
import time
import torch
import laspy
import multiprocessing
from glob import glob
from functools import partial
import numpy as np


# my_q = PQueue()

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indata', type=str, default='', required=True, help='input data')
    parser.add_argument('--outdir', type=str, default='', required=True, help='output folder')
    parser.add_argument('--workers', type=int, default=1, help='number of procedure workers')
    parser.add_argument('--skipsplit', type=int, default=0, help='put non-zero integer if skip split procedure')
    parser.add_argument('--model', type=str, default='', required=True, help='names of trained models, can evaluate multiple models')
    parser.add_argument('--passfirst', help='This is fast test', action="store_true")
    parser.add_argument('--passsecond', help='This is fast test', action="store_true")
    parser.add_argument('--passthird', help='This is fast test', action="store_true")
    parser.add_argument('--Deep', help='Deep filter', action="store_true")
    parser.add_argument('--BL', help='BL filter', action="store_true")
    parser.add_argument('--second_sparse_patches', type=int, default=0, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--second_sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')	
    parser.add_argument('--second_patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')                        
    parser.add_argument('--second_workers', type=int, default=2, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--third_n_neighbours', type=int, default=100, help='nearest neighbour used for inflation step')
    parser.add_argument('--third_sparse_patches', type=int, default=0, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--third_sampling', type=str, default='full',
                        help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--third_patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')    
    parser.add_argument('--third_workers', type=int, default=2, help='number of data loading workers - 0 means same thread as main execution')    
    parser.add_argument('--third_r', type=float, default=0.5, help='(recommended) neighborhood radius will be translated into a distance weight.')
    parser.add_argument('--third_n', type=float, default=0.5, help='(recommended) normal neighborhood radius will be translated into a normal distance weight.')
    parser.add_argument('--third_N', type=int, default=1, help='Number of filter iterations (default: 1).')
    parser.add_argument('--third_p', help='(recommended) will perform the computation in parallel.', action="store_true")
                     
    return parser.parse_args()


def Dnoise_outliers_removal(outpath, datapath):
    eval_opt = parse_arguments()
    passsecond = eval_opt.passsecond
    workers = eval_opt.second_workers
    sampling = eval_opt.second_sampling
    model = '%s/'%(eval_opt.model)
    skipsplit = eval_opt.skipsplit
    sparse_patches = eval_opt.second_sparse_patches
    patches_per_shape = eval_opt.second_patches_per_shape
    gpu_id = datapath[0].get()
    inputfilename = os.path.basename(eval_opt.indata).split('.')[0]
    
    outliersinput = os.path.join(outpath,inputfilename+"_splited/")
    file_validation = open(outliersinput+inputfilename+"_validationset_" + str(os.getpid()) + ".txt", 'w')
    file_validation.write(os.path.basename(datapath[1]).split('.')[0] + '\n')
    file_validation.close()
    
    outliersoutput = os.path.join(outpath,inputfilename+"_outremoved/")

    exe = ""
    pcrexe = []
    pcrexe_start = 'python OR.py'
    pcrexe_indir = ' --indir '
    pcrexe_outdir = ' --outdir '
    pcrexe_dataset = ' --dataset '
    pcrexe_model = ' --model '
    pcrexe_sparsepatches = ' --sparse_patches '
    pcrexe_sampling = ' --sampling '
    pcrexe_patchespershape = ' --patches_per_shape '

    pcrexe_workers = ' --workers '
    pcrexe_gpuid = ' --GPU_ID '
    
    pcrexe.append(pcrexe_start)
    pcrexe.append(pcrexe_indir)
    pcrexe.append(outliersinput)
    pcrexe.append(pcrexe_outdir)
    pcrexe.append(outliersoutput)
    pcrexe.append(pcrexe_dataset)
    pcrexe.append(inputfilename+"_validationset_" + str(os.getpid()) + ".txt")
    pcrexe.append(pcrexe_model)
    pcrexe.append(model)
    """
    if skipsplit == 0: pcrexe.append('10')
    else: pcrexe.append('100')
    """
    pcrexe.append(pcrexe_sparsepatches)
    pcrexe.append(str(sparse_patches))
    pcrexe.append(pcrexe_sampling)
    pcrexe.append(sampling)
    pcrexe.append(pcrexe_patchespershape)
    pcrexe.append(str(patches_per_shape))

    pcrexe.append(pcrexe_workers)
    pcrexe.append(str(workers))
    pcrexe.append(pcrexe_gpuid)
    pcrexe.append(str(gpu_id))
    
    for i in pcrexe : exe += i

    if passsecond:
        print('outliers_removal {} PASS'.format(os.path.basename(datapath[1])))
        pass
    else:
        print("outliers_removal {} start".format(os.path.basename(datapath[1])))
        try:
            os.system(exe)
        except OSError:
            print("outliers_removal error : Please set the correct setting value.")
        print("outliers_removal {} done".format(os.path.basename(datapath[1])))

    datapath[0].put(gpu_id)


def Dnoise_Denoiser(outpath, datapath):
    eval_opt = parse_arguments()
    passthird = eval_opt.passthird
    workers = eval_opt.third_workers
    sampling = eval_opt.third_sampling
    model = '%s/'%(eval_opt.model)
    skipsplit = eval_opt.skipsplit
    sparse_patches = eval_opt.third_sparse_patches
    patches_per_shape = eval_opt.third_patches_per_shape
    n_neighbours = eval_opt.third_n_neighbours
    inputfilename = os.path.basename(eval_opt.indata).split('.')[0]
    gpu_id = datapath[0].get()
    Dnoiseinput = os.path.join(outpath,inputfilename+"_outremoved/")    
    Dnoiseoutput = os.path.join(outpath,inputfilename+"_Deep_denoised/")  
    
    exe = ""
    pdexe = []
    pdexe_start = 'python CNR.py'
    pdexe_indir = ' --indir '
    pdexe_outdir = ' --outdir '
    pdexe_model = ' --model '
    pdexe_n_neighbours = ' --n_neighbours '
    pdexe_sparsepatches = ' --sparse_patches '
    pdexe_sampling = ' --sampling '
    pdexe_patchespershape = ' --patches_per_shape '
    pdexe_workers = ' --workers '
    pdexe_nrun = ' --nrun '
    pdexe_shapename = ' --shapename '
    pdexe_gpuid = ' --GPU_ID '
    
    pdexe.append(pdexe_start)
    pdexe.append(pdexe_indir)
    pdexe.append(Dnoiseinput)
    pdexe.append(pdexe_outdir)
    pdexe.append(Dnoiseoutput)

    pdexe.append(pdexe_model)
    pdexe.append(model)
    """
    if skipsplit == 0: pdexe.append('10')
    else : pdexe.append('100')
    """
    pdexe.append(pdexe_n_neighbours)
    pdexe.append(str(n_neighbours))
    pdexe.append(pdexe_sparsepatches)
    pdexe.append(str(sparse_patches))
    pdexe.append(pdexe_sampling)
    pdexe.append(sampling)
    pdexe.append(pdexe_patchespershape)
    pdexe.append(str(patches_per_shape))
    
    pdexe.append(pdexe_workers)
    pdexe.append(str(workers))
    pdexe.append(pdexe_nrun)
    pdexe.append(str(datapath[1][0]))
    pdexe.append(pdexe_shapename)
    pdexe.append(os.path.basename(datapath[1][1]).split('.')[0] + "_{i}")
    pdexe.append(pdexe_gpuid)
    pdexe.append(str(gpu_id))

    for i in pdexe : exe += i

    if passthird:
        print("Deep_Denoiser PASS")
        pass
    else:
        print("Deep_Denoiser {} start".format(str(datapath[1][0])))
        try:
            os.system(exe)
        except OSError:
            print("Deep_Denoiser error : Please set the correct setting value.")
        print("Deep_Denoiser {} done".format(str(datapath[1][0])))
    datapath[0].put(gpu_id)

def Bilateral_denoiser(outpath, datapath):    
    eval_opt = parse_arguments()
    exe = ""
    bdexe = []
    bdexe_start = "BL.exe"
    bdexe_indata = " " + datapath[1]
    bdexe_outdata = " " + outpath + os.path.basename(datapath[1]).split(".")[0] + "_bilat.xyz"
    bdexe_r = " -r "
    bdexe_n = " -n "
    bdexe_p = " -p "
    bdexe_N = " -N "
    
    bdexe.append(bdexe_start)
    bdexe.append(bdexe_indata)
    bdexe.append(bdexe_outdata)
    bdexe.append(bdexe_r)
    bdexe.append(str(eval_opt.third_r))
    bdexe.append(bdexe_n)
    bdexe.append(str(eval_opt.third_n))
    if eval_opt.third_p : bdexe.append(str(bdexe_p))
    bdexe.append(bdexe_N)
    bdexe.append(str(eval_opt.third_N))   
    
    for i in bdexe : exe += i
    if eval_opt.passthird:
        print("BL_denoiser PASS")
        pass
    else:
        print("BL_denoiser ",os.path.basename(datapath[1]).split(".")[0]," start")
        try:
            os.system(exe)
        except OSError:
            print("BL_denoiser error : Please set the correct setting value.")
        print("BL_denoiser ",os.path.basename(datapath[1]).split(".")[0]," done")
            
    exe = ""
    bdexe = []

if __name__ == '__main__':
    multiprocessing.freeze_support()#multiprocessing을 사용하는 프로그램이 고정되어(frozen) 윈도우 실행 파일을 생성할 때를 위한 지원을 추가.

    eval_opt = parse_arguments()

    filepath = eval_opt.indata
    outpath = eval_opt.outdir
    num_of_cpus = eval_opt.workers
    skipsplit = eval_opt.skipsplit
    model = eval_opt.model

    passfirst = eval_opt.passfirst  
    passsecond = eval_opt.passsecond    
    passthird = eval_opt.passthird
    inputfilename = os.path.basename(eval_opt.indata).split('.')[0]

    #### log start
    logfilename = "%s_Integrated_log.txt"%(filepath.split(".")[0])
    logfile = open(logfilename, mode='at')
    now = time.localtime()
    logDate = "%04d-%02d-%02d %02d:%02d:%02d" % (
    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    logfile.write("\n%s\n" % logDate)
    logfile.write("filepath: %s\n" % filepath)
    logfile.write("outpath: %s\n" % outpath)
    logfile.write("workers: %d\n" % num_of_cpus)
    logfile.write("skipsplit: %d\n" % skipsplit)
    logfile.write("model: %s\n" % model)
    logfile.write("passfirst: %s\n" % str(passfirst))
    logfile.write("passsecond: %s\n" % str(passsecond))
    logfile.write("passthird: %s\n" % str(passthird))
    
    logfile.write("second_sparse_patches: %s\n" % str(eval_opt.second_sparse_patches))
    logfile.write("second_sampling: %s\n" % eval_opt.second_sampling)
    logfile.write("second_workers: %d\n" % eval_opt.second_workers)
    if eval_opt.Deep:
        logfile.write("third_n_neighbours: %d\n" % eval_opt.third_n_neighbours)
        logfile.write("third_sparse_patches: %s\n" % str(eval_opt.third_sparse_patches))
        logfile.write("third_sampling: %s\n" % eval_opt.third_sampling)
        logfile.write("third_patches_per_shape: %d\n" % eval_opt.third_patches_per_shape)
        logfile.write("third_workers: %d\n\n" % eval_opt.third_workers)
    if eval_opt.BL:
        logfile.write("third_r: %2.f\n" % eval_opt.third_r)
        logfile.write("third_n: %2.f\n" % eval_opt.third_n)
        logfile.write("third_N: %d\n" % eval_opt.third_N)
        logfile.write("third_p: %s\n" % eval_opt.third_p)
 
    startTime = time.time()
    logfile.close()

    #### las to xyz convert & split procedure start
    outliersinput = os.path.join(outpath,inputfilename+"_splited/")
    procedureStart = time.time()
    if passfirst:
        print("las to xyz convert & split procedure PASS")
        pass
    else:
        try : 
            print("las to xyz convert & split procedure start")
            os.system("python PASM.py" +
                      " --indata " + filepath + " --outdir " + outliersinput +
                      " --workers " + str(num_of_cpus) +
                      " --skipsplit " + str(skipsplit))
            print("las to xyz convert & split procedure done")
        except OSError:
            print("las to xyz convert & split procedure error : Please set the correct setting value.")
    procedureDuration = divmod(time.time() - procedureStart, 60)
    logfile = open(logfilename, mode='at')
    logfile.write("LAS file split duration: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
    logfile.close()


    #### Dnoise_outliers_removal start
    procedureStart = time.time()
    xyzList = glob(outliersinput + "*.xyz")
    currentPath = os.getcwd()
    os.chdir(currentPath + "/outliers_removal")

    m = multiprocessing.Manager()
    my_q = m.Queue()

    for gpu_id in range(0, torch.cuda.device_count()):
        my_q.put(gpu_id)

    func_Dnoise_OR = partial(Dnoise_outliers_removal, outpath)
    pool = multiprocessing.Pool(processes=torch.cuda.device_count())
    pool.map(func_Dnoise_OR, [(my_q, elmt) for elmt in xyzList])

    pool.close()
    pool.join()

    os.chdir(currentPath)
    procedureDuration = divmod(time.time() - procedureStart, 60)
    logfile = open(logfilename, mode='at')
    logfile.write("Outlier_removal duration: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
    logfile.close()

    #### info to xyz convert start with default threshold 50
    procedureStart = time.time()
    Dnoiseinput = os.path.join(outpath,inputfilename+"_outremoved/")
    infoList = glob(Dnoiseinput + model + "/" +"*.info")

    for datapath in infoList:
        if passsecond:
            print("info to xyz convert start with default threshold 50 PASS")
            pass
        else:
            print("info to xyz convert start with default threshold 50 start")
            try:
                os.system("python ORT.py " +
                          " --indata " + datapath +
                          " --outdir " + Dnoiseinput)
            except OSError:
                print("info to xyz convert start with default threshold 50 error : Please set the correct setting value.")
            print("info to xyz convert start with default threshold 50 done")
	
    infoList = glob(Dnoiseinput + "*_R.xyz")
    if os.path.basename(filepath).split('.')[1] == 'las':
        originLAS = laspy.file.File(filepath, mode="r")	
    else:
        del_XYZ = open(outpath + "/Del_" + os.path.basename(filepath).split('.')[0] + '.xyz','w')
    good_indices = []
    for targetPath in infoList:
        print("convert " + os.path.basename(targetPath))
        targetFile = open(targetPath, 'r')
        target_indices = []
        while True:
            if os.path.basename(filepath).split('.')[1] == 'las':
                pointID_XYZ = targetFile.readline().rstrip('\n')
                if not pointID_XYZ:
                    break
                split_ID_XYZ = pointID_XYZ.split(' ')
                target_indices.append(int(float(split_ID_XYZ[0])))  
            else :
                pointXYZ_PC = targetFile.readline().rstrip('\n').split(' ')  
                try:                
                    del_XYZ.write(pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + ' ' + pointXYZ_PC[0] + '\n')    
                except IndexError:
                    break                
        if os.path.basename(filepath).split('.')[1] == 'las':          
            good_indices.extend(target_indices)
    if os.path.basename(filepath).split('.')[1] == 'las':
        del_LAS = laspy.file.File(outpath + "/Del_" + os.path.basename(filepath), mode='w', header=originLAS.header)
        del_LAS.points = originLAS.points[good_indices]
        del_LAS.close()
    else:
        del_XYZ.close()

    
	
    procedureDuration = divmod(time.time() - procedureStart, 60)
    logfile = open(logfilename, mode='at')
    logfile.write("Info file convert duration: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
    logfile.close()

    procedureStart = time.time()
    Dnoiseoutput = os.path.join(outpath, inputfilename + "_Deep_denoised/")
    Bilateraloutput = os.path.join(outpath, inputfilename + "_BL_denoised/") 
    
    if eval_opt.Deep and not eval_opt.BL:
        outremovedList = glob(Dnoiseinput + "*_D.xyz")
    elif not eval_opt.Deep and eval_opt.BL:
        outremovedList = glob(Dnoiseinput + "*_B.xyz")
    elif not eval_opt.Deep and not eval_opt.BL and passthird:
        outremovedList = glob(Dnoiseinput + "*_D.xyz")
    else : raise Exception('Please choose either "Deep" or "BL."')

    if eval_opt.Deep and not eval_opt.BL:
        #### Dnoise_denoiser start
        try:
            if not os.path.exists(Dnoiseoutput):
                os.makedirs(Dnoiseoutput)
        except OSError:
            print("Error: Creating output directory..\n" + Dnoiseoutput)
                 
        currentPath = os.getcwd()
        os.chdir(currentPath + "/noise_removal")

        for nrun in range(1, 4):

            nrun_outremovedList = []
            for datapath in outremovedList:
                nrun_outremovedList.append([nrun, datapath])

            func_Dnoise_DN = partial(Dnoise_Denoiser, outpath)
            pool = multiprocessing.Pool(processes=torch.cuda.device_count())
            pool.map(func_Dnoise_DN, [(my_q, elmt) for elmt in nrun_outremovedList])

            pool.close()
            pool.join()

        os.chdir(currentPath)
        procedureDuration = divmod(time.time() - procedureStart, 60)
        logfile = open(logfilename, mode='at')
        logfile.write("Denoise duration: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
        logfile.close()
        
    elif not eval_opt.Deep and eval_opt.BL:
        #### Bilateral_denoiser start
        try:
            if not os.path.exists(Bilateraloutput):
                os.makedirs(Bilateraloutput)
        except OSError:
            print("Error: Creating output directory..\n" + Bilateraloutput)
            
        m = multiprocessing.Manager()
        my_q = m.Queue()

        func_Bilateral_D = partial(Bilateral_denoiser, Bilateraloutput)
        pool = multiprocessing.Pool(processes=num_of_cpus)
        pool.map(func_Bilateral_D, [(my_q, elmt) for elmt in outremovedList])

        pool.close()
        pool.join()
        
        procedureDuration = divmod(time.time() - procedureStart, 60)
        logfile = open(logfilename, mode='at')
        logfile.write("Bilateral_denoiser: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
        logfile.close()
    elif not eval_opt.Deep and not eval_opt.BL and passthird: pass
    else : raise Exception('Please choose either "Deep" or "BL."')

    #### attach & convert xyz to las
    procedureStart = time.time()
    
    if eval_opt.Deep and not eval_opt.BL : infoList = glob(Dnoiseoutput +"*_3.xyz")
    if not eval_opt.Deep and eval_opt.BL : infoList = glob(Bilateraloutput + "*_bilat.xyz")
    if not eval_opt.Deep and not eval_opt.BL and passthird : infoList = glob(Dnoiseinput + "*_D.xyz")
    
    if os.path.basename(filepath).split('.')[1] == 'las':
        originLAS = laspy.file.File(filepath, mode="r")	
    else:
        if eval_opt.Deep and not eval_opt.BL : result_XYZ = open(outpath + "/Deep_filtered_" + os.path.basename(filepath),'w')
        if not eval_opt.Deep and eval_opt.BL : result_XYZ = open(outpath + "/BL_filtered_" + os.path.basename(filepath),'w')
        if not eval_opt.Deep and not eval_opt.BL and passthird : result_XYZ = open(outpath + "/OR_filtered_" + os.path.basename(filepath),'w')
        #originLAS = np.loadtxt(filepath)	
    good_indices = []
    blpointscount = 1
    for targetPath in infoList:
        print("convert " + os.path.basename(targetPath))
        targetFile = open(targetPath, 'r')
        target_indices = []
        while True:
            if os.path.basename(filepath).split('.')[1] == 'las':
                pointID_XYZ = targetFile.readline().rstrip('\n')
                if not pointID_XYZ:
                    break
                if eval_opt.Deep and not eval_opt.BL:
                    split_ID_XYZ = pointID_XYZ.split(' ')
                    target_indices.append(int(float(split_ID_XYZ[0])))
                elif not eval_opt.Deep and eval_opt.BL:
                    split_ID_XYZ = pointID_XYZ.split('\t')[:4]
                    for i in range(len(split_ID_XYZ)):
                        if split_ID_XYZ[i] == '-nan(ind)':
                            print('Deletes the neighboring point because it cannot be found in the filter.({} thing)'.format(blpointscount))
                            blpointscount += 1
                            pass
                        else:
                            split_ID_XYZ[i] = float(split_ID_XYZ[i])
                    target_indices.append(int(float(split_ID_XYZ[-1])))
                elif not eval_opt.Deep and not eval_opt.BL and passthird:
                    split_ID_XYZ = pointID_XYZ.split(' ')
                    target_indices.append(int(float(split_ID_XYZ[0])))    
            else:
                if not eval_opt.Deep and eval_opt.BL :
                    pointXYZ_PC = targetFile.readline().rstrip('\n').split('\t')
                else:
                    pointXYZ_PC = targetFile.readline().rstrip('\n').split(' ')                
                #print(pointXYZ_PC[:1])
                try:       
                    if eval_opt.Deep and not eval_opt.BL : result_XYZ.write(pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + ' ' + pointXYZ_PC[0] + '\n') 
                    if not eval_opt.Deep and eval_opt.BL : result_XYZ.write(pointXYZ_PC[0] + ' ' + pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + '\n')                    
                    if not eval_opt.Deep and not eval_opt.BL and passthird : result_XYZ.write(pointXYZ_PC[1] + ' ' + pointXYZ_PC[2] + ' ' + pointXYZ_PC[3] + ' ' + pointXYZ_PC[0] + '\n') 
                except IndexError:
                    break    
        if os.path.basename(filepath).split('.')[1] == 'las':                    
            good_indices.extend(target_indices)

    print("combine LAS files..")
    if os.path.basename(filepath).split('.')[1] == 'las':
        if eval_opt.Deep and not eval_opt.BL : output_LAS = laspy.file.File(outpath + "/Deep_filtered_" + os.path.basename(filepath), mode='w', header=originLAS.header)
        if not eval_opt.Deep and eval_opt.BL : output_LAS = laspy.file.File(outpath + "/BL_filtered_" + os.path.basename(filepath), mode='w', header=originLAS.header)
        if not eval_opt.Deep and not eval_opt.BL and passthird : output_LAS = laspy.file.File(outpath + "/OR_filtered_" + os.path.basename(filepath), mode='w', header=originLAS.header)
        output_LAS.points = originLAS.points[good_indices]
        output_LAS.close()
    else:
        result_XYZ.close()
        
    procedureDuration = divmod(time.time() - procedureStart, 60)
    logfile = open(logfilename, mode='at')
    logfile.write("xyz to LAS converting duration: %d min %d sec\n" % (procedureDuration[0], procedureDuration[1]))
    logfile.close()

    totalTime = time.time() - startTime
    totalTimeSplit = divmod(totalTime, 60)
    print("##################################################")
    print("Total Processing Time : %d min %d sec" % (totalTimeSplit[0], totalTimeSplit[1]))
    print("##################################################")

    logfile = open(logfilename, mode='at')
    logfile.write("Total Processing Time : %d min %d sec\n" % (totalTimeSplit[0], totalTimeSplit[1]))
    logfile.write("Process End\n")
    logfile.close()