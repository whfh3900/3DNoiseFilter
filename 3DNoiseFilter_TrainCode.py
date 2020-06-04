
import sys
import os
from glob import glob
import torch
import pandas as pd
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--train', type=str, default='', required=True, help='[o,d]')
	
    # naming / file handling	
    parser.add_argument('--makefile', type=bool, default=True, help='training run name')
    parser.add_argument('--name', type=str, default='TestOutlier', required=True, help='training run name')
    parser.add_argument('--desc', type=str, default='', help='description')
    parser.add_argument('--indir', type=str, default='', required=True, help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='', required=True, help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    
    parser.add_argument('--O_trainset', type=str, default='TrainlistO.txt', help='training set file name')
    parser.add_argument('--O_testset', type=str, default='VallistO.txt', help='test set file name')
    parser.add_argument('--D_trainset', type=str, default='TrainlistD.txt', help='training set file name')
    parser.add_argument('--D_testset', type=str, default='VallistD.txt', help='test set file name')
    
    parser.add_argument('--O_saveinterval', type=int, default=500, help='save model each n epochs')
    parser.add_argument('--D_saveinterval', type=int, default=200, help='save model each n epochs')

    parser.add_argument('--refine', type=str, default='None', help='refine model at this path')

    # training parameters
    parser.add_argument('--O_nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--D_nepoch', type=int, default=2000, help='number of epochs to train for')

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n' 'point: center point\n' 'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=100, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=600, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n''random: fully random over the entire dataset (the set of all patches is permuted)\n''random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=bool, default=False, help='use same patches in each epoch, mainly for debugging')
    
    parser.add_argument('--O_lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--D_lr', type=float, default=1e-10, help='learning rate')

    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    # model hyperparameters
    parser.add_argument('--O_outputs', type=str, nargs='+', default=['outliers'], help='outputs of the network')
    parser.add_argument('--D_outputs', type=str, nargs='+', default=['clean_points'], help='outputs of the network')

    parser.add_argument('--use_point_stn', type=bool, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=bool, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')
    return parser.parse_args()
	

	
def Outlier_training(opt):

    makefile = opt.makefile
    name = opt.name
    desc = opt.desc
    indir = opt.indir
    outdir = opt.outdir
    logdir = opt.logdir
    trainset = opt.O_trainset
    testset = opt.O_testset
    saveinterval = opt.O_saveinterval
    refine = opt.refine
    nepoch = opt.O_nepoch
    batchSize = opt.batchSize
    patch_radius = opt.patch_radius
    patch_center = opt.patch_center
    patch_point_count_std = opt.patch_point_count_std
    patches_per_shape = opt.patches_per_shape
    workers = opt.workers
    cache_capacity = opt.cache_capacity
    seed = opt.seed
    training_order = opt.training_order
    identical_epochs = opt.identical_epochs
    lr = opt.O_lr
    momentum = opt.momentum
    use_pca = opt.use_pca
    outputs = opt.O_outputs
    use_point_stn = opt.use_point_stn
    use_feat_stn = opt.use_feat_stn
    sym_op = opt.sym_op
    point_tuple = opt.point_tuple
    points_per_patch = opt.points_per_patch
    
 
	## makeoutliersfile start ##
    original_list = []
    listdir = indir
    for dir1 in os.listdir(listdir):
        if dir1[-3:] == 'txt':pass
        else:
            dir1path = os.path.join(listdir,dir1)
            for dir2 in os.listdir(dir1path):
                gt1path = os.path.join(dir1path,dir2,"GT1")
                gt1pathlist = glob(gt1path+"/"+"*.xyz")
                for gt1 in gt1pathlist:
                    original_list.append(gt1)
    try:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    except OSError:
        print("Error: Creating output directory..\n" + outdir)
            
    if makefile:
        for oripath in tqdm(original_list):
                
            originalPath = open(oripath, 'r')
            rePath = ""
            datapath = os.path.dirname(oripath).split('\\')[:-1]
            datapath.append('GT2')
            datapath.append(os.path.basename(oripath))
            for i in datapath : rePath += i+'\\'
            rePath = rePath[:-1]
            gt2Path = open(rePath,'r')
                    
            outpath = oripath.split('.')[0] + ".outliers"
            if os.path.isfile(outpath) == False:
                outliersPath = open(outpath, 'w')
                
                oriData = pd.read_csv(originalPath, names=['x','y','z'],sep=' ')
                reData = pd.read_csv(gt2Path, names=['x','y','z'],sep=' ')

                oriData['x'] = oriData['x'].apply(str)
                oriData['y'] = oriData['y'].apply(str)
                oriData['z'] = oriData['z'].apply(str)

                reData['x'] = reData['x'].apply(str)
                reData['y'] = reData['y'].apply(str)
                reData['z'] = reData['z'].apply(str)

                oriData['id_name'] = oriData[['x', 'y', 'z']].apply(lambda x: '_'.join(x), axis=1)
                reData['id2_name'] = reData[['x', 'y', 'z']].apply(lambda x: '_'.join(x), axis=1)

                uniq = reData['id2_name'].unique()
                a = oriData['id_name'].isin(uniq)
                oriData['0/1'] = "1.000000"
                oriData.loc[a,'0/1'] = "0.000000"

                oriData['0/1'].to_csv(outliersPath, index=False, header=None, sep="\t")  
                                
                originalPath.close()  
                gt2Path.close()   
                outliersPath.close()                    
                
    currentPath = os.getcwd()
    os.chdir(currentPath)
    O_outdir = outdir + "/models_outliter"
                        
    moveexe = os.path.join(currentPath,"outliers_removal_triaining")
    os.chdir(moveexe)
    print("PCN_outliers_removal_trianing start")
			
    Oexe = "python train_pcpnet.py"
    Oexelist = []
    Oexelist.append(" --name ")
    Oexelist.append(name)
    Oexelist.append(" --indir ")
    Oexelist.append(listdir)
    Oexelist.append(" --outdir ")
    Oexelist.append(outdir)
    Oexelist.append(" --trainset ")
    Oexelist.append(trainset)
    Oexelist.append(" --testset ")
    Oexelist.append(testset)
    Oexelist.append(" --saveinterval ")
    Oexelist.append(str(saveinterval))
    if refine != 'None':
        Oexelist.append(" --refine ")
        Oexelist.append(refine)
    Oexelist.append(" --nepoch ")
    Oexelist.append(str(nepoch))
    Oexelist.append(" --batchSize ")
    Oexelist.append(str(batchSize))
    Oexelist.append(" --patch_radius ")
    Oexelist.append(str(patch_radius[0]))
    Oexelist.append(" --patch_center ")
    Oexelist.append(patch_center)
    Oexelist.append(" --patch_point_count_std ")
    Oexelist.append(str(patch_point_count_std))
    Oexelist.append(" --patches_per_shape ")
    Oexelist.append(str(patches_per_shape))
    Oexelist.append(" --workers ")
    Oexelist.append(str(workers))
    Oexelist.append(" --seed ")
    Oexelist.append(str(seed))
    Oexelist.append(" --training_order ")
    Oexelist.append(str(training_order))
    Oexelist.append(" --identical_epochs ")
    Oexelist.append(str(identical_epochs))
    Oexelist.append(" --use_point_stn ")
    Oexelist.append(str(use_point_stn))
    Oexelist.append(" --use_feat_stn ")
    Oexelist.append(str(use_feat_stn))
    Oexelist.append(" --sym_op ")
    Oexelist.append(sym_op)
    Oexelist.append(" --point_tuple ")
    Oexelist.append(str(point_tuple))
    Oexelist.append(" --points_per_patch ")
    Oexelist.append(str(points_per_patch))
    Oexelist.append(" --lr ")
    Oexelist.append(str(lr))
    Oexelist.append(" --momentum ")
    Oexelist.append(str(momentum))
    for i in Oexelist : Oexe += i
    os.system(Oexe)
    print("PCN_outliers_removal done")
    os.chdir(currentPath)
            
            
	# Denoiser def ###################################################################

def Denoise_training(opt):

    name = opt.name
    desc = opt.desc
    indir = opt.indir
    outdir = opt.outdir
    logdir = opt.logdir
    trainset = opt.D_trainset
    testset = opt.D_testset
    saveinterval = opt.D_saveinterval
    refine = opt.refine
    nepoch = opt.D_nepoch
    batchSize = opt.batchSize
    patch_radius = opt.patch_radius
    patch_center = opt.patch_center
    patch_point_count_std = opt.patch_point_count_std
    patches_per_shape = opt.patches_per_shape
    workers = opt.workers
    cache_capacity = opt.cache_capacity
    seed = opt.seed
    training_order = opt.training_order
    identical_epochs = opt.identical_epochs
    lr = opt.D_lr
    momentum = opt.momentum
    use_pca = opt.use_pca
    outputs = opt.D_outputs
    use_point_stn = opt.use_point_stn
    use_feat_stn = opt.use_feat_stn
    sym_op = opt.sym_op
    point_tuple = opt.point_tuple
    points_per_patch = opt.points_per_patch        

    currentPath = os.getcwd()
    os.chdir(currentPath)

    D_outdir = outdir + "/models_denoise"

    denoise_list = []
    listdir = indir
            
	#try:
    moveexe = os.path.join(currentPath,"noise_removal_training")
    os.chdir(moveexe)    
    print("PCN_denoiser_trianing start")
			
    Dexe = "python train_pcpnet.py"
    Dexelist = []
    Dexelist.append(" --name ")
    Dexelist.append(name)
    Dexelist.append(" --indir ")
    Dexelist.append(listdir)
    Dexelist.append(" --outdir ")
    Dexelist.append(outdir)
    Dexelist.append(" --trainset ")
    Dexelist.append(trainset)
    Dexelist.append(" --testset ")
    Dexelist.append(testset)
    Dexelist.append(" --saveinterval ")
    Dexelist.append(str(saveinterval))
    if refine != 'None':
        Dexelist.append(" --refine ")
        Dexelist.append(refine)
    Dexelist.append(" --nepoch ")
    Dexelist.append(str(nepoch))
    Dexelist.append(" --batchSize ")
    Dexelist.append(str(batchSize))
    Dexelist.append(" --patch_radius ")
    Dexelist.append(str(patch_radius[0]))
    Dexelist.append(" --patch_center ")
    Dexelist.append(str(patch_center))
    Dexelist.append(" --patch_point_count_std ")
    Dexelist.append(str(patch_point_count_std))
    Dexelist.append(" --patches_per_shape ")
    Dexelist.append(str(patches_per_shape))
    Dexelist.append(" --workers ")
    Dexelist.append(str(workers))
    Dexelist.append(" --seed ")
    Dexelist.append(str(seed))
    Dexelist.append(" --training_order ")
    Dexelist.append(training_order)
    Dexelist.append(" --identical_epochs ")
    Dexelist.append(str(identical_epochs))
    Dexelist.append(" --use_point_stn ")
    Dexelist.append(str(use_point_stn))
    Dexelist.append(" --use_feat_stn ")
    Dexelist.append(str(use_feat_stn))
    Dexelist.append(" --sym_op ")
    Dexelist.append(sym_op)
    Dexelist.append(" --point_tuple ")
    Dexelist.append(str(point_tuple))
    Dexelist.append(" --points_per_patch ")
    Dexelist.append(str(points_per_patch))
    Dexelist.append(" --lr ")
    Dexelist.append(str(lr))
    Dexelist.append(" --momentum ")
    Dexelist.append(str(momentum))
    for i in Dexelist : Dexe += i
    os.system(Dexe)
			
    print("PCN_denoiser done")
    os.chdir(currentPath)
            
if __name__ == "__main__":
    opt = parse_arguments()
    if opt.train == 'o':
        Outlier_training(opt)
    elif opt.train == 'd':
        Denoise_training(opt)
    else : print("No train Type")
  
