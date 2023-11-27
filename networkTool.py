'''
Author: fuchy@stu.pku.edu.cn
LastEditors: fcy
Description: Network parameters and helper functions
FilePath: /compression/networkTool.py
'''
from glob import glob
import torch
import os, random
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_NUM = 1
MAX_NUM_POINTS = 8192 * 4
LOD_MAX = 2  #TMC start lod idx
GROUP = 1
INTRA_LOD_START = 2
MAX_SLICE_NUM = 16
MAX_SLICE_NUM1 = 2
MAX_BATCH = 48
QLEVEL = 16
ATTIBUTE_RES_RANGE = 511
ATTIBUTE_HALF_RANGE = 255
MAX_DIS = 2**(40-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('use', device)
os.environ["H5PY_DEFAULT_READONLY"] = "1"
# Network parameters
bptt = 1024  # Context window length
KNN = 9
EXPNAME = 'Exp/sparse/qint_fastLoD-KD'
checkpointPath = EXPNAME + '/checkpoint'
levelNumK = 0
DataRoot = "/home/fuchy/workspace/data/Atri"
trainDataRoot = DataRoot + "/*/*.mat"  # DON'T FORGET RUN ImageFolder.calcdataLenPerFile() FIRST
MAX_OCTREE_LEVEL = 16
expComment = 'Trained on MPEG 8i,MVUB, Thaidancer_viewdep,Sketchfab 1~15 level.Add norm'
# Random seed
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Tool functions
def save(index,
         saveDict,
         modelDir='checkpoint',
         pthType='epoch',
         keep_pth_num=10):
    savePath = modelDir + '/encoder_{}_{:08d}.pth'
    if os.path.dirname(savePath) != '' and not os.path.exists(
            os.path.dirname(savePath)):
        os.makedirs(os.path.dirname(savePath))
    torch.save(saveDict,
               modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, index))
    dir_list = glob(modelDir + '/*.pth')
    if len(dir_list) > keep_pth_num:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(x))
        for i in dir_list[:-keep_pth_num]:
            os.remove(i)


def reload(checkpoint,
           modelDir='checkpoint',
           pthType='epoch',
           print=print,
           multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(
                modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, checkpoint),
                map_location=device)
            pth = modelDir + '/encoder_{}_{:08d}.pth'.format(
                pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir, map_location=device)
            pth = modelDir
        saveDict['path'] = pth
        # print('load: ',pth)
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            new_state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        # print('load: ',pth)
        return saveDict
    except Exception as e:
        print('**warning**', e, ' start from initial model')
        # saveDict['path'] = e
    return None


class CPrintl():

    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '' and not os.path.exists(
                os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))

    def __call__(self, *args, end='\n'):
        print(*args, end=end)
        print(*args, file=open(self.log_file, 'a'), end=end)


def model_structure(model, print=print):
    print('-' * 120)
    print('|'+' '*30+'weight name'+' '*31+'|' \
            +' '*10+'weight shape'+' '*10+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-' * 120)
    num_para = 0
    for _, (key, w_variable) in enumerate(model.named_parameters()):
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape),
                                                    each_para))
    print('-' * 120)
    print('The total number of parameters: ' + str(num_para))
    print('-' * 120)
