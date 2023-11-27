import numpy as np
import testTool.pt as pointcloud
from LOD import pred_from_lod
from networkTool import EXPNAME, KNN, INTRA_LOD_START, ATTIBUTE_HALF_RANGE,MAX_DIS,device,MAX_SLICE_NUM, MAX_BATCH
import numpy as np
import os
from LOD import *
from copy import deepcopy
import time

from resAc.ac_warpper import encode_res, decode_res
import torch
import MinkowskiEngine as ME
import math
from scipy.sparse import csr_matrix,csc_matrix

class TPreprocess():

    def __init__(self) -> None:
        self.m_attributeType = "rgb"
        self.m_colorSpace = "YUV"
        self.m_quanParm = {'qs': 1, 'atq': 1, 'offset': 'min'}
        self.m_pointCloudIn = TComPointCloud()
        self.m_pointCloudOut = None

    def quantize(self, copy=True):
        if copy:
            self.m_pointCloudOut = deepcopy(self.m_pointCloudIn)
        else:
            self.m_pointCloudOut = self.m_pointCloudIn
        self.m_pointCloudOut.quantization(self.m_quanParm)

    def readOriPointCloud(self, filein):
        self.m_pointCloudIn.readFromFile(filein, self.m_attributeType)
        return self.m_pointCloudIn

    def preprocess(self, copy_ori_pt=False):
        self.quantize(copy=copy_ori_pt)
        if self.m_colorSpace == "YUV" and self.m_pointCloudOut.m_hasColors:
            self.m_pointCloudOut.convertRGBToYUV()
        return self.m_pointCloudOut


class TComPointCloud():

    def __init__(self) -> None:
        self.m_pos = np.empty((0, 3), float)
        self.m_color = np.empty((0, 3), int)
        self.m_numPoint = 0
        self.m_hasColors = False

    def sortByIdx(self, idx):
        self.m_pos = self.m_pos[idx]
        self.m_color = self.m_color[idx]

    def getByIdx(self, idx):
        return np.c_[self.m_pos[idx], self.m_color[idx]]

    def readFromFile(self, path, atriType):
        pt_data = pointcloud.pcread(path, [atriType])
        self.m_pos = pt_data[0]
        self.m_numPoint = self.m_pos.shape[0]
        assert self.m_pos.ndim == 2 and self.m_pos.shape[
            1] == 3 and self.m_numPoint > 0, 'Error read ' + path

        if atriType == 'rgb':
            self.m_color = pt_data[1]
            assert self.m_color.shape[0] == self.m_numPoint
            self.m_hasColors = True

    def saveToFile(self, path):
        if self.m_hasColors:
            pointcloud.pcwrite(path, self.m_pos, self.m_color)
        else:
            pointcloud.pcwrite(path, self.m_pos)
        print('save file ', path)

    def convertRGBToYUV(self):
        self.m_color = pointcloud.RGB2YCoCg(self.m_color[:, :3])

    def convertYUVToRGB(self):
        self.m_color = pointcloud.YCoCg2RGB(self.m_color[:, :3])

    def clearColors(self):
        self.m_color = np.zeros((self.m_numPoint, 3), int)
        self.m_hasColors = True

    def quantization(self, quan_parm):
        refPt = self.m_pos[:, 0:3]
        offset = quan_parm['offset']
        if offset is 'min':
            offset = refPt.min(0, keepdims=True)
        if offset is 'mean':
            offset = refPt.mean(0, keepdims=True)
        points = refPt - offset
        if 'qlevel' in quan_parm and quan_parm['qlevel'] is not None:
            quan_parm['qs'] = (points.max() -
                               points.min()) / (2**quan_parm['qlevel'] - 1)
        qpt = np.round(points / quan_parm['qs']).astype(int)
        quanpt, idx = np.unique(qpt, axis=0, return_index=True)
        self.m_pos = quanpt
        if self.m_hasColors:
            self.m_color = np.round(self.m_color *
                                    quan_parm['atq'])[idx].astype(int)
        quan_parm['offset'] = offset
        self.m_numPoint = quanpt.shape[0]

    def deQuantization(self, dequan_parm):
        self.m_pos = self.m_pos * dequan_parm['qs'] + dequan_parm['offset']
        self.m_color = np.round(self.m_color / dequan_parm['atq']).astype(int)


#_________________________________________________________________________________________________
class TContext():

    def __init__(self, pointCloudOrg: TComPointCloud) -> None:
        self.m_pointCloudOrg = pointCloudOrg
        # data
        self.p_m = None  # p_m: xyz,yuv,ptid, sliceID, ispadding
        self.pred_nnid = None
        self.pred_dis = None

        self.gt_target = None
        self.target = None
        self.pred_yuv =  None
        self.target1 = None
        self.pred_yuv1 =  None

        self.base_layer = None
        self.infer_layer = None
        self.predictors = None

        self.r_idx = None
        self.slice_infer = None

    def Base_LOD(self):
        indexes, self.r_idx, self.predictors = levelOfDetailLayeringStructure(
            self.m_pointCloudOrg.m_pos)
        # sort the org pointcloud by lod order
        self.m_pointCloudOrg.sortByIdx(indexes)
        self.base_layer = self.m_pointCloudOrg.getByIdx(self.r_idx > INTRA_LOD_START)
        self.infer_layer = self.m_pointCloudOrg.getByIdx(
            self.r_idx <= INTRA_LOD_START)
        return self.base_layer.astype(int)

    def Infer_LOD(self):
        # t = time.time()
        # self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct(self.base_layer, self.infer_layer, self.predictors, K=KNN)
        # self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct1(self.base_layer, self.infer_layer, self.r_idx, self.predictors, K=KNN)
        # self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct2(self.base_layer, self.infer_layer, self.r_idx, self.predictors, K=KNN)
        # self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct3(self.base_layer, self.infer_layer, self.predictors, K=KNN)
        # self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct4(self.base_layer, self.infer_layer, self.predictors, K=KNN)

        self.p_m, self.pred_nnid, self.pred_dis, self.slice_infer = InferLodConstruct5(self.base_layer, self.infer_layer, self.r_idx, self.predictors, K=KNN)
        
        # p_m, pred_nnid, pred_dis, slice_infer = InferLodConstruct6(self.base_layer, self.infer_layer, self.r_idx, self.predictors, K=KNN)

        # print(time.time() - t)
        return self.p_m

    def Update_baselayer(self, base_layer):
        self.base_layer = base_layer
        self.m_pointCloudOrg.m_color[:len(base_layer)] = base_layer[:, 3:6]
    
    def Update_pred(self, idx=slice(0,None)):
        # self.Infer_LOD()
        inv_dis = ((MAX_DIS/ (self.pred_dis[idx,0:KNN]))).astype(int)
        pred_nn = self.p_m[self.pred_nnid[idx,0:KNN]]
        pred_yuv = pred_from_lod(None, pred_nn,inv_dis)
        if self.pred_yuv is not None:
            self.pred_yuv[idx] = pred_yuv
        else:
            self.pred_yuv = pred_yuv

    
    def kernelFun(self, x, y, z, half_size,dilation):
        x1 = abs(x - half_size) / 1 * dilation
        y1 = abs(y - half_size) / 1 * dilation
        z1 = abs(z - half_size) / 1 * dilation
        # return math.exp(-(x1**2 + y1**2+ z1**2)/3) # 667646
        # return math.exp(-(x1**2 + y1**2+ z1**2)) # 641077
        return 1/math.sqrt((x1**2 + y1**2+ z1**2)+1e-10) # 656544
        # return 1/(x1+y1+z1+1e-10) # 646464


    
    
    def genKernel(self, size, channels,dilation):
        assert(size % 2 == 1)
        kernel_3D = np.zeros((size,size,size,channels,channels))
        half_size = size // 2
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    kernel_3D[x,y,z] = self.kernelFun(x,y,z, half_size,dilation)
        # kernel_3D[half_size,half_size,half_size] = 0
        return torch.from_numpy(kernel_3D.reshape(-1, channels, channels).astype(np.float32))#.to(torch.float32)

    def Update_pred_Minkowski(self, idx=slice(0,None)):
        totle_ptnum = len(self.p_m)
        pre_ptnum = len(self.base_layer)
        slice_ptnum  =int((totle_ptnum - pre_ptnum) /  MAX_SLICE_NUM)
        pred_yuv = np.zeros((totle_ptnum,3))
        pred_yuv[:pre_ptnum] = self.p_m[:pre_ptnum,3:6]
        coordinatesAll = ME.utils.batched_coordinates([self.p_m[:,0:3]]).to(device)
        # coordinatesAll_idx = self.p_m[:,6] 
        is_no_padding = self.p_m[:,-1]==-1
        dim = 3
        kernel_size = 3
        
        self.pred_nnid1 = np.zeros(self.pred_nnid.shape) 

        with torch.no_grad():

            Conv_fun = ME.MinkowskiConvolution(in_channels=dim+1,out_channels=dim+1,kernel_size=kernel_size,dimension=3,dilation=1).to(device)

            # Conv_fun1 = ME.MinkowskiConvolution(in_channels=dim+1,out_channels=dim+1,kernel_size=kernel_size,dimension=3,dilation=2).to(device)
            # kernel_dis = torch.zeros_like(Conv_fun.kernel.data)#np.sqrt(3)
            # kernel_dis = kernel_dis1
            # kernel_dis[[0,2,6,8,18,20,24,26]] = 1/np.sqrt(3)
            # kernel_dis[[1,3,5,7,9,11,15,17,19,21,23,25]] = 1/np.sqrt(2)
            # kernel_dis[[4,10,12,14,16,22]] = 1
            # kernel_dis[[13]] = 0
            kernel_dis1 = self.genKernel(kernel_size, dim + 1,1)
            # kernel_dis2 = self.genKernel(kernel_size, dim + 1,2)
            Conv_fun.kernel.data = (torch.eye(dim+1,dim+1)[None,...]*kernel_dis1).to(device)
            # Conv_fun1.kernel.data = (torch.eye(dim+1,dim+1)[None,...]*kernel_dis2).to(device)

            for _ in range(MAX_SLICE_NUM):
                idx_pre_slice = np.arange(pre_ptnum)
                idx_in_slice = np.arange(pre_ptnum, pre_ptnum + slice_ptnum)

                not_pd_idx_pre = idx_pre_slice[is_no_padding[idx_pre_slice]]
                yep_pd_idx_pre = idx_pre_slice[~is_no_padding[idx_pre_slice]]
                not_pd_idx_in = idx_in_slice[is_no_padding[idx_in_slice]]
                yep_pd_idx_in = idx_in_slice[~is_no_padding[idx_in_slice]]

                feat =  torch.from_numpy(self.p_m[:pre_ptnum,3:6]).to(device).float()
                feat_add_one = torch.cat((feat,torch.ones_like(feat[:,0:1]).to(device)),1)
                input = ME.SparseTensor(features=feat_add_one,coordinates=coordinatesAll[:pre_ptnum])

                output = Conv_fun(input,coordinatesAll[not_pd_idx_in]) #.features###

                map = input.coordinate_manager.kernel_map(input.coordinate_map_key,output.coordinate_map_key,kernel_size=kernel_size) 

                self.FindNNfromKernalMap(map, not_pd_idx_pre, not_pd_idx_in)


                # output += Conv_fun1(input,coordinatesAll[not_pd_idx]).features.numpy()
                output = output.features
                divide = output[:,3:4]
                divide[divide == 0] = 1
                pred_yuv[not_pd_idx_in] = (output[:,0:3] / divide).cpu().numpy()
                pred_yuv[yep_pd_idx_in] = pred_yuv[self.p_m[yep_pd_idx_in,-1]]
                pre_ptnum += slice_ptnum

        pred_yuv = np.round(pred_yuv).astype(int)
        if self.pred_yuv1 is not None:
            self.pred_yuv1[idx] = pred_yuv
        else:
            self.pred_yuv1 = pred_yuv

    def FindNNfromKernalMap(self, map, not_pd_idx_pre, not_pd_idx_in):
        iomap = torch.cat([i for _,i in map.items()],1)
        org_idx_in = not_pd_idx_in[iomap[1].cpu()]
        org_idx_pre = not_pd_idx_pre[iomap[0].cpu()]
        nnmap = csr_matrix((np.ones(iomap.shape[1]), (org_idx_in, org_idx_pre)), shape=(self.p_m.shape[0], self.p_m.shape[0]))
        rows, cols = nnmap.nonzero()
        qur_indices, row_counts = np.unique(rows, return_counts=True)
        nn_indices = np.split(cols, np.cumsum(row_counts[:-1]))
        # assert(len(nn_indices) == qur_indices.shape)
        # self.p_m[nn_indices[500],0:3] - self.p_m[qur_indices[500],0:3]
        # [self.p_m[nn_indices[x],0:3] - self.p_m[qur_indices[x],0:3] for x in np.where(np.array([len(x) for x in nn_indices])>3)[0] ]
        return qur_indices, nn_indices

    def Predict(self):
        # t = time.time()
        self.Update_pred()
        self.gt_target = self.p_m[:, 3:6] - self.pred_yuv
        
        # code1 = encode_res(self.gt_target[len(self.base_layer):])
        # print("189", time.time() - t,len(code1))

        # t = time.time()
        # self.Update_pred_Minkowski()
        # self.gt_target1 = self.p_m[:, 3:6] - self.pred_yuv1
        # code2 = encode_res(self.gt_target1[len(self.base_layer):])
        # print("195", time.time() - t,len(code2))


        self.target = np.clip(self.gt_target, -ATTIBUTE_HALF_RANGE,
                              ATTIBUTE_HALF_RANGE)
        self.out_target = self.gt_target - self.target
        self.target += ATTIBUTE_HALF_RANGE

    def construct_context(self, slice_id, MAX_BATCH=MAX_BATCH):
        idxs = np.where((self.p_m[:, 7] == -slice_id))[0]
        total_Batch = len(idxs) // 1024
        idxs = idxs.reshape(total_Batch, 1024).tolist()
        dict_datas = []
        for idx in [
                idxs[x:MAX_BATCH + x] for x in range(0, len(idxs), MAX_BATCH)
        ]:
            nn_idx = self.pred_nnid[idx, 0:KNN]
            target = self.target[idx, :]
            dict_datas.append({
                        'geo': self.p_m[idx,:6], \
                        'target': target,\
                        'resnn':  self.p_m[:,3:6][nn_idx,:] - self.pred_yuv[ nn_idx,:] ,\
                        'predint': self.pred_yuv[idx,:], \
                        'nn' : self.p_m[nn_idx,:6],\
                        'padding': self.p_m[idx,8],\
                        'idx': self.p_m[idx,6]
                        }
            )
        return dict_datas


if __name__=='__main__':
    PointCloudReader = TPreprocess()
    PointCloudReader.readOriPointCloud('Data/simple_test_ply/loot_vox10_1200.ply')
    context = TContext(PointCloudReader.preprocess())
    context.Base_LOD()
    context.Infer_LOD()
    context.Predict()