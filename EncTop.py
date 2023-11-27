from networkTool import MAX_SLICE_NUM, ATTIBUTE_RES_RANGE, MAX_SLICE_NUM1
from Common import TContext, TPreprocess
import time
import os
from copy import deepcopy
from networkTool import EXPNAME, CPrintl
from testTool.testTMC import TestFile
import torch
from collections import defaultdict
import numpy as np
from networkTool import EXPNAME, reload
from EntropyCoder import TEntropyCoder
from train import Train_and_Test


class TEncCf(TPreprocess):
    from train import Model
    model_path = 'Exp/sparse/qint_fastLoD-KD/checkpoint/encoder_epoch_00016320.pth'
    saveDic = reload(None, model_path)
    assert saveDic is not None
    Model.load_state_dict(saveDic['encoder'])

    def __init__(self) -> None:
        super().__init__()  #TPreprocess
        self.m_inputFileName = ""
        self.m_bitstreamFileName = ""
        self.m_reconFileName = ""

        self.m_encBac = TEntropyCoder()
        self.m_deepEntroyModel = Train_and_Test(
            model=TEncCf.Model, model_path=None)  #do not load again
        self.m_deepEntroyModel.model_path = TEncCf.model_path

    def pharseInput(self, inputFileName, out_bin_dir=EXPNAME):
        self.m_inputFileName = inputFileName
        self.m_bitstreamFileName = os.path.join(
            out_bin_dir + '/bin',
            os.path.basename(inputFileName)[:-3])
        self.m_encBac.bin_path = self.m_bitstreamFileName
        self.m_reconFileName = os.path.join(out_bin_dir + '/dec',
                                            inputFileName[:-4] + '_rec.ply')

        # for test
        self.m_deepEntroyModel.net_acbit = 0
        self.m_deepEntroyModel.net_celoss = 0
        self.m_deepEntroyModel.net_ptNum = 0
        self.m_deepEntroyModel.net_elapsed = 0


class TEncTop(TEncCf):

    def __init__(self) -> None:
        super().__init__()
        self.m_pointCloudOrg = None
        self.m_pointCloudQuant = None
        self.m_pointCloudRecon = None
        self.m_attrEncoder = None
        self.encode_bitnum = defaultdict(int)
        self.encode_ptnum = 0
        self.actual_code = True
        self.encode_context = None
    # def quantization(self):
    #     self.m_pointCloudQuant = deepcopy(self.m_pointCloudOrg)
    #     self.m_pointCloudQuant.quantization(self.m_quanParm)

    def encode(self, inputFileName, out_bin_dir=EXPNAME):
        self.__init__()
        self.pharseInput(inputFileName, out_bin_dir)

        self.readOriPointCloud(self.m_inputFileName)
        self.m_pointCloudOrg = self.m_pointCloudIn
        self.m_pointCloudQuant = self.preprocess(copy_ori_pt=True)

        # geo coding:
        self.m_pointCloudRecon = deepcopy(self.m_pointCloudQuant)
        # atr coding:
        result = self.predictAndEncodeAttribute()
        return result

    def predictAndEncodeAttribute(self):
        t0 = time.time()
        context = TContext(self.m_pointCloudRecon)
        self.encode_context = context
        # t1 = time.time()
        base_layer = context.Base_LOD()
        # t2 = time.time()
        gpcc_codec = self.m_encBac.gpcc_encode(base_layer, self.m_colorSpace)
        self.encode_bitnum['gpcc'] += gpcc_codec['atr_bits']
        self.encode_ptnum += base_layer.shape[0]
        # t3 = time.time()
        context.Infer_LOD()
        # t4 = time.time()
        context.Predict()
        # print(t0 - t1,t1 - t2,t2 - t3,t3 - t4,t4 - time.time())
        context_time = (time.time() - t0)

        # net

        net_bin_code_len = 0
        net_elapsed = 0
        entropy_coding_time = 0
        actual_code_ptNum = 0

        for slice_id in range(-context.slice_infer):
            pros, residuals = [], []
            for dict_data in context.construct_context(slice_id):
                for k in ['geo', 'resnn', 'nn', 'predint']:
                    dict_data[k] = torch.IntTensor(dict_data[k]).cuda()
                dict_data['target'] = torch.LongTensor(
                    dict_data['target']).cuda()
                is_not_padding = (dict_data['padding'] == -1)
                t_net = time.time()
                pro = self.m_deepEntroyModel.network(dict_data,
                                                     test_type='encode')
                net_elapsed += (time.time() - t_net)
                pros.append(pro[is_not_padding, :, :])
                residuals.append(dict_data["target"][is_not_padding, :, :])

            # entropy coding
            residuals = torch.cat(residuals).reshape(-1, 3)
            actual_code_ptNum += len(residuals)
            t_en = time.time()
            if self.actual_code:
                pros = torch.cat(pros).reshape(-1, 3, ATTIBUTE_RES_RANGE)
                net_bin_code_len += self.m_encBac.ac_encoder3(
                    residuals, pros, slice_id)
            else:
                net_bin_code_len = self.m_deepEntroyModel.net_acbit
            entropy_coding_time += (time.time() - t_en)

        net_bin_code_len += self.m_encBac.res_encoder(context.out_target,
                                                      'outTarget')

        self.encode_bitnum['net'] = net_bin_code_len
        self.encode_ptnum += actual_code_ptNum
        total_time = (time.time() - t0)

        # stat
        assert self.encode_ptnum == self.m_pointCloudRecon.m_numPoint
        bpp = [(a, np.round(b / self.encode_ptnum, 2))
               for a, b in self.encode_bitnum.items()]
        # self.print('bpp',np.round(sum([x[1] for x in bpp]),4))
        our_bpp = sum([x[1] for x in bpp])
        result = {
                    'File': os.path.basename(self.m_inputFileName),\
                    'inPtNum': self.m_pointCloudOrg.m_numPoint,\
                    'outPtNum': self.encode_ptnum,\
                    'bpop':  our_bpp,\
                    'ConTime': context_time,\
                    'NetTime': net_elapsed,\
                    'Ac_time': entropy_coding_time,\
                    'EnToTaltime': total_time,\
                    'Bppinfo': bpp,\
                    'Bits': sum(self.encode_bitnum.values()),\
                }
        # print(result)
        return result


if __name__ == '__main__':
    printl = CPrintl(EXPNAME + '/test.log')
    encoder = TEncTop()
    printl('deep entropy model:', encoder.m_deepEntroyModel.model_path)
    test = TestFile(path='Data/simple_test_ply/Frog_00067_vox12.ply')
    result = test.testByFun(encoding_fun=lambda x: encoder.encode(x),
                            print=printl)
    # result.iloc[-1, 0] = model_path
    result.to_csv(EXPNAME + '/test.csv')
    printl(result)
