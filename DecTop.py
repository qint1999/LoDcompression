from networkTool import MAX_SLICE_NUM, ATTIBUTE_HALF_RANGE, ATTIBUTE_RES_RANGE, CPrintl, EXPNAME, MAX_SLICE_NUM1
from Common import TContext, TComPointCloud
from EncTop import TEncTop
import numpy as np
import torch
import time
from testTool.testTMC import run_cmd, TestFile


class TDecTop(TEncTop):

    def __init__(self) -> None:
        super().__init__()
        self.m_pointCloudDecode = TComPointCloud()

    def decodeChannel(self,forwordFun, dict_data,pros:list):
        is_not_padding = dict_data['padding']
        t = time.time()
        output_pro = forwordFun(dict_data['base_context'], dict_data['target'])
        softmaxR = torch.softmax(output_pro.detach(), dim=-1)
        net_elapsed = (time.time() - t)
        pros.append(softmaxR[is_not_padding, :])
        return net_elapsed


    def decode(self, inputFileName, bin_dir=EXPNAME, decodeFileName=None):
        # encodeRuslt = self.encode(inputFileName)
        self.__init__()
        self.pharseInput(inputFileName, bin_dir)
        if decodeFileName is not None:
            self.m_reconFileName = decodeFileName
        # dec geo:
        self.m_pointCloudDecode.readFromFile(inputFileName, atriType="xyz")
        self.m_pointCloudDecode.quantization(self.m_quanParm)

        # set dummy color
        self.m_pointCloudDecode.clearColors()

        t0 = time.time()
        context = TContext(self.m_pointCloudDecode)
        base_layer = context.Base_LOD()
        recon_base_layer = self.m_encBac.gpcc_decode(base_layer,
                                                     self.m_colorSpace)
        # assert (recon_base_layer==self.encode_context.base_layer).all()
        context.Update_baselayer(recon_base_layer)
        context.Infer_LOD()
        context.Predict()

        outTarget = self.m_encBac.res_decoder('outTarget', context.p_m.shape[0],3)

        net_elapsed = 0
        # decode net
        for slice_id in range(MAX_SLICE_NUM1):
            dict_data_list, decdidx, paddings = [], [], []
            proR, proG, proB = [], [], []
            target_all = torch.zeros((context.target.shape)).short().cuda()
 
            for dict_data in context.construct_context(slice_id):
                for k in ['geo', 'resnn', 'nn', 'predint']:
                    dict_data[k] = torch.IntTensor(dict_data[k]).cuda()
                dict_data['target'] = torch.LongTensor(dict_data['target']).cuda()
                
                pointNum = dict_data['geo'].shape[0] * dict_data['geo'].shape[1]
                self.m_deepEntroyModel.net_ptNum += pointNum
                is_not_padding = (dict_data['padding'] == -1)
                decdidx.append(dict_data['idx'][is_not_padding])
                paddings.append(np.c_[dict_data['idx'][:, :, None],
                                dict_data['padding'][:, :, None]][~is_not_padding])
                net_t = time.time()
                base_context = self.m_deepEntroyModel.model.forward_base(dict_data, None)
                net_elapsed += ( time.time() - net_t)
                dict_context = {}
                dict_context['target'] = dict_data['target']
                dict_context['padding'] = is_not_padding
                dict_context['base_context'] = base_context
                dict_context['idx'] =  dict_data['idx']
                dict_data_list.append(dict_context)

            decode_idx = np.concatenate(decdidx).reshape(-1)
            padding_idx = np.concatenate(paddings).reshape(-1, 2)
#_________________________________________________
            for dict_data in dict_data_list:
                net_elapsed+=self.decodeChannel(self.m_deepEntroyModel.model.forward1,dict_data,proR)
            decode_prosR = torch.cat(proR).reshape(-1, ATTIBUTE_RES_RANGE)
            del proR
            targetR = self.m_encBac.ac_decoder(decode_prosR,'Rbin{}'.format(slice_id))
            target_all[decode_idx,0] = targetR.reshape(-1)
            # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,0]).all()
            del decode_prosR
            for dict_data in dict_data_list:
                dict_data['target'][:,:,0] = target_all[dict_data['idx'],0]
                net_elapsed+=self.decodeChannel(self.m_deepEntroyModel.model.forward2,dict_data,proG)
            decode_prosG = torch.cat(proG).reshape(-1, ATTIBUTE_RES_RANGE)
            targetG = self.m_encBac.ac_decoder(decode_prosG,'Gbin{}'.format(slice_id))
            target_all[decode_idx,1] = targetG.reshape(-1)
            # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,1]).all()
            del decode_prosG
            del proG
            for dict_data in dict_data_list:
                dict_data['target'][:,:,1] = target_all[dict_data['idx'],1]
                net_elapsed+=self.decodeChannel(self.m_deepEntroyModel.model.forward3,dict_data,proB)
            decode_prosB = torch.cat(proB).reshape(-1, ATTIBUTE_RES_RANGE)
            targetB = self.m_encBac.ac_decoder(decode_prosB,'Bbin{}'.format(slice_id))
            target_all[decode_idx,2] = targetB.reshape(-1)
            # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,2]).all()
            del decode_prosB
            del proB
#_________________________________________________
            context.p_m[decode_idx, 3:6] = context.pred_yuv[decode_idx] + target_all[decode_idx].cpu().numpy() - ATTIBUTE_HALF_RANGE + outTarget[decode_idx]
            context.p_m[padding_idx[:, 0], 3:6] = context.p_m[padding_idx[:, 1],3:6]  # padding
            # assert ( self.encode_context.p_m[decode_idx,:6] == context.p_m[decode_idx,:6] ).all()
            # context.Update_pred(decode_idx)
            context.Update_pred(np.where((context.p_m[:, 7] == (-slice_id-1)))[0]) #  only need update next slice

        self.m_pointCloudDecode.m_color = context.p_m[context.p_m[:, 8] == -1,3:6]
        self.m_pointCloudDecode.m_pos = context.p_m[context.p_m[:, 8] == -1,0:3]
        total_time = time.time() - t0
        self.m_pointCloudDecode.deQuantization(self.m_quanParm)
        if self.m_colorSpace == "YUV":
            self.m_pointCloudDecode.convertYUVToRGB()
        self.m_pointCloudDecode.saveToFile(self.m_reconFileName)
        # run_cmd('testTool/pc_error -a {} -b {} -c 1'.format(self.m_inputFileName,self.m_reconFileName),True)
        return {'DeNet_time': net_elapsed, 'totalDetime': total_time}


if __name__ == '__main__':
    # decoder = TDecTop()
    # print(decoder.decode('andrew9_frame0000.ply'))
    # run_cmd('testTool/pc_error -a {} -b {} -c 1'.format(encoder.m_inputFileName,decoder.m_reconFileName),True)
    MPEGCAT1A = [
    # 'basketball_player_vox11_00000200.ply',
    # 'boxer_viewdep_vox12.ply',
    # 'dancer_vox11_00000001.ply',
    'Egyptian_mask_vox12.ply',
    'Facade_00009_vox12.ply',
    # 'Facade_00015_vox14.ply',
    'Facade_00064_vox11.ply',
    # 'Frog_00067_vox12.ply',
    # 'Head_00039_vox12.ply',
    # 'House_without_roof_00057_vox12.ply',
    # 'longdress_viewdep_vox12.ply',
    # 'longdress_vox10_1300.ply',
    # 'loot_viewdep_vox12.ply',
    # 'loot_vox10_1200.ply',
    'queen_0200.ply',
    # 'redandblack_viewdep_vox12.ply',
    # 'redandblack_vox10_1550.ply',
    # 'Shiva_00035_vox12.ply',
    # 'soldier_viewdep_vox12.ply',
    # 'soldier_vox10_0690.ply',
    # 'Thaidancer_viewdep_vox12.ply',
    'ULB_Unicorn_vox13.ply'
    ]
    import os 
    # printl = CPrintl(EXPNAME + '/test.log')
    # encoder = TEncTop()
    # decoder = TDecTop()
    # test = TestFile(path='Data/test/Owlii/*.ply')  #
    # result = test.testByFun(
    #     encoding_fun=lambda x: dict( **(decoder.decode(x))),
    #     print=printl)
    # result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    # result.to_csv(EXPNAME + '/test.csv')
    # printl(result)


    # printl = CPrintl(EXPNAME + '/test.log')
    # encoder = TEncTop()
    # decoder = TDecTop()
    # test = TestFile(path='Data/test/MVUB/*.ply')  #
    # result = test.testByFun(
    #     encoding_fun=lambda x: dict( **(decoder.decode(x))),
    #     print=printl)
    # result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    # result.to_csv(EXPNAME + '/test2.csv')
    # printl(result)


    # printl = CPrintl(EXPNAME + '/test.log')
    # encoder = TEncTop()
    # decoder = TDecTop()
    # test = TestFile(path='Data/test/8iVFBv2/*.ply')  #
    # result = test.testByFun(
    #     encoding_fun=lambda x: dict( **(decoder.decode(x))),
    #     print=printl)
    # result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    # result.to_csv(EXPNAME + '/test3.csv')
    # printl(result)


    # printl = CPrintl(EXPNAME + '/test.log')
    # encoder = TEncTop()
    # decoder = TDecTop()
    # test = TestFile(path='Data/test/8iViewDep/*.ply')  #
    # result = test.testByFun(
    #     encoding_fun=lambda x: dict( **(decoder.decode(x))),
    #     print=printl)
    # result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    # result.to_csv(EXPNAME + '/test4.csv')
    # printl(result)


    printl = CPrintl(EXPNAME + '/test.log')
    encoder = TEncTop()
    decoder = TDecTop()
    test = TestFile(path='Data/test/Cat1A/*.ply',filter_fun= lambda x:any([y in x for y in MPEGCAT1A]))  #
    result = test.testByFun(
        encoding_fun=lambda x: dict( **(decoder.decode(x))),
        print=printl)
    result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    result.to_csv(EXPNAME + '/test5.csv')
    printl(result)
