from genlod.lod_warapper import genLod
from networkTool import INTRA_LOD_START, MAX_SLICE_NUM, KNN, MAX_DIS, MAX_SLICE_NUM1, MAX_BATCH,device
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import testTool.pt as pt
import time
from testTool.kdtree import FaissKDTree
import torch

def convertRow2ColIdx(shape, padid):
    row = padid // shape[1]
    col = padid - row * shape[1]
    s1 = col * shape[0] + row
    return s1.astype(int)


def gen_block2(len_base, pre_infer_layer, post_infer_layer, bptt=1024, SLICE=1):
    numOfPointinBlock = int(bptt * SLICE)
    len_pre = len(pre_infer_layer)
    len_post = len(post_infer_layer)
    t = (len_pre // numOfPointinBlock + 1) * numOfPointinBlock - len_pre
    sample_indices = np.linspace(0, len_post - 1, t, dtype=np.int32)
    pre_infer_layer = np.vstack((pre_infer_layer, post_infer_layer[sample_indices]))
    post_infer_layer = np.delete(post_infer_layer, sample_indices, axis=0)

    pre_infer_layer[:, 8] = -1
    pre_infer_layer, _ = pt.sortByMorton(pre_infer_layer, return_idx=True)
    
    len_post = len(post_infer_layer)
    t = (len_post // numOfPointinBlock + 1) * numOfPointinBlock - len_post
    padding_indices = np.linspace(0, len_post - 1, t, dtype=np.int32)
    idxWithPadding = np.hstack((np.arange(len_post), padding_indices))
    
    post_infer_layer = post_infer_layer[idxWithPadding]
    post_infer_layer[:-t, 8] = -1
    post_infer_layer[-t:, 8] = padding_indices + len_base + len(pre_infer_layer)
    post_infer_layer, _ = pt.sortByMorton(post_infer_layer, return_idx=True)
    return pre_infer_layer, post_infer_layer



def gen_block_by_kmeans1(post_base_layer,
                        post_infer_layer,
                        pre_infer_layer,
                        geo_weight=255,
                        bptt=1024,
                        SLICE=4):
    numOfPointinBlock = bptt * SLICE * 4
    tree_post_base = KDTree(post_base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_post_base.query(post_base_layer[:, :3], k=50)
    base_layer_ave = post_base_layer[min_nnIDx, 3:6].mean(1)
    post_base_ord = post_base_layer[:, :3].copy().astype(float)
    post_base_ord = post_base_ord - post_base_ord.min()
    post_base_ord /= post_base_ord.max()
    pre_block_num = pre_infer_layer.shape[0]  // numOfPointinBlock + 1
    # pre_lackN = int(pre_block_num * numOfPointinBlock) - pre_infer_layer.shape[0]

    kmeans = KMeans(n_clusters=pre_block_num, random_state=0,
                    max_iter=10).fit(np.c_[post_base_ord[:, :3] * geo_weight,
                                           base_layer_ave])
    post_base_label = kmeans.labels_
    
    _, min_nnIDx = tree_post_base.query(pre_infer_layer[:, :3], k=1)
    _, min_nnIDx1 = tree_post_base.query(post_infer_layer[:, :3], k=1)

    infer_label = post_base_label[min_nnIDx]
    infer_label1 = post_base_label[min_nnIDx1]

    label = np.ones((len(pre_infer_layer), 2)).astype(int) * (-1)  # pointid,bkid,ispadding
    label[:, 0] = np.arange(len(pre_infer_layer))
    label[:, 1] = infer_label
    label1 = np.ones((len(post_infer_layer), 2)).astype(int) * (-1)  # pointid,bkid,ispadding
    label1[:, 0] = np.arange(len(post_infer_layer)) + len(pre_infer_layer)
    label1[:, 1] = infer_label1
    sortBks = np.argsort(
        [post_base_layer[post_base_label == x, 0:3].mean() for x in range(pre_block_num)])
    
    data = []
    label_set = []
    # borrowDatas = []
    remaingdatas = []
    for i in sortBks:
        idx = label[label[:, 1] == i, 0]
        idx1 = label1[label1[:, 1] == i, 0] - len(pre_infer_layer)
        t = (len(idx) // numOfPointinBlock + 1) * numOfPointinBlock - len(idx)
        if t <= len(idx1):
            data.append(np.hstack((label[idx, 0], label1[idx1[0:t], 0])).reshape(-1, SLICE))
            label_set.append(np.hstack((label[idx, 1], label1[idx1[0:t], 1])).reshape(-1, SLICE))
            remaingdatas.append(label1[idx1[t:], 0])
        else:
            remaingdatas.append(np.hstack((label[idx, 0], label1[idx1[0:t], 0])))


    remaingdata = np.concatenate(remaingdatas, 0)
    pre_infer_label = np.concatenate(label_set, 0).T.reshape(-1)
    IDbyBK = np.concatenate(data, 0)
    pre_slice_ptnum = IDbyBK.shape[0]
    infer_layer = np.vstack((pre_infer_layer,post_infer_layer))
    pre_infer_layer = infer_layer[IDbyBK.T.reshape(-1)]
    pre_infer_layer[:, 8] = -1 # 仅仅是为了对齐
    post_infer_layer = infer_layer[remaingdata]

    ########################################################################
    base_With_Infer_layer = np.concatenate([post_base_layer[:, :3], pre_infer_layer[:, :3]], 0)

    tree_Base_With_Infer = KDTree(base_With_Infer_layer[:, :3], compact_nodes=False)

    #########################################################################
    # min_dis, min_nnIDx = tree_Base_With_Infer.query(base_With_Infer_layer[:, :3], k=20)
    # base_With_Infer_layer_ave = base_With_Infer_layer[min_nnIDx, 3:6].mean(1)
    # base_With_Infer_base_ord = base_With_Infer_layer[:, :3].copy().astype(float)
    # base_With_Infer_base_ord = base_With_Infer_base_ord - base_With_Infer_base_ord.min()
    # base_With_Infer_base_ord /= base_With_Infer_base_ord.max()
    # # base_With_Infer_block_num = post_infer_layer.shape[0]  // numOfPointinBlock
    # # pre_lackN = int(pre_block_num * numOfPointinBlock) - pre_infer_layer.shape[0]

    # kmeans = KMeans(n_clusters=pre_block_num, random_state=0,
    #                 max_iter=10).fit(np.c_[base_With_Infer_base_ord[:, :3] * geo_weight,
    #                                        base_With_Infer_layer_ave])
    # base_With_Infer_label1 = kmeans.labels_
    
    #########################################################################
    base_With_Infer_label = np.hstack((post_base_label, pre_infer_label))

    min_dis, min_nnIDx = tree_Base_With_Infer.query(post_infer_layer[:, :3], k=1)
    infer_label = base_With_Infer_label[min_nnIDx]
    label = np.ones(
        (len(post_infer_layer), 3)).astype(int) * (-1)  # pointid,bkid,ispadding
    label[:, 0] = np.arange(len(post_infer_layer))
    label[:, 1] = infer_label
    # sortBks = np.argsort(
    #     [base_With_Infer_layer[base_With_Infer_label == x, 0:3].mean() for x in range(pre_block_num)])
    data = []
    remaingdatas = []
    for i in sortBks:
        idx = label[label[:, 1] == i, 0]
        t = len(idx) // numOfPointinBlock * numOfPointinBlock
        data.append(label[idx[:t], 0].reshape(-1, SLICE))
        remaingdatas.append(label[idx[t:], 0])

    ########################################################################
    remaingdata = np.concatenate(remaingdatas, 0)
    remainN = remaingdata.shape[0]
    lackN = int(np.ceil(remainN / numOfPointinBlock) * numOfPointinBlock) - remainN
    IDremaing = np.concatenate((remaingdata, np.ones(
        (lackN,), int) * -1)).reshape(-1, SLICE)
    IDbyBK = np.concatenate(data, 0)


    # gen infer points with padding
    BLS = np.r_[IDbyBK, IDremaing]  # BLS order: SLICE_PTNUM * SLICE
    post_slice_ptnum = BLS.shape[0]
    padid = np.where(BLS.reshape(-1) == -1)[0][0] + np.arange(
        0, lackN)  # locate the first padding data
    s1 = convertRow2ColIdx((post_slice_ptnum, SLICE),
                           padid)  # the padding data in new order (BLS)
    s2 = convertRow2ColIdx(
        (post_slice_ptnum, SLICE), padid - np.ceil(lackN / SLICE) * SLICE
    )  # the refer data (remaingdata[-lackN:]) of padding data in new order (BLS)
    BLS = BLS.T.reshape(-1)
    BLS[s1] = BLS[s2]
    post_infer_layer = post_infer_layer[BLS]
    post_infer_layer[:, 8] = -1
    post_infer_layer[s1, 8] = s2
    return pre_slice_ptnum, post_slice_ptnum  # xyz,yuv,ptid(no use), sliceID (no use), ispadding (>-1 for paading,-1 for not padding)


def gen_block_by_kmeans(base_layer,
                        infer_layer,
                        block_num,
                        geo_weight=255,
                        bptt=1024,
                        SLICE=MAX_SLICE_NUM):
    tree_base = KDTree(base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_base.query(base_layer[:, :3], k=50)
    base_layer_ave = base_layer[min_nnIDx, 3:6].mean(1)
    base_ord = base_layer[:, :3].copy().astype(float)
    base_ord = base_ord - base_ord.min()
    base_ord /= base_ord.max()
    kmeans = KMeans(n_clusters=block_num, random_state=0,
                    max_iter=10).fit(np.c_[base_ord[:, :3] * geo_weight,
                                           base_layer_ave])
    base_label = kmeans.labels_
    min_dis, min_nnIDx = tree_base.query(infer_layer[:, :3], k=1)
    infer_label = base_label[min_nnIDx]
    label = np.ones(
        (len(infer_layer), 3)).astype(int) * (-1)  # pointid,bkid,ispadding
    label[:, 0] = np.arange(len(infer_layer))
    label[:, 1] = infer_label
    _, sortBks = pt.sortByMorton(np.concatenate([
        base_layer[base_label == x, 0:3].mean(0, keepdims=True)
        for x in range(block_num)
    ], 0).astype(int),
                                 return_idx=True)
    sortBks = np.argsort(
        [base_layer[base_label == x, 0:3].mean() for x in range(block_num)])
    data = []
    remaingdatas = []
    for i in sortBks:
        idx = label[label[:, 1] == i, 0]
        t = len(idx) // (SLICE * bptt) * (SLICE * bptt)
        data.append(label[idx[:t], 0].reshape(-1, SLICE))
        remaingdatas.append(label[idx[t:], 0])
    remaingdata = np.concatenate(remaingdatas, 0)
    remainN = remaingdata.shape[0]
    lackN = int(np.ceil(remainN / (MAX_SLICE_NUM * bptt)) * (MAX_SLICE_NUM * bptt)) - remainN
    IDremaing = np.concatenate((remaingdata, np.ones(
        (lackN,), int) * -1)).reshape(-1, SLICE)
    IDbyBK = np.concatenate(data, 0)

    # gen infer points with padding
    BLS = np.r_[IDbyBK, IDremaing]  # BLS order: SLICE_PTNUM * SLICE
    slice_ptnum = BLS.shape[0]
    padid = np.where(BLS.reshape(-1) == -1)[0][0] + np.arange(
        0, lackN)  # locate the first padding data
    s1 = convertRow2ColIdx((slice_ptnum, MAX_SLICE_NUM),
                           padid)  # the padding data in new order (BLS)
    s2 = convertRow2ColIdx(
        (slice_ptnum, MAX_SLICE_NUM), padid - np.ceil(lackN / MAX_SLICE_NUM) * MAX_SLICE_NUM
    )  # the refer data (remaingdata[-lackN:]) of padding data in new order (BLS)
    BLS = BLS.T.reshape(-1)
    BLS[s1] = BLS[s2]
    infer_padding = infer_layer[BLS]
    all_layer = np.r_[base_layer, infer_padding]
    all_layer[:, 8] = -1
    all_layer[s1 + len(base_layer), 8] = s2 + len(base_layer)
    # infer_padding.reshape(-1,16); each clos is a slice; continued rows are bks
    # a = all_layer.astype(int)
    # assert (a[a[a[:,8]>-1,8]]==a[a[:,8]>-1])[:,:8].all()
    return all_layer, slice_ptnum  # xyz,yuv,ptid(no use), sliceID (no use), ispadding (>-1 for paading,-1 for not padding)


def levelOfDetailLayeringStructure(pos):
    ptNum = pos.shape[0]
    p = np.c_[pos, np.zeros((ptNum, 3))]
    pred_pt, indexes, r_idx, predictors, gt_lodoerder_pt = genLod(
        p, return_lod_order=True)
    r_idx = r_idx.max() - r_idx + 1
    return indexes, r_idx, predictors


def pred_from_lod(p_m, pred_nn, pred_dis_inv):
    nn_color = pred_nn[:, :, 3:6]
    dis = pred_dis_inv
    pred_color = (dis[...,None].transpose(0,2,1) @ nn_color).squeeze(1)
    DIS = dis.sum(1, keepdims=True)
    pred_color_norm = np.round(pred_color/np.where(DIS == 0, 1, DIS)).astype(int)
    return pred_color_norm


def InferLodConstruct1(base_layer, infer_layer, r_idx, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]


    is_post_in_base = r_idx[0:base_layer.shape[0]] == INTRA_LOD_START + 1
    post_base_layer = base_layer[is_post_in_base]
    pre_base_layer = base_layer[~is_post_in_base]
    is_post_in_infer = r_idx[base_layer.shape[0]:] != INTRA_LOD_START
    post_infer_layer = infer_layer[is_post_in_infer]
    pre_infer_layer = infer_layer[~is_post_in_infer]

    pre_slice_ptnum, post_slice_ptnum = gen_block_by_kmeans1(post_base_layer, post_infer_layer, pre_infer_layer, SLICE=MAX_SLICE_NUM) #MAX_SLICE_NUM

    p_m = np.concatenate((base_layer, pre_infer_layer, post_infer_layer), axis = 0)
    len_pre_base = len(pre_base_layer)
    len_base = len(base_layer)
    len_pre_infer = len(pre_infer_layer)
    ptNum = p_m.shape[0]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS * 2
    tree_gloabl = KDTree(pre_base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(post_base_layer[:, :3], K, workers=-1)
    pred_dis[len_pre_base:len_base] = min_dis
    pred_nnid[len_pre_base:len_base] = min_nnIDx

    tree_gloabl = KDTree(base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(pre_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base:len_base + len_pre_infer] = min_dis
    pred_nnid[len_base:len_base + len_pre_infer] = min_nnIDx + 0

    tree_gloabl = KDTree(np.r_[base_layer[:, :3], pre_infer_layer[:, :3]], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(post_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base + len_pre_infer:] = min_dis
    pred_nnid[len_base + len_pre_infer:] = min_nnIDx + 0

    return p_m.astype(int), pred_nnid, pred_dis.astype(int)

def InferLodConstruct2(base_layer, infer_layer, r_idx, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]


    is_post_in_base = r_idx[0:base_layer.shape[0]] == INTRA_LOD_START + 1
    post_base_layer = base_layer[is_post_in_base]
    pre_base_layer = base_layer[~is_post_in_base]
    is_post_in_infer = r_idx[base_layer.shape[0]:] != INTRA_LOD_START
    post_infer_layer = infer_layer[is_post_in_infer]
    pre_infer_layer = infer_layer[~is_post_in_infer]

    # pre_slice_ptnum, post_slice_ptnum = gen_block_by_kmeans1(post_base_layer, post_infer_layer, pre_infer_layer, SLICE=MAX_SLICE_NUM) #MAX_SLICE_NUM

    p_m = np.concatenate((base_layer, pre_infer_layer, post_infer_layer), axis = 0)
    len_pre_base = len(pre_base_layer)
    len_base = len(base_layer)
    len_pre_infer = len(pre_infer_layer)
    ptNum = p_m.shape[0]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS * 2
    tree_gloabl = KDTree(pre_base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(post_base_layer[:, :3], K, workers=-1)
    pred_dis[len_pre_base:len_base] = min_dis
    pred_nnid[len_pre_base:len_base] = min_nnIDx

    tree_gloabl = KDTree(base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(pre_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base:len_base + len_pre_infer] = min_dis
    pred_nnid[len_base:len_base + len_pre_infer] = min_nnIDx

    tree_gloabl = KDTree(np.r_[base_layer[:, :3], pre_infer_layer[:, :3]], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(post_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base + len_pre_infer:] = min_dis
    pred_nnid[len_base + len_pre_infer:] = min_nnIDx

    return p_m.astype(int), pred_nnid, pred_dis.astype(int)

def InferLodConstruct6(base_layer, infer_layer, r_idx, predictors, K=KNN):
    numOfPointinBlock = MAX_BATCH * 1024
    slice_infer = 0
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]

    is_post_in_infer = r_idx[base_layer.shape[0]:] != INTRA_LOD_START
    post_infer_layer = infer_layer[is_post_in_infer]
    pre_infer_layer = infer_layer[~is_post_in_infer]

    pre_infer_layer, post_infer_layer = gen_block2(base_layer.shape[0], pre_infer_layer, post_infer_layer) #MAX_SLICE_NUM

    p_m = np.concatenate((base_layer, pre_infer_layer, post_infer_layer), axis = 0)
    len_base = len(base_layer)
    len_pre_infer = len(pre_infer_layer)
    ptNum = p_m.shape[0]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS * 2
    
    tree_gloabl = FaissKDTree(base_layer[:, :3])
    Block_list = list(range(0, len_base, numOfPointinBlock)) + [len_base]
    for i in range(len(Block_list) - 1):
        min_dis, min_nnIDx = tree_gloabl.search(p_m[Block_list[i]:Block_list[i + 1], :3], K + 1)
        pred_dis[Block_list[i]:Block_list[i + 1]] = min_dis[:,1:]
        pred_nnid[Block_list[i]:Block_list[i + 1]] = min_nnIDx[:,1:]
    p_m[:len_base, 7] = 1 

    len_base_infer = len_base + len_pre_infer

    Block_list = list(range(len_base, len_base_infer, numOfPointinBlock)) + [len_base_infer]
    for i in range(len(Block_list) - 1):
        min_dis, min_nnIDx = tree_gloabl.search(p_m[Block_list[i]:Block_list[i + 1], :3], K)
        pred_dis[Block_list[i]:Block_list[i + 1]] = min_dis
        pred_nnid[Block_list[i]:Block_list[i + 1]] = min_nnIDx
        p_m[Block_list[i]:Block_list[i + 1], 7] = slice_infer
        slice_infer -= 1
        
    tree_gloabl = FaissKDTree(np.r_[base_layer[:, :3], pre_infer_layer[:, :3]])
    Block_list = list(range(len_base_infer, ptNum, numOfPointinBlock)) + [ptNum]
    for i in range(len(Block_list) - 1):
        min_dis, min_nnIDx = tree_gloabl.search(p_m[Block_list[i]:Block_list[i + 1], :3], K)
        pred_dis[Block_list[i]:Block_list[i + 1]] = min_dis
        pred_nnid[Block_list[i]:Block_list[i + 1]] = min_nnIDx
        p_m[Block_list[i]:Block_list[i + 1], 7] = slice_infer
        slice_infer -= 1
        
    # p_m[len_base_infer:, 7] = -1
    return p_m.astype(int), pred_nnid, pred_dis.astype(int), slice_infer

def InferLodConstruct5(base_layer, infer_layer, r_idx, predictors, K=KNN):
    numOfPointinBlock = MAX_BATCH * 1024
    slice_infer = 0
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]

    is_post_in_infer = r_idx[base_layer.shape[0]:] != INTRA_LOD_START
    post_infer_layer = infer_layer[is_post_in_infer]
    pre_infer_layer = infer_layer[~is_post_in_infer]

    pre_infer_layer, post_infer_layer = gen_block2(base_layer.shape[0], pre_infer_layer, post_infer_layer) #MAX_SLICE_NUM

    p_m = np.concatenate((base_layer, pre_infer_layer, post_infer_layer), axis = 0)
    len_base = len(base_layer)
    len_pre_infer = len(pre_infer_layer)
    ptNum = p_m.shape[0]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS * 2
    
    tree_gloabl = KDTree(base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(base_layer[:, :3], K + 1, workers=-1)
    pred_dis[:len_base] = min_dis[:,1:]
    pred_nnid[:len_base] = min_nnIDx[:,1:]
    p_m[:len_base, 7] = 1 

    len_base_infer = len_base + len_pre_infer
    min_dis, min_nnIDx = tree_gloabl.query(pre_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base:len_base_infer] = min_dis
    pred_nnid[len_base:len_base_infer] = min_nnIDx
    Block_list = list(range(len_base, len_base_infer, numOfPointinBlock)) + [len_base_infer]
    for i in range(len(Block_list) - 1):
        p_m[Block_list[i]:Block_list[i + 1], 7] = slice_infer
        slice_infer -= 1
        
    tree_gloabl = KDTree(np.r_[base_layer[:, :3], pre_infer_layer[:, :3]], compact_nodes=False)
    min_dis, min_nnIDx = tree_gloabl.query(post_infer_layer[:, :3], K, workers=-1)
    pred_dis[len_base_infer:] = min_dis
    pred_nnid[len_base_infer:] = min_nnIDx
    Block_list = list(range(len_base_infer, ptNum, numOfPointinBlock)) + [ptNum]
    for i in range(len(Block_list) - 1):
        p_m[Block_list[i]:Block_list[i + 1], 7] = slice_infer
        slice_infer -= 1
        
    # p_m[len_base_infer:, 7] = -1
    return p_m.astype(int), pred_nnid, pred_dis.astype(int), slice_infer


def InferLodConstruct3(base_layer, infer_layer, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    
    infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer,
                                                     infer_layer,
                                                     len(infer_layer) //
                                                     (MAX_SLICE_NUM * 4096),
                                                     SLICE=MAX_SLICE_NUM)
    p_m = infer_padding
    ptNum = len(p_m)
    p_m[:, 6] = np.arange(ptNum)
    order_pm = np.zeros((p_m.shape[0],),dtype = 'uint8')
    base_ptnum = len(base_layer)
    
    for i in range(MAX_SLICE_NUM):
        order_pm[base_ptnum + i * slice_ptnum: base_ptnum + i * slice_ptnum + slice_ptnum] = i + 1
    
    tree_gloabl = KDTree(p_m[:, :3], compact_nodes=False)
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS*2
    pred_nnid[:base_ptnum, :3] = predictors[:base_ptnum, 2:5]
    dis = ((p_m[predictors[:base_ptnum, 2:5], 0:3] -
            np.expand_dims(p_m[:base_ptnum, 0:3], 1))**2).sum(2)
    pred_dis[:base_ptnum, 0:3] = dis
    nearest_idx = np.arange(ptNum)
    for i in range(MAX_SLICE_NUM-1,0,-1):
        idx_in_slice = slice(base_ptnum + i * slice_ptnum, base_ptnum + i * slice_ptnum + slice_ptnum)
        temp_fine = p_m[idx_in_slice]
        kk = int(np.ceil(K * ptNum / idx_in_slice.start)) + 1
        _, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], kk, workers=-1)
        min_nnIDx = min_nnIDx[:,1: K + 1]
        # assert((np.count_nonzero(order_pm[min_nnIDx] < i,axis=1) == 0).any() == False)
        while (np.count_nonzero(order_pm[min_nnIDx] < i,axis=1) == 0).any():
            kk += K
            nothingList = np.where((np.count_nonzero(order_pm[min_nnIDx] < i,axis=1) == 0))[0]
            _, min_nnIDx1 = tree_gloabl.query(temp_fine[nothingList, :3], kk, workers=-1)
            min_nnIDx[nothingList] = min_nnIDx1[:,-K:]
        rows, cols = (min_nnIDx<idx_in_slice.start).nonzero()
        RR, row_counts = np.unique(rows, return_counts=True)
        firstTrueofRR = cols[np.append(0,np.cumsum(row_counts[:-1]))]
        nearest_idx[idx_in_slice] = min_nnIDx[RR,firstTrueofRR]
        nearest_idx = nearest_idx[nearest_idx] # idx_in_slice.stop之后的点云的idx还是很大也要更新
        min_nnIDx = nearest_idx[min_nnIDx]
        assert((min_nnIDx<idx_in_slice.start).all())

        pred_nnid[idx_in_slice] = min_nnIDx
        p_m[idx_in_slice, 7] = -i
        pred_dis[idx_in_slice] =  ((p_m[min_nnIDx, 0:3] - np.expand_dims(p_m[idx_in_slice, 0:3], 1))**2).sum(2)
    p_m[:base_layer.shape[0], 7] = 1
    pred_dis = np.clip(pred_dis,1,2*MAX_DIS)
    return p_m.astype(int), pred_nnid, pred_dis.astype(int)

def InferLodConstruct4(base_layer, infer_layer, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    
    infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer,
                                                     infer_layer,
                                                     len(infer_layer) //
                                                     (MAX_SLICE_NUM * 4096),
                                                     SLICE=MAX_SLICE_NUM)
    
    p_m = infer_padding
    base_ptnum = len(base_layer)
    ptNum = p_m.shape[0]
    p_m[:, 6] = np.arange(ptNum)
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS*2

    treeBase = KDTree(p_m[:len(base_layer), :3], compact_nodes=False)
    min_dis, min_nnIDx = treeBase.query(p_m[:len(base_layer), :3], K + 1, workers=-1)
    pred_nnid[:base_ptnum] = min_nnIDx[:, 1:]
    pred_dis[:base_ptnum] =  min_dis[:, 1:]
    
    treeSet = []
    for i in range(MAX_SLICE_NUM):
        KK = int(max(np.ceil(K / (i + 1)), 1))
        idx_in_slice = slice(base_ptnum + i * slice_ptnum,
                             base_ptnum + (i + 1) * slice_ptnum)
        temp_fine = p_m[idx_in_slice]
        
        min_dis0, min_nnIDx0 = treeBase.query(temp_fine[:, :3], KK, workers=-1)
        if min_nnIDx0.ndim == 1:
            min_dis0 = np.expand_dims(min_dis0, axis = -1)
            min_nnIDx0 = np.expand_dims(min_nnIDx0, axis = -1)
        min_dis, min_nnIDx = [min_dis0], [min_nnIDx0]
        for j in range(len(treeSet)):
            min_dis0, min_nnIDx0 = treeSet[j].query(temp_fine[:, :3], KK, workers=-1)
            if min_nnIDx0.ndim == 1:
                min_dis0 = np.expand_dims(min_dis0, axis = -1)
                min_nnIDx0 = np.expand_dims(min_nnIDx0, axis = -1)
            min_dis.append(min_dis0)
            min_nnIDx.append(min_nnIDx0 + base_ptnum + j * slice_ptnum)

        min_dis = np.hstack(min_dis)
        min_nnIDx = np.hstack(min_nnIDx)
        if min_dis.shape[1] != K:
            idx_minK = np.argpartition(min_dis, K, axis = 1)[:,:K]
            min_dis = np.take_along_axis(min_dis, idx_minK, axis=1)
            min_nnIDx = np.take_along_axis(min_nnIDx, idx_minK, axis=1)

        pred_nnid[idx_in_slice] = min_nnIDx
        p_m[idx_in_slice, 7] = -i
        pred_dis[idx_in_slice] = min_dis**2
        if i != MAX_SLICE_NUM - 1:
            treeSet.append(KDTree(p_m[idx_in_slice, :3], compact_nodes=False))

    p_m[:base_layer.shape[0], 7] = 1
    pred_dis = np.clip(pred_dis,1,2*MAX_DIS)
    return p_m.astype(int), pred_nnid, pred_dis.astype(int)


def InferLodConstruct(base_layer, infer_layer, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    
    infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer,
                                                     infer_layer,
                                                     len(infer_layer) //
                                                     (MAX_SLICE_NUM * 4096),
                                                     SLICE=MAX_SLICE_NUM)
    p_m = infer_padding
    base_ptnum = len(base_layer)
    ptNum = p_m.shape[0]
    p_m[:, 6] = np.arange(ptNum)
    gloabl = p_m[:len(base_layer), :]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS*2
    pred_nnid[:base_ptnum, :3] = predictors[:base_ptnum, 2:5]
    dis = ((p_m[predictors[:base_ptnum, 2:5], 0:3] -
            np.expand_dims(p_m[:base_ptnum, 0:3], 1))**2).sum(2)
    pred_dis[:base_ptnum, 0:3] = dis

    for i in range(MAX_SLICE_NUM):
        idx_in_slice = slice(base_ptnum + i * slice_ptnum,
                             base_ptnum + i * slice_ptnum + slice_ptnum)
        temp_fine = p_m[idx_in_slice]
        tree_gloabl = KDTree(gloabl[:, :3], compact_nodes=False)
        min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)
        pred_nnid[idx_in_slice] = gloabl[min_nnIDx, 6]
        p_m[idx_in_slice, 7] = -i
        pred_dis[idx_in_slice] = min_dis**2
        gloabl = np.r_[gloabl, temp_fine]
    p_m[:base_layer.shape[0], 7] = 1
    pred_dis = np.clip(pred_dis,1,2*MAX_DIS)
    return p_m.astype(int), pred_nnid, pred_dis.astype(int)


def InferLodConstruct_qint(base_layer, infer_layer, predictors, K=KNN):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    
    infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer,
                                                     infer_layer,
                                                     len(infer_layer) //
                                                     (MAX_SLICE_NUM * 4096),
                                                     SLICE=MAX_SLICE_NUM)
    p_m = infer_padding
    base_ptnum = len(base_layer)
    ptNum = p_m.shape[0]
    p_m[:, 6] = np.arange(ptNum)
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS*2
    pred_nnid[:base_ptnum, :3] = predictors[:base_ptnum, 2:5]
    dis = ((p_m[predictors[:base_ptnum, 2:5], 0:3] -
            np.expand_dims(p_m[:base_ptnum, 0:3], 1))**2).sum(2)
    pred_dis[:base_ptnum, 0:3] = dis

    tree_gloabl = KDTree(p_m[:, :3], compact_nodes=False)
    is_not_used = np.ones((ptNum, 1), bool)
    is_not_used[:base_ptnum] = ~is_not_used[:base_ptnum]

    p_m[:base_ptnum, 7] = 1
    for i in range(MAX_SLICE_NUM):
        p_m[base_ptnum + i * slice_ptnum, base_ptnum + i * slice_ptnum + slice_ptnum, 7] = -i
    
    while is_not_used.all() == False:
        temp_fine = p_m[is_not_used]
        min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)        


    for i in range(MAX_SLICE_NUM):
        idx_in_slice = slice(base_ptnum + i * slice_ptnum,
                             base_ptnum + i * slice_ptnum + slice_ptnum)
        temp_fine = p_m[idx_in_slice]
        tree_gloabl = KDTree(gloabl[:, :3], compact_nodes=False)
        min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)
        pred_nnid[idx_in_slice] = gloabl[min_nnIDx, 6]
        p_m[idx_in_slice, 7] = -i
        pred_dis[idx_in_slice] = min_dis**2
        gloabl = np.r_[gloabl, temp_fine]
    p_m[:base_layer.shape[0], 7] = 1
    pred_dis = np.clip(pred_dis,1,2*MAX_DIS)
    return p_m.astype(int), pred_nnid, pred_dis.astype(int)