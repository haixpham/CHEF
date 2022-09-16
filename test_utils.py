import numpy as np
import random
from utils import get_name

def get_random_splits(rec_IDs, N_folds=10, fold_size=1000):
    N = len(rec_IDs)
    st = random.getstate()
    random.seed(1234)

    assert N_folds >= 1
    if fold_size <= 0:
        fold_size = len(rec_IDs) // N_folds

    splits = []
    for i in range(N_folds):
        idxs = random.sample(range(N), fold_size)
        splits.append(idxs)

    random.setstate(st)
    return splits


def rank_only(img_embs, rec_embs, mode="i2t"):
    assert mode in ["i2t", "t2i"], "unsupported cross modal ranking"
    assert img_embs.shape == rec_embs.shape
    N = img_embs.shape[0]
    ranks = []
    recall = {1: 0.0, 5: 0.0, 10: 0.0}
    if N <= 30000:
        if mode == "i2t":
            sims = np.dot(img_embs, rec_embs.T)
        else:
            sims = np.dot(rec_embs, img_embs.T)
        for i in range(N):
            # sort in descending order
            sorting = np.argsort(sims[i,:])[::-1].tolist()
            # where this index 'i' is in the sorted list
            pos = sorting.index(i)
            if pos == 0:
                recall[1] += 1
            if pos < 5:
                recall[5] += 1
            if pos < 10:
                recall[10] += 1
            ranks.append(pos+1)
    else:
        for i in range(N):
            if mode == "i2t":
                sims = np.dot(img_embs[i,:], rec_embs.T)
            else:
                sims = np.dot(rec_embs[i,:], img_embs.T)
            sorting = np.argsort(sims)[::-1].tolist()
            # where this index 'i' is in the sorted list
            pos = sorting.index(i)
            if pos == 0:
                recall[1] += 1
            if pos < 5:
                recall[5] += 1
            if pos < 10:
                recall[10] += 1
            ranks.append(pos+1)
    medRank = np.median(ranks)
    for k in recall:
        recall[k] = recall[k] / N
    dcg = np.array([1/np.log2(r+1) for r in ranks]).mean()

    return medRank, recall, dcg, ranks


def rank(img_embs, rec_embs, rec_IDs, mode="i2t", N_folds=10, fold_size=1000):
    assert img_embs.shape == rec_embs.shape
    assert img_embs.shape[0] == len(rec_IDs)
    assert N_folds >= 1
    if fold_size <= 0:
        fold_size = len(rec_IDs) // N_folds
    st = random.getstate()
    random.seed(1234)

    N = len(rec_IDs)

    global_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    global_rank = []
    all_ranks = []
    global_dcg = []

    for i in range(N_folds):
        # sampling fold_size samples
        idxs = random.sample(range(N), fold_size)
        img_emb_sub = img_embs[idxs,:]
        rec_emb_sub = rec_embs[idxs,:]

        medRank, recall, dcg, ranks = rank_only(img_emb_sub, rec_emb_sub, mode)

        global_rank.append(medRank)
        all_ranks.append(np.mean(ranks))
        global_dcg.append(dcg)
        for k in global_recall:
            global_recall[k] += recall[k]

    for k in global_recall:
        global_recall[k] = global_recall[k] / N_folds

    random.setstate(st)
    return np.average(global_rank), global_recall, np.mean(all_ranks), np.mean(global_dcg)


def rank_2(img_embs, rec_embs, rec_IDs, mode="i2t", splits=None):
    assert img_embs.shape == rec_embs.shape
    assert img_embs.shape[0] == len(rec_IDs)
    assert splits is not None
    N_folds = len(splits)
    #fold_size = len(splits[0])

    N = len(rec_IDs)

    global_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    global_rank = []
    all_ranks = []
    global_dcg = []

    for i, idxs in enumerate(splits):
        # sampling fold_size samples
        img_emb_sub = img_embs[idxs,:]
        rec_emb_sub = rec_embs[idxs,:]

        medRank, recall, dcg, ranks = rank_only(img_emb_sub, rec_emb_sub, mode)

        global_rank.append(medRank)
        all_ranks.append(np.mean(ranks))
        global_dcg.append(dcg)
        for k in global_recall:
            global_recall[k] += recall[k]

    for k in global_recall:
        global_recall[k] = global_recall[k] / N_folds

    return np.average(global_rank), global_recall, np.mean(all_ranks), np.mean(global_dcg)


def rank_3(img_embs, rec_embs, img_splits, rec_splits, mode="i2t"):
    assert img_splits is not None and rec_splits is not None
    assert len(img_splits) == len(rec_splits)
    N_folds = len(img_splits)

    global_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    global_rank = []
    all_ranks = []
    global_dcg = []

    for i in range(N_folds):
        # sampling fold_size samples
        img_emb_sub = img_embs[img_splits[i],:]
        rec_emb_sub = rec_embs[rec_splits[i],:]

        medRank, recall, dcg, ranks = rank_only(img_emb_sub, rec_emb_sub, mode)

        global_rank.append(medRank)
        all_ranks.append(np.mean(ranks))
        global_dcg.append(dcg)
        for k in global_recall:
            global_recall[k] += recall[k]

    for k in global_recall:
        global_recall[k] = global_recall[k] / N_folds

    return np.average(global_rank), global_recall, np.mean(all_ranks), np.mean(global_dcg)


def find_top_k(query_emb, embs, ids, k=1, emb_pos=None):
    d = np.dot(query_emb, embs.T)
    sort_ind = np.argsort(d)[::-1].tolist()
    nns  = sort_ind[:k]
    IDs = [ids[j] for j in nns]
    if emb_pos is None:
        return IDs
    else:
        rank = sort_ind.index(emb_pos) + 1
        return rank, IDs


def find_top_k_both(query_emb, emb_data, k=1):
    img_embs = emb_data["img_embeds"]
    txt_embs = emb_data["rec_embeds"]
    ids = emb_data["rec_ids"]
    img_IDs = find_top_k(query_emb, img_embs, ids, k)
    txt_IDs = find_top_k(query_emb, txt_embs, ids, k)
    return img_IDs, txt_IDs


def find_top_k_image(query_emb, emb_data, k=1):
    img_embs = emb_data["img_embeds"]
    ids = emb_data["rec_ids"]
    img_IDs = find_top_k(query_emb, img_embs, ids, k)
    return img_IDs


def find_top_k_text(query_emb, emb_data, k=1):
    txt_embs = emb_data["rec_embeds"]
    ids = emb_data["rec_ids"]
    txt_IDs = find_top_k(query_emb, txt_embs, ids, k)
    return txt_IDs


def get_model_signature(opts):
    name = get_name(opts.test_model_path)
    name = "{:s}_{:s}_{:s}_{:s}_{:s}_{:s}".format(name, opts.ingrInLayer, opts.instInLayer, opts.docInLayer, "ingrAtt" if opts.ingrAtt else "noIngrAtt", "instAtt" if opts.instAtt else "noIngrAtt")
    return name