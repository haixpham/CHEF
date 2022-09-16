import numpy as np
import json
import pickle
import os
from torchvision.transforms import transforms

from args import get_parser
from utils import exists, make_dir, exists, PadToSquareResize
from tree_pruning import prune_structure
from test_utils import get_random_splits, rank_3
from eval import load_model, extract_partition_embeddings
from data_loader import foodSpaceLoader, error_catching_loader

parser = get_parser()
parser.add_argument("--test-save-dir", default="save_test_pruning", type=str)
parser.add_argument("--prune-mode", default="REMOVE_LAST", type=str, help="Pick one option among REMOVE_LAST / KEEP_FIRST / KEEP_DEPTH")
opts = parser.parse_args()


DATA_PATH = opts.data_path
IMG_PATH = opts.img_path

VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))
INGR_VOCAB = pickle.load(open(DATA_PATH + "/ingr_vocab.pkl", "rb"))

def test_rank(model, partition, save_path, prune_mode, prune_K, N_folds=10, fold_size=1000):
    use_cuda = not opts.no_cuda
    # load full embedding data
    full_embs = extract_partition_embeddings(model, opts, partition, opts.batch_size)
    full_IDs_d = {}
    for i, id in enumerate(full_embs["rec_ids"]):
        full_IDs_d[id] = i

    # load pruned embeddings
    pruned_emb_file = save_path + "/" + prune_mode + "_" + str(prune_K) + ".npz"
    if exists(pruned_emb_file):
        pruned_embs = np.load(pruned_emb_file)
        pruned_IDs = pruned_embs["rec_ids"]
        pruned_embs = pruned_embs["rec_embeds"]
    else:
        dataset = foodSpaceLoader(opts.img_path,
                                transforms.Compose([
                                    PadToSquareResize(resize=256, padding_mode='reflect'),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
                                data_path=opts.data_path,
                                partition=partition,
                                loader=error_catching_loader)
        pruned_IDs, pruned_embs = prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, mode=prune_mode, k=prune_K, use_cuda=use_cuda)
    pruned_IDs_d = {}
    for i, id in enumerate(pruned_IDs):
        pruned_IDs_d[id] = i

    if len(pruned_IDs) < fold_size:
        print("sample set too small. Stop!")
        return None

    # generate rank test splits
    pruned_splits = get_random_splits(pruned_IDs, N_folds, fold_size)
    full_splits = []
    for sps in pruned_splits:
        full_split = []
        for idx in sps:
            id = pruned_IDs[idx]
            full_idx = full_IDs_d[id]
            full_split.append(full_idx)
        full_splits.append(full_split)
    
    # calculate retrieval
    print("i -> t")
    medR_1, recall_1, avgR_1, dcg_1 = rank_3(full_embs["img_embeds"], pruned_embs, img_splits=full_splits, rec_splits=pruned_splits, mode="i2t")
    print("t -> i")
    medR_2, recall_2, avgR_2, dcg_2 = rank_3(full_embs["img_embeds"], pruned_embs, img_splits=full_splits, rec_splits=pruned_splits, mode="t2i")
    ret = {}
    ret["i2t"] = [medR_1, recall_1, avgR_1, dcg_1]
    ret["t2i"] = [medR_2, recall_2, avgR_2, dcg_2]

    print("Test result for mode = {} K = {}")
    print(ret)
    return ret


def main():
    model = load_model(opts.test_model_path, opts)
    if model is None:
        raise RuntimeError("model not loaded")
    model.eval()

    save_path = opts.test_save_dir
    make_dir(save_path)

    mode = opts.prune_mode
    if mode not in ["REMOVE_LAST", "KEEP_FIRST", "KEEP_DEPTH"]:
        raise ValueError("invalid prune mode")

    partition = opts.test_split
    results = {}
    
    if mode == "REMOVE_LAST":
        for K in [1, 2, 3, 4]:
            ret = test_rank(model, partition, save_path, mode, K, opts.test_N_folds, opts.test_K)
            if ret is not None:
                results[K] = ret
    else:
        for K in [3, 4, 5, 6]:
            ret = test_rank(model, partition, save_path, mode, K, opts.test_N_folds, opts.test_K)
            if ret is not None:
                results[K] = ret
            
    #save
    json_file = os.path.join(save_path, "test_{:s}_summary_{:d}_folds_{:d}_fold_size.json".format(mode, opts.test_N_folds, opts.test_K))
    with open(json_file, "w") as fp:
        json.dump(results, fp, indent=4)

if __name__ == "__main__":
    main()