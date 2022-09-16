import argparse
import pickle
import numpy as np
import lmdb
import os
import shutil
from tqdm import tqdm

from utils import tok


parser = argparse.ArgumentParser()
# general
parser.add_argument('--maxSeqlen', default=20, type=int)
parser.add_argument('--maxInsts', default=20, type=int) 
parser.add_argument('--maxImgs', default=5, type=int)  

opts = parser.parse_args()


TRAIN_DATA = "data/train_samples.pkl" 
VAL_DATA = "data/val_samples.pkl"
TEST_DATA = "data/test_samples.pkl"

INGR_VOCAB_FILE = "data/ingr_vocab.pkl"

TRAIN_LMDB_PATH = "data/train_lmdb"
VAL_LMDB_PATH = "data/val_lmdb"
TEST_LMDB_PATH = "data/test_lmdb"

DATASETS = [(TRAIN_DATA, TRAIN_LMDB_PATH), (TEST_DATA, TEST_LMDB_PATH), (VAL_DATA, VAL_LMDB_PATH)]

ingr_vocab = pickle.load(open(INGR_VOCAB_FILE, "rb"))


def create_lmdb(dataset_path, lmdb_path):
    if "train" in dataset_path:
        print("Creating Train LMDB")
    elif "test" in dataset_path:
        print("Creating Test LMDB")
    else:
        print("Creating Val LMDB")

    if os.path.isdir(lmdb_path):
        shutil.rmtree(lmdb_path)
    env = lmdb.open(os.path.abspath(lmdb_path), map_size=int(1e11))

    dataset = pickle.load(open(dataset_path, "rb"))

    keys = []

    for id in tqdm(dataset, total=len(dataset)):
        sample = dataset[id]
        title = tok(sample["title"])
        ingredients = sample["ingredients"]
        instructions = [tok(x) for x in sample["instructions"]]
        images = sample["imgs"]

        if len(instructions) >= opts.maxInsts or len(ingredients) < 2 or len(ingredients) >= opts.maxSeqlen:
            continue

        title_word_inds = np.zeros(opts.maxSeqlen)
        for column, w in enumerate(title.split()):
            if column == opts.maxSeqlen:
                break
            try:
                title_word_inds[column] = ingr_vocab[w]
            except:
                title_word_inds[column] = ingr_vocab['<UNK>']

        ingr_vec = np.zeros((opts.maxSeqlen))
        for column, ing in enumerate(ingredients):
            try:
                ingr_vec[column] = ingr_vocab[ing]
            except:
                ingr_vec[column] = ingr_vocab['<UNK>']
        
        insts = np.zeros((opts.maxSeqlen, opts.maxInsts))
        for row, inst_text in enumerate(instructions):
            for column, w in enumerate(inst_text.split()):
                if column == opts.maxSeqlen:
                    break
                try:
                    insts[row, column] = ingr_vocab[w]
                except:
                    insts[row, column] = ingr_vocab['<UNK>']
        
        serialized_sample = pickle.dumps( {'ingrs':ingr_vec,
                                           'intrs_novec': insts,
                                           'imgs':images,
                                           'title_word_inds':title_word_inds} )
        with env.begin(write=True) as txn:
                txn.put('{}'.format(id).encode(), serialized_sample)
        keys.append(id)
    
    pickle.dump(keys, open(os.path.join(lmdb_path, "keys.pkl"), "wb"))
    print("Dataset includes ", len(keys), " samples")
    return len(keys)

for d_path, db_path in DATASETS:
    create_lmdb(d_path, db_path)