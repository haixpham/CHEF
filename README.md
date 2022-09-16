# CHEF: Cross-modal Hierarchial Embedding for Food Domain Retrieval

This repository holds the code for the work presented in: Hai. X Pham, Ricardo Guerrero, Jiatong Li and Vladimir Pavlovic. [CHEF: Cross-modal Hierarchial Embedding for Food Domain Retrieval](https://arxiv.org/abs/2102.02547) (Appeared in AAAI 2021)

---
### Python dependency requirements

- Anaconda python enviroment (Python 3.6 or above), which already includes: numpy, nltk and most other necessary libraries
- Pytorch & torchvision (latest - currently 1.10)
- gensim (version >= 4)
- lmdb
- OpenCV

---
### Using the code

**Note: this code has been refactored, including data preparation. Combined with latest Pytorch (1.10), these changes result in improved performance.**

0. Data preparation

All data will be stored in **"data"** by default.

- Download original Recipe1M data from http://pic2recipe.csail.mit.edu/

Put the 3 following files: **layer1.json**, **layer2.json**, **det_ingrs.json** into **"data"** folder.

The images should be stored somewhere with the following structures:

```
image_root (ideally put inside "data", declared as default in the command options script "args.py")
    |_____ train
             |___ subfolders
    |_____ test
             |___ subfolders
    |_____ val
             |___ subfolders
```

- Generate data & word2vec
```
python prepare_data.py
```
This step creates the following files inside "data" folder:

| File | Description | 
|---|---|
| train_samples.pkl      | training set recipes                     |
| test_samples.pkl       | test set recipes                         |
| val_samples.pkl        | validation set recipes                   |
| w2v_tokenized_text.txt | word2vec training data                   |
| w2v.bin                | the trained Word2Vec model, using Gensim |
| vocab.bin              | word2vec vectors, C format               |
| ingr_vocab.pkl         | text to vocab mapping                    |
| vocab_ingr.pkl         | vocab to text mapping                    |


- Create LMDB
```
python create_lmdb.py
```
This step creates 3 subfolders in **"data"**: **train_lmdb**, **test_lmdb**, **val_lmdb**, corresponding to training set, test set and validation set, respectively.

**NOTE: All of the data described above can be found on the Cnode machine: "106.1.153.40:/home/nfs/hai.xuanpham/CHEF_repo/data"**

1. Train model
```
python train.py --gpu 0,1,2,3 --batch-size 160 --ingrInLayer [RNN/dense/tstsLSTM] --instInLayer [LSTM/tstsLSTM] --docInLayer [LSTM/tstsLSTM] --img-path [image root path] --data-path [data root path]
```
where **data root path** is the place where data is stored (which is **"data"** in the default setting), and **image root path** is the root folder of all images as shown in the above hierarchy.
**ingrInLayer** can be one among [RNN/dense/tstsLSTM], likewise for **instInLayer** and **docInLayer**. The log and checkpoints of this training session are stored in **"tensorboard/timestamp"** where *timestamp* is when training started. It's essential to train different models for all combinations of these three options in order to recreate the tables in paper. The saved models can be found in **"tensorboard/timestamp/models"**

Users can try different input options declared in **args.py**. Some examples:

| Model | Train command options |
|---|---|
| T+T+T | --gpu 0,1,2,3 --batch-size 160 --ingrInLayer tstsLSTM --instInLayer tstsLSTM --docInLayer tstsLSTM |
| T+T+L | --gpu 0,1,2,3 --batch-size 160 --ingrInLayer tstsLSTM --instInLayer tstsLSTM --docInLayer LSTM |
| T+L+L | --gpu 0,1,2,3 --batch-size 160 --ingrInLayer tstsLSTM --instInLayer LSTM --docInLayer LSTM |
| G+L+L | --gpu 0,1,2,3 --batch-size 160 --ingrInLayer RNN --instInLayer LSTM --docInLayer LSTM |

**NOTE: Pretrained models can be found on the Cnode machine: "106.1.153.40:/home/nfs/hai.xuanpham/CHEF_repo/models"**

2. Test retrieval
In order to carry out retrieval test on a trained model, use the following command:
```
python test_retrieval.py --test-model-path [trained model file path] --ingrInLayer [RNN/dense/tstsLSTM] --instInLayer [LSTM/tstsLSTM] --docInLayer [LSTM/tstsLSTM] --test-split [data split, default="test] --test-N-folds [N, default=10] --test-K [K, default=1000]
```
where **trained model file path** is the path to the model file (such as "tensorboard/20211203-171011__train/models/model_BEST_REC_e008_v-10.200_cr-1.0507.pth.tar"). **test split* can be either "test" or "val" (or "train" if you so wish, but it will take 6 times longer). 

By default, the test will be 10 folds (test-N-folds) retrieval rankings of 1000 samples (test-K) each time.

**ingrInLayer**, **instInLayer** and **docInLayer** should be specified to load the model weights correctly.

Some results (image-to-recipe retrieval on the "test" set) are given below.

| Model | MedR | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| T+T+T | 1.2 | 50.8 | 78.9 | 86 |
| T+T+L | - | - | - | - |
| T+L+L | - | - | - | - |
| G+L+L | - | - | - | - |

3. Extract embedding structure
To extract the tree structures of ingredients/sentences/whole instruction, run the following script:
```
python test_structure.py --test-model-path [trained model file path] --ingrInLayer [RNN/dense/tstsLSTM] --instInLayer [LSTM/tstsLSTM] --docInLayer [LSTM/tstsLSTM] --test-save-dir [save path] --test-split [data split, default="test]
```
Upon completion, the recipes will be saved inside **save path**. Each recipe includes its text and inferred tree structures. This script also reports the *main action word* detection performance as described in the paper.

4. Test ingredient pruning
To evaluation retrieval performance after performing ingredient tree pruning, execute the following script:
```
python test_pruning_retrieval.py --test-model-path [trained model file path] --ingrInLayer [RNN/dense/tstsLSTM] --instInLayer [LSTM/tstsLSTM] --docInLayer [LSTM/tstsLSTM] --test-split [data split, default="test] --test-save-dir [save path] --mode [prune mode]
```
**prune mode** is among REMOVE_LAST (default - reported in paper), KEEP_FIRST, KEEP_DEPTH.

The pruned recipes as well as their (pruned) embeddings will be stored in **save path**. The new ranking metrics are reported as well as stored in `"test_{:s}_summary_{:d}_folds_{:d}_fold_size.json".format(mode, opts.test_N_folds, opts.test_K))`

5. Test ingredient subsitution
To perform ingredient substitution task on a particular data split, run the following script:
```
python test_substitution.py --test-model-path [trained model file path] --ingrInLayer [RNN/dense/tstsLSTM] --instInLayer [LSTM/tstsLSTM] --docInLayer [LSTM/tstsLSTM] --test-split [data split, default="test] --test-save-dir [save path] --ingr-to-replace chicken --new-ingr beef
```
where **ingr-to-replace** is the ingredient to be subsituted (detault="chicken") and **new-ingr** is the ingredient it is replaced with (default="beef").