import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no_cuda', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gpu', default='0', type=str)

    # data
    parser.add_argument('--img-path', default='data/images/')
    parser.add_argument('--data-path', default='data/')
    parser.add_argument('--workers', default=4, type=int)

    # FoodSpaceNet model
    parser.add_argument('--embDim', default=1024, type=int)          # Dim of FoodSpace
    parser.add_argument('--rnnDim', default=300, type=int)           # Dim of RNN (x2 as biRNN)
    parser.add_argument('--imfeatDim', default=2048, type=int)       # Dim from resnet50 before last fc layer
    parser.add_argument('--w2vDim', default=300, type=int)           # Dim from word2vec model
    parser.add_argument("--w2vInit", type=str2bool, nargs='?', const=True, default=True, help="Initialize word embeddings with w2v model?")
    parser.add_argument('--ingrAtt', type=str2bool, nargs='?', const=True, default=False) # Attention over ingredient words?
    parser.add_argument('--instAtt', type=str2bool, nargs='?', const=True, default=False) # Attention over instructions words?
    parser.add_argument('--ingrInLayer', default='tstsLSTM',type=str)  #default: RNN                        # 'dense', 'RNN' or 'none' or 'tstsLSTM', if 'dense' or 'none' ingrAtt is always used
    parser.add_argument('--instInLayer', default='tstsLSTM',type=str)   #default: LSTM                        # 'LSTM', 'tstsLSTM'
    parser.add_argument('--docInLayer', default='tstsLSTM', type=str)   #default: LSTM
    parser.add_argument('--maxSeqlen', default=20, type=int)         # Used when building LMDB
    parser.add_argument('--maxInsts', default=20, type=int)          # Used when building LMDB
    parser.add_argument('--maxImgs', default=5, type=int)            # Used when building LMDB
    parser.add_argument('--visualmodel', default='', type=str)  # Model of visual space: 'simple_14', 'simple_28', 'simple_28-14' or empty

    # training
    parser.add_argument('--batch-size', default=40, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=720, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1,type=int)
    parser.add_argument("--freeVision", type=str2bool, nargs='?', const=True, default=False, help="Train resnet parameters?")
    parser.add_argument("--freeRecipe", type=str2bool, nargs='?', const=True, default=True, help="Train non-resnet parameters?")
    parser.add_argument("--freeEmbs", type=str2bool, nargs='?', const=True, default=False, help="Train non-resnet parameters?")
    parser.add_argument("--ohem", type=str2bool, nargs='?', const=True, default=True, help="Train with hard mining?")
    parser.add_argument("--intraClassLoss", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--patience', default='20', type=str)
    parser.add_argument('--patienceLR', default='40', type=str)
    parser.add_argument('--lossDistance', default='cosine', type=str) #
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--warmup', default=0, type=int)

    # testing
    parser.add_argument('--embtype', default='image', type=str)
    parser.add_argument('--test-K', default=1000, type=int) # 0 = full set test
    parser.add_argument('--test-N-folds', default=10, type=int)
    parser.add_argument("--test-model-path", default="tensorboard/20211213-170824__train/models/model_BEST_REC_e067_v-1.000_cr-2.1847.pth.tar", type=str)
    parser.add_argument("--test-split", default="test", type=str)


    return parser




