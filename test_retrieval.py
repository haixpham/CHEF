import torch
import torchvision.transforms as transforms

from eval import load_model, validate, run_model, rank, extract_partition_embeddings
from data_loader import foodSpaceLoader, error_catching_loader
from args import get_parser
from utils import PadToSquareResize

parser = get_parser()
opts = parser.parse_args()

partition = opts.test_split.lower()
if partition not in ["test", "val", "train"]:
    raise ValueError("Test split not specified")

# get model
model = load_model(opts.test_model_path, opts)
if not model:
    raise RuntimeError("Model not loaded")

# preparing the valitadion loader
print(" *** Testing on {} split *** ".format(partition))

# run test
print("Test {} folds of {} samples each".format(opts.test_N_folds, opts.test_K))
if opts.test_K <= 0:
    #test full set
    opts.test_K = 0
    opts.test_N_folds = 1
    print("Test on full set")


print(" ------------ Encoding ------------ ")
emb_data = extract_partition_embeddings(model, opts, opts.test_split, opts.batch_size)
im_embs = emb_data["img_embeds"]
re_embs = emb_data["rec_embeds"]
rec_ids = emb_data["rec_ids"]

print(" ------------ Test image2recipe retrieval ------------ ")
opts.embtype = "image"
medR, recall, meanR, meanDCG = rank(opts, im_embs, re_embs, rec_ids)
print('\t* Val medR {medR:.4f}\tRecall {recall}\tVal meanR {meanR:.4f}\tVal meanDCG {meanDCG:.4f}'.format(medR=medR, recall=recall, meanR=meanR, meanDCG=meanDCG))

print(" ------------ Test recipe2image retrieval ------------ ")
opts.embtype = "recipe"
medR, recall, meanR, meanDCG = rank(opts, im_embs, re_embs, rec_ids)
print('\t* Val medR {medR:.4f}\tRecall {recall}\tVal meanR {meanR:.4f}\tVal meanDCG {meanDCG:.4f}'.format(medR=medR, recall=recall, meanR=meanR, meanDCG=meanDCG))