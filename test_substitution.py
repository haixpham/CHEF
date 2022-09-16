import torch
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
import json
import pickle
import os
import lmdb
import numpy as np
from datetime import datetime

from eval import load_model, run_model, extract_partition_embeddings
from data_loader import foodSpaceLoader, error_catching_loader
from one_recipe import get_recipe_info_from_ID, batch_recipes, load_recipe_list
from tree_utils import *
from test_utils import find_top_k_both, get_model_signature
from utils import make_dir, PadToSquareResize, get_name, exists
from args import get_parser
from ingr_sub import prepare_recipes_by_ingredient

#IMG_PATH = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/images_bicubic"
#DATA_PATH = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/data_merged_ingrs4"

parser = get_parser()
parser.add_argument("--test-save-dir", default="", type=str)
parser.add_argument("--replace-title", default=1, type=int)
parser.add_argument("--keep-hierarchy", default=1, type=int)
parser.add_argument("--retrieve-split", default="test", type=str)
parser.add_argument("--save-sub-embeddings", default=1, type=int)
parser.add_argument("--ingr-to-replace", default="chicken", type=str)
parser.add_argument("--new-ingr", default="beef", type=str)
opts = parser.parse_args()

model_file_path = opts.test_model_path
model_filename = get_name(model_file_path)

SAVE_JSON_PATH = opts.test_save_dir
if SAVE_JSON_PATH == "":
    SAVE_JSON_PATH = "data/save_test_substitution/" + model_filename
DATA_PATH = opts.data_path
IMG_PATH = opts.img_path

VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))
INGR_VOCAB = pickle.load(open(DATA_PATH + "/ingr_vocab.pkl", "rb"))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


# def get_new_recipe(recipe, ingr_out, ingr_in):
#     return recipe.replace_ingredient(ingr_out, ingr_in, INGR_VOCAB)


def get_new_recipe_batch(recipes, ingr_out, ingr_in, replace_title=True):
    new_recipes = []
    for rp in recipes:
        new_rp = rp.replace_ingredient(ingr_out, ingr_in, INGR_VOCAB, replace_title=replace_title)
        if type(new_rp) is list:
            new_recipes.append(new_rp[0])
        else:
            new_recipes.append(new_rp)
    return new_recipes


def get_embedding_structure_batch(model, opts, recipes):
    use_cuda = not opts.no_cuda
    # get data
    title_data, title_ln, ingr_data, ingr_ln, inst_data, inst_ln, inst_num = batch_recipes(recipes, INGR_VOCAB, max_seq_len=opts.maxSeqlen, use_cuda=use_cuda)
    ingr_masks = None
    if opts.ingrInLayer == "tstsLSTM":
        ingr_masks = get_ingredient_structure_batch(model, ingr_data, ingr_ln, return_embedding=False, use_cuda=use_cuda, copy_to_cpu=False)
    sent_masks = None
    sent_embs = None
    if opts.instInLayer == "tstsLSTM":
        sent_masks, sent_embs = get_sentence_structure_batch(model, inst_data, inst_ln, inst_num, return_embedding=True, use_cuda=use_cuda, copy_to_cpu=False)
    inst_masks = None
    if opts.docInLayer == "tstsLSTM":
        if sent_embs is not None:
            inst_masks = get_instruction_structure_batch(model, sent_embs, inst_num, use_cuda=use_cuda, copy_to_cpu=False)
        else:
            inst_masks = get_instruction_structure_batch_2(model, inst_data, inst_ln, inst_num, use_cuda=use_cuda, copy_to_cpu=False)
    return ingr_masks, sent_masks, inst_masks


def get_embedding_of_batch(model, opts, recipes, ingr_masks=None, sent_masks=None, inst_masks=None):
    use_cuda = not opts.no_cuda
    title_data, title_ln, ingr_data, ingr_ln, inst_data, inst_ln, inst_num = batch_recipes(recipes, INGR_VOCAB, max_seq_len=opts.maxSeqlen, use_cuda=use_cuda)

    # title embedding
    title_out = model.titleNet_(title_data, title_ln, opts)

    # ingredient embedding
    if ingr_masks is not None:
        ingr_out = model.ingrNet_.ingr_LSTM.forward_with_masks(ingr_data, ingr_ln, ingr_masks)
    else:
        ingr_out = model.ingrNet_(ingr_data, ingr_ln, opts)

    # instruction embedding
    doc_encoder = model.instNet_.doc_encoder
    # 4 cases
    if sent_masks is not None and inst_masks is not None:
        # both encoders are tree-lstm
        inst_out = doc_encoder.forward_with_both_masks(inst_data, inst_num, inst_ln, sent_masks, inst_masks)
    elif sent_masks is not None:
        # only sentence encoder is tree-lstm, the other is lstm
        inst_out = doc_encoder.forward_with_masks(inst_data, inst_num, inst_ln, sent_masks)
    elif inst_masks is not None:
        # only instruction encoder is tree-lstm
        inst_out = doc_encoder.forward_with_inst_masks(inst_data, inst_num, inst_ln, inst_masks)
    else:
        # no tree-lstm
        inst_out = doc_encoder.forward(inst_data, inst_num, inst_ln)

    # get the final embedding
    recipe_emb = torch.cat([inst_out, ingr_out, title_out], 1)
    recipe_emb = model.recipe_embedding(recipe_emb)
    recipe_emb = model.align(recipe_emb)
    recipe_emb = model.align_rec(recipe_emb)
    recipe_emb = norm(recipe_emb)

    # return the embedding
    if use_cuda:
        recipe_emb = recipe_emb.cpu()
    return recipe_emb.detach().numpy()



def substitute_batch(model, opts, recipes, ingr_out, ingr_in, keep_hierarchy=True, replace_title=True):
    """

    :param model:
    :param opts:
    :param recipe:
    :param ingr_out: must be a string
    :param ingr_in: must be a string
    :return:
    """
    # create new recipe
    new_recipes = get_new_recipe_batch(recipes, ingr_out, ingr_in, replace_title=replace_title)

    if keep_hierarchy:
        ingr_masks, sent_masks, inst_masks = get_embedding_structure_batch(model, opts, recipes)
    else:
        ingr_masks = None
        sent_masks = None
        inst_masks = None

    # get new embedding
    recipe_embs = get_embedding_of_batch(model, opts, new_recipes, ingr_masks, sent_masks, inst_masks)
    return new_recipes, recipe_embs


def print_metrics(metrics):
    print("Number of retrieved images: {:d} - {:f}\%".format(metrics["img"], float(metrics["img"]) * 100 / metrics["num"]))
    print("Number of retrieved texts: {:d} - {:f}\%".format(metrics["txt"], float(metrics["txt"]) * 100 / metrics["num"]))


def run_substitution(model, partition, retrieve_partition, ingr_out, ingr_in, result_file, recipe_file, batch_size=50, keep_hierarchy=True, replace_title=True, use_cuda=True):
    print("--- testing substitution ---\n")
    print("testing on {:s} split: {:s} -> {:s} retrieved on {:s}".format(partition, ingr_out, ingr_in, retrieve_partition))
    print("\n----------------------------")

    test_data = foodSpaceLoader(opts.img_path,
                            transforms.Compose([
                                PadToSquareResize(resize=256, padding_mode='reflect'),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]),
                            data_path=opts.data_path,
                            partition=partition,
                            loader=error_catching_loader)
    all_recipes = {} # store all matching recipes

    test_file = "data/recipes_of_{:s}_with_{:s}.json".format(partition, ingr_out)
    if not exists(test_file):
        # use batch_size=1
        test_data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)
        test_recipes = prepare_recipes_by_ingredient(test_data_loader, ingr_out, VOCAB_INGR)
        print("Find recipes on {:s} set".format(partition))
    else:
        test_recipes = load_recipe_list(test_file)
        print("Load recipes on {:s} set".format(partition))
    recipes = test_recipes

    for rp in recipes:
        id = rp.recipe_id
        all_recipes[id] = {"matched_img": "", "correct_img": 0, "matched_txt": "", "correct_txt": 0}


    all_metrics = {"img": 0, "txt": 0, "num": len(recipes)}

    #extract embeddings of retrieve_partition
    print("Extract embedding on {:s}".format(retrieve_partition))
    emb_data = extract_partition_embeddings(model, opts, retrieve_partition, batch_size)
    retrieve_lmdb = lmdb.open(os.path.join(DATA_PATH, retrieve_partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

    # get embedding
    print("\n                  test on: {:s}".format(partition))
    print("----------------------------------------")
    # get recipes
    N = len(recipes)

    print("There are {:d} recipes containing \"{:s}\"".format(N, ingr_out))

    recipe_batchs = [recipes[i:i + batch_size] for i in range(0, N, batch_size)]

    if opts.save_sub_embeddings:
        all_new_recipes = []
        all_new_embs = None

    for batch in tqdm(recipe_batchs, total=len(recipe_batchs)):
        new_recipes, new_embs = substitute_batch(model, opts, batch, ingr_out, ingr_in, keep_hierarchy=keep_hierarchy, replace_title=replace_title)
        K = len(batch)
        for k in range(K):
            # loop to retrieve
            # retrieve
            img_IDs, txt_IDs = find_top_k_both(new_embs[k, :], emb_data, k=1)

            # fetch data
            img_recipe = get_recipe_info_from_ID(img_IDs[0], retrieve_partition, retrieve_lmdb, VOCAB_INGR, IMG_PATH)
            txt_recipe = get_recipe_info_from_ID(txt_IDs[0], retrieve_partition, retrieve_lmdb, VOCAB_INGR, IMG_PATH)

            all_recipes[batch[k].recipe_id]["matched_img"] = img_IDs[0]
            all_recipes[batch[k].recipe_id]["matched_txt"] = txt_IDs[0]

            if img_recipe.is_containing(ingr_in):
                all_metrics["img"] += 1
                all_recipes[batch[k].recipe_id]["correct_img"] = 1
            if txt_recipe.is_containing(ingr_in):
                all_metrics["txt"] += 1
                all_recipes[batch[k].recipe_id]["correct_txt"] = 1
        if opts.save_sub_embeddings:
            all_new_recipes.extend(new_recipes)
            if all_new_embs is not None:
                all_new_embs = np.concatenate([all_new_embs, new_embs], axis=0)
            else:
                all_new_embs = new_embs

    print("\n------------------------------------------------------------------")
    print("Substitution metrics on retrieve partition - {:s} - embedding".format(retrieve_partition))
    print_metrics(all_metrics)
    print("------------------------------------------------------------------\n")

    print("Done!")
    json.dump(all_metrics, open(result_file, "w"), indent=4)
    json.dump(all_recipes, open(recipe_file, "w"), indent=4)

    if opts.save_sub_embeddings:
        print("Save substituted embeddings")
        save_file = os.path.join(SAVE_JSON_PATH, "sub_"+partition+"_"+ingr_out+"_"+ingr_in+".npz")
        rec_ids = [rp.recipe_id for rp in new_recipes]
        rec_ids = np.array(rec_ids, dtype=object)
        np.savez(save_file, rec_ids=rec_ids, rec_embs=all_new_embs)


def main():
    keep_hierarchy = opts.keep_hierarchy > 0
    replace_title = opts.replace_title > 0
    use_cuda = not opts.no_cuda
    partition = opts.test_split
    retrieve_partition = opts.retrieve_split
    batch_size = opts.batch_size

    # substitutions = [("chicken", "beef"), ("beef", "chicken"), 
    #                 ("chicken", "fish"), ("fish", "chicken"), 
    #                 ("chicken", "pork"), ("pork", "chicken"),
    #                 ("chicken", "apple"), ("apple", "chicken"),
    #                 ("beef", "fish"), ("fish", "beef"),
    #                 ("beef", "pork"), ("pork", "beef")]
    substitutions = [(opts.ingr_to_replace, opts.new_ingr)]

    model = load_model(opts.test_model_path, opts)
    if model is None:
        raise RuntimeError("model not loaded")
    model.eval()

    print("partition: ", partition, " |retrieve partition: ", retrieve_partition, " |keep_hierarchy: ", keep_hierarchy, " |replace_title: ", replace_title)

    for (ingr_out, ingr_in) in substitutions:
        if keep_hierarchy:
            if replace_title:
                result_file = SAVE_JSON_PATH + "/{:s}-{:s}_substitution_metrics_kph_rpltlt_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                               ingr_in)
                recipe_file = SAVE_JSON_PATH + "/{:s}-{:s}_substitution_recipes_kph_rpttlt_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                               ingr_in)
            else:
                result_file = SAVE_JSON_PATH + "/{:s}-{:s}_substitution_metrics_kph_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                        ingr_in)
                recipe_file = SAVE_JSON_PATH + "/{:s}-{:s}_substitution_recipes_kph_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                        ingr_in)
        else:
            if replace_title:
                result_file = SAVE_JSON_PATH + "/{:s}_substitution_metrics_rpltlt_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                           ingr_in)
                recipe_file = SAVE_JSON_PATH + "/{:s}_substitution_recipes_rpltlt_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out,
                                                                                                           ingr_in)
            else:
                result_file = SAVE_JSON_PATH + "/{:s}_substitution_metrics_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out, ingr_in)
                recipe_file = SAVE_JSON_PATH + "/{:s}_substitution_recipes_{:s}_2_{:s}.json".format(partition, retrieve_partition, ingr_out, ingr_in)

        run_substitution(model, partition, retrieve_partition, 
                        ingr_out=ingr_out, ingr_in=ingr_in,
                        result_file=result_file, recipe_file=recipe_file,
                        batch_size=batch_size,
                        keep_hierarchy=keep_hierarchy, 
                        replace_title=replace_title,
                        use_cuda=use_cuda)

make_dir(SAVE_JSON_PATH)
main()