import nltk
import torch
import numpy as np
import json
from tqdm import tqdm
import pickle
import os
from tree_utils import *

from utils import make_dir
from one_recipe import OneRecipe
from nlp_utils import *


#new_DATA_PATH = DATA_PATH + "_extended"
#SAVE_PATH = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/pruned_embedding"

#INGR_VOCAB = pickle.load(open(new_DATA_PATH+"/ingr_vocab.pkl", "rb"))
#VOCAB_INGR = pickle.load(open(new_DATA_PATH+"/vocab_ingr.pkl", "rb"))

RUN_DEBUG = False

def to_nltk_tree_from_list(selections, ingredients_list, show_order=True, use_normal_tree=False):
    if not selections:
        return None

    #ss = [s[0].max(0)[1] for s in selections] + [0]
    ss = []
    for s in selections:
        x = s[0].max(0)[1]
        ss.append((x))
    ss += [0]
    nodes = [i for i in ingredients_list]
    for k, i in enumerate(ss):
        if show_order:
            if use_normal_tree:
                new_node = nltk.tree.Tree(f'({k + 1})', [nodes[i], nodes[i + 1]])
            else:
                new_node = nltk.tree.ParentedTree(f'({k+1})', [nodes[i], nodes[i+1]])
        else:
            if use_normal_tree:
                new_node = nltk.tree.Tree(f'({k + 1})', [nodes[i], nodes[i + 1]])
            else:
                new_node = nltk.tree.ParentedTree('o', [nodes[i], nodes[i + 1]])
        del nodes[i:i + 2]
        nodes.insert(i, new_node)
    return nodes


def get_mask_from_queue(queue):
    n = len(queue)
    chosen_index = -1
    chosen_label = 0
    for i, item in enumerate(queue):
        if isinstance(item, nltk.tree.Tree):
            item_label = int(item.label()[1:-1])
            if item_label > chosen_label:
                chosen_label = item_label
                chosen_index = i
    if chosen_index < 0:
        return None
    else:
        mask = [0 for i in range(n)]
        mask[chosen_index] = 1
        return (mask, chosen_index)


def convert_tree_to_mask(tree, max_len=None):
    # rebalance the tree first
    if RUN_DEBUG:
        print("Before balancing")
        tree.pretty_print()
    rebalance_tree(tree)
    if RUN_DEBUG:
        print("After balancing")
        tree.pretty_print()
    if not max_len:
        max_len = tree.height()
    # the tree should be fully-binary at this point
    # build the mask
    masks = [] # the root has 1 mask
    cur_len = 0
    queue = []
    for child in tree:
        queue.append(child)
    while len(queue) > 0 or cur_len < max_len:
        ret = get_mask_from_queue(queue)
        if not ret:
            break
        mask, chosen_index = ret
        item_to_remove = queue.pop(chosen_index)
        iii = chosen_index
        for child in item_to_remove:
            queue.insert(iii, child)
            iii += 1
        masks.append(mask)
        cur_len += 1

    ret_masks = masks[::-1]
    return ret_masks



def remove_highest_leaf(tree_root):
    leaves, max_depth = find_highest_leaves(tree_root)
    removed_leaf = None
    if len(leaves) > 1:
        # delete the last one
        removed_leaf = leaves[-1]
        remove_leaf(tree_root, leaves[-1])
    elif len(leaves) == 1:
        # delete it
        removed_leaf = leaves[0]
        remove_leaf(tree_root, leaves[0])
    else:
        return False, None

    return True, removed_leaf


def prune_tree_remove_last_k_leaves(select_masks, ingredient_list, k=5):
    tree_root = to_nltk_tree_from_list(select_masks, ingredient_list)[0]
    if RUN_DEBUG:
        print("Original tree")
        tree_root.pretty_print()
    # remove leaves:
    removed_leaves = []
    for i in range(k):
        ret, removed_leaf = remove_highest_leaf(tree_root)
        if not ret:
            break
        removed_leaves.append(removed_leaf)
    return tree_root, removed_leaves


def prune_tree_keep_first_k_leaves(select_masks, ingredient_list, k=5):
    return prune_tree_remove_last_k_leaves(select_masks, ingredient_list, len(ingredient_list)-k)


def prune_tree_keep_k_depth(select_masks, ingredient_list, k=5):
    tree_root = to_nltk_tree_from_list(select_masks, ingredient_list)[0]
    if RUN_DEBUG:
        print("Original tree")
        tree_root.pretty_print()
    leaves = get_leaves(tree_root)
    to_remove = []
    for le in leaves:
        if leaves[le] > k:
            to_remove.append(le)
    for le in to_remove:
        remove_leaf(tree_root, le)
    return tree_root, to_remove


def prune_tree(select_masks, ingredient_list, k=5, mode="REMOVE_LAST"):
    """
    mode: REMOVE_LAST, KEEP_FIRST, KEEP_DEPTH
    return:
        - new list
        - removed list // should check this list, if it is empty, means no change
        - new select_masks
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ingredient_list) <= k+2: # keep at least 2 ingredients
        return ingredient_list, [], select_masks
    if mode == "REMOVE_LAST":
        tree, removed = prune_tree_remove_last_k_leaves(select_masks, ingredient_list, k)
    elif mode == "KEEP_FIRST":
        tree, removed = prune_tree_keep_first_k_leaves(select_masks, ingredient_list, k)
    elif mode == "KEEP_DEPTH":
        tree, removed = prune_tree_keep_k_depth(select_masks, ingredient_list, k)
    else:
        raise ValueError("unsupported prune mode")

    if len(removed) > 0:
        #tree.pretty_print()
        new_ingredient_list = []
        for x in ingredient_list:
            if x not in removed:
                new_ingredient_list.append(x)
        max_len = len(new_ingredient_list)-2
        new_masks = convert_tree_to_mask(tree, max_len)
        new_masks = [torch.FloatTensor([x]) for x in new_masks]
        return new_ingredient_list, removed, new_masks
    else:
        return ingredient_list, [], select_masks


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def get_embedding_of_recipe(model, opts, recipe, INGR_VOCAB, ingr_masks=None, sent_masks=None, inst_masks=None, use_cuda=True):
    ingrs, ingrs_ln, title, title_ln, insts, insts_ln, insts_num = recipe.get_torch_data(INGR_VOCAB)
    if use_cuda:
        ingrs = ingrs.cuda()
        ingrs_ln = ingrs_ln.cuda()
        title = title.cuda()
        title_ln = title_ln.cuda()
        insts = insts.cuda()
        insts_ln = insts_ln.cuda()
        insts_num = insts_num.cuda()
    # title embedding
    title_out = model.titleNet_(title, title_ln, opts)

    # ingredient embedding
    if ingr_masks is not None:
        ingr_out = model.ingrNet_.ingr_LSTM.forward_with_masks(ingrs, ingrs_ln, ingr_masks)
    else:
        ingr_out = model.ingrNet_(ingrs, ingrs_ln, opts)

    # instruction embedding
    doc_encoder = model.instNet_.doc_encoder
    # 4 cases
    if sent_masks is not None and inst_masks is not None:
        # both encoders are tree-lstm
        inst_out = doc_encoder.forward_with_both_masks(insts, insts_num, insts_ln, sent_masks, inst_masks)
    elif sent_masks is not None:
        # only sentence encoder is tree-lstm, the other is lstm
        inst_out = doc_encoder.forward_with_masks(insts, insts_num, insts_ln, sent_masks)
    elif inst_masks is not None:
        # only instruction encoder is tree-lstm
        inst_out = doc_encoder.forward_with_inst_masks(insts, insts_num, insts_ln, inst_masks)
    else:
        # no tree-lstm
        inst_out = doc_encoder.forward(insts, insts_num, insts_ln)

    # get the final embedding
    recipe_emb = torch.cat([inst_out, ingr_out, title_out], 1)
    recipe_emb = model.recipe_embedding(recipe_emb)
    recipe_emb = model.align(recipe_emb)
    recipe_emb = model.align_rec(recipe_emb)
    recipe_emb = norm(recipe_emb)

    if use_cuda:
        recipe_emb = recipe_emb.cpu()
    recipe_emb = recipe_emb.squeeze(0).detach().numpy()
    return recipe_emb


def prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, max_samples=-1, mode="REMOVE_LAST", k=1, use_cuda=True):
    if opts.ingrInLayer != "tstsLSTM":
        print("Non-structured ingredient encoder, nothing to do. Exit!")
        return

    model.eval()

    # use batch_size=1 always - structure analysis function does not work with batch
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)

    ret_IDs = []
    ret_embs = []

    cnt = 0

    save_recipe_path = os.path.join(save_path, "pruned_recipes")
    make_dir(save_recipe_path)

    for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get recipe
        recipe = OneRecipe(input, VOCAB_INGR, id)
        # make directory for recipe:
        recipe_path = os.path.join(save_recipe_path, recipe.get_recipe_filename())
        make_dir(recipe_path)
        recipe.save_recipe(recipe_path)
        # save ingredient structure
        select_masks = get_ingredient_structure(model, recipe.ingrs, INGR_VOCAB, use_cuda=use_cuda)
        # get new structure
        try:
            new_ingr_list, removed, new_masks = prune_tree(select_masks, recipe.ingrs, k, mode)
            if RUN_DEBUG:
                print(new_ingr_list)
                print(new_masks)
            if len(removed) > 0:
                recipe.replace_all_ingredients(new_ingr_list)
                for x in removed:
                    recipe.replace_ingredient(x, "<UNK>", INGR_VOCAB, replace_title=True, replace_ingredient=False)
                # calculate new embedding
                if use_cuda:
                    new_masks_cuda = [x.cuda() for x in new_masks]
                else:
                    new_masks_cuda = new_masks
                emb = get_embedding_of_recipe(model, opts, recipe, INGR_VOCAB, ingr_masks=new_masks_cuda, use_cuda=use_cuda)
                try:
                    tree_root = to_nltk_tree_from_list(new_masks, recipe.ingrs, show_order=True, use_normal_tree=True)[0]
                    # save tree
                    if tree_root:
                        if RUN_DEBUG:
                            tree_root.pretty_print()
                        image_file = os.path.join(recipe_path, "ingredient_tree.svg")
                        text_file = os.path.join(recipe_path, "ingredient_latex.txt")
                        string_file = os.path.join(recipe_path, "ingredient_tree_string.txt")
                        save_tree_to_svg_and_latex(tree_root, image_file, text_file)
                        save_tree_string(tree_root, string_file)

                    ret_IDs.append(id[0])
                    ret_embs.append(emb)
                    cnt += 1
                except:
                    if RUN_DEBUG:
                        print("something wrong with " + id[0] + " when build new tree")
        except:
            if RUN_DEBUG:
                print("cannot get prune tree of " + id[0])

        if max_samples > 0 and cnt == max_samples:
            break
    if len(ret_IDs) > 0:
        print("Saving " + str(len(ret_IDs)))
        # save the embedding
        ret_embs = np.stack(ret_embs, axis=0)
        npz_file = os.path.join(save_path, mode + "_" + str(k) + ".npz")
        np.savez(npz_file, rec_ids=ret_IDs, rec_embeds=ret_embs)
    else:
        print("Nothing changed!")
    return ret_IDs, ret_embs


def test_prune_remove_last(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=True):
    print("\t REMOVE_LAST 1")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="REMOVE_LAST", k=1)
    print("\t REMOVE_LAST 2")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="REMOVE_LAST", k=2)
    print("\t REMOVE_LAST 3")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="REMOVE_LAST", k=3)
    print("\t REMOVE_LAST 4")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="REMOVE_LAST", k=4)


def test_prune_keep_first(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=True):
    print("\t KEEP_FIRST 3")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_FIRST", k=3)
    print("\t KEEP_FIRST 4")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_FIRST", k=4)
    print("\t KEEP_FIRST 5")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_FIRST", k=5)
    print("\t KEEP_FIRST 6")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_FIRST", k=6)


def test_prune_keep_depth(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=True):
    print("\t KEEP_DEPTH 3")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_DEPTH", k=3)
    print("\t KEEP_DEPTH 4")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_DEPTH", k=4)
    print("\t KEEP_DEPTH 5")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_DEPTH", k=5)
    print("\t KEEP_DEPTH 6")
    prune_structure(model, opts, dataset, INGR_VOCAB, VOCAB_INGR, save_path, use_cuda=use_cuda, mode="KEEP_DEPTH", k=6)

def load_pruned_embedding(npz_data_path):
    embs = np.load(npz_data_path)
    rec_IDs = list(embs["rec_ids"])
    rec_IDs_d = {}
    for i, id in enumerate(rec_IDs):
        rec_IDs_d[id] = i
    if embs["rec_embeds"].ndim == 1:
        rec_embs = np.reshape(embs["rec_embeds"], (-1, 1024))
        assert rec_embs.shape[0] == len(rec_IDs)
    elif embs["rec_embeds"].ndim == 2:
        rec_embs = embs["rec_embeds"]
    else:
        raise ValueError("incorrect embedding dimension")
    return rec_IDs, rec_IDs_d, rec_embs
