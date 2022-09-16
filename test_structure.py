import torch
import torchvision.transforms as transforms
import os
import pickle
import json
from tqdm import tqdm

from eval import load_model
from data_loader import foodSpaceLoader, error_catching_loader
from args import get_parser
from utils import PadToSquareResize, make_dir, get_parent_path, get_name
from one_recipe import OneRecipe
from tree_utils import *
from nlp_utils import *

parser = get_parser()
parser.add_argument("--test-save-dir", default="save_test_structure", type=str)
opts = parser.parse_args()

partition = opts.test_split.lower()
if partition not in ["test", "val", "train"]:
    raise ValueError("Test split not specified")

if not opts.test_save_dir:
    opts.test_save_dir = "data/test_structure"


# get model
model = load_model(opts.test_model_path, opts)
if not model:
    raise RuntimeError("Model not loaded")
model.eval()

# preparing the valitadion loader
print(" *** Testing on {} split *** ".format(partition))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_data = foodSpaceLoader(opts.img_path,
                        transforms.Compose([
                            PadToSquareResize(resize=256, padding_mode='reflect'),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize]),
                        data_path=opts.data_path,
                        partition=partition,
                        loader=error_catching_loader)
# use batch_size=1 always - structure analysis function does not work with batch
data_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1, shuffle=False,
    # sampler=SubsetSequentialSampler(np.arange(1000)),
    num_workers=0, pin_memory=True)
print('Validation loader prepared.')


VOCAB_INGR = pickle.load(open(opts.data_path + "/vocab_ingr.pkl", "rb"))
INGR_VOCAB = pickle.load(open(opts.data_path + "/ingr_vocab.pkl", "rb"))
USE_CUDA = not opts.no_cuda


def extract_structure(model, partition="val", max_samples=-1):
    if opts.ingrInLayer != "tstsLSTM" and opts.instInLayer != "tstsLSTM" and opts.docInLayer != "tstsLSTM":
        print("Non-structure, nothing to do. Exit!")
        return

    model.eval()

    save_path = opts.test_save_dir
    make_dir(save_path)

    if opts.instInLayer == "tstsLSTM":
        count_sentences = 0
        count_correct_verbs = 0
        count_words = 0

    for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get recipe
        recipe = OneRecipe(input, VOCAB_INGR, id)
        # make directory for recipe:
        recipe_path = save_path + "/" + recipe.get_recipe_filename()
        make_dir(recipe_path)
        recipe.save_recipe(recipe_path)
        # save ingredient structure
        if opts.ingrInLayer == "tstsLSTM":
            select_masks = get_ingredient_structure(model, recipe.ingrs, INGR_VOCAB, use_cuda=USE_CUDA)
            tree_root = to_nltk_tree_from_list(select_masks, recipe.ingrs, show_order=True)
            # save tree
            if tree_root:
                image_file = recipe_path + "/ingredient_tree.svg"
                text_file = recipe_path + "/ingredient_latex.txt"
                string_file = recipe_path + "/ingredient_tree_string.txt"
                save_tree_to_svg_and_latex(tree_root, image_file, text_file)
                save_tree_string(tree_root, string_file)
        # save each sentence structure
        if opts.instInLayer == "tstsLSTM":
            for k, inst in enumerate(recipe.intrs):
                select_masks = get_sentence_structure(model, inst, INGR_VOCAB, use_cuda=USE_CUDA)
                tree_root = to_nltk_tree(select_masks, inst, show_order=True)
                # save
                if tree_root:
                    image_file = recipe_path + f"/instruction_{k+1}_tree.svg"
                    text_file = recipe_path + f"/instruction_{k+1}_latex.txt"
                    string_file = recipe_path + f"/instruction_{k + 1}_tree_string.txt"
                    save_tree_to_svg_and_latex(tree_root, image_file, text_file)
                    save_tree_string(tree_root, string_file)

                    count_sentences += 1
                    count_words += len(tree_root.leaves())
                    # find the leaf closest to root
                    leaves, depth = find_lowest_leaves(tree_root)
                    for le in leaves:
                        if le not in [".", ","]:
                            if is_verb(le):
                                count_correct_verbs += 1
        # save full instruction structure
        if opts.docInLayer == "tstsLSTM":
            select_masks = get_instruction_structure(model, recipe.intrs, INGR_VOCAB, use_cuda=USE_CUDA)
            tree_root = to_nltk_tree_from_list(select_masks, [f"{k+1}" for k in range(len(recipe.intrs))], show_order=True)

            # save tree
            if tree_root:
                image_file = recipe_path + "/full_instruction_tree.svg"
                text_file = recipe_path + "/full_instruction_latex.txt"
                string_file = recipe_path + "/full_instruction_tree_string.txt"
                save_tree_to_svg_and_latex(tree_root, image_file, text_file)
                save_tree_string(tree_root, string_file)

        if max_samples > 0 and i == max_samples - 1:
            break
    
    if opts.instInLayer == "tstsLSTM":
        print(count_correct_verbs, count_words, count_sentences)
        metrics = {"word_count": count_words, "sentence_count": count_sentences, "correct_verb_count": count_correct_verbs}
        save_file = save_path + "/count_verb_in_sentences.json"
        json.dump(metrics, open(save_file, "w"), indent=4)


def extract_attention(model, partition="val", max_samples=-1):
    if not (opts.ingrAtt or opts.instAtt):
        print("Non-attention, nothing to do. Exit!")
        return

    model.eval()

    save_path = opts.test_save_dir
    make_dir(save_path)

    for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get recipe
        recipe = OneRecipe(input, VOCAB_INGR, id, IMG_PATH=opts.img_path)
        # make directory for recipe:
        recipe_path = save_path + "/" + recipe.get_recipe_filename()
        make_dir(recipe_path)
        recipe.save_recipe(recipe_path)
        # save ingredient structure
        if opts.ingrAtt:
            h, attn = get_ingredient_attention(model, recipe.ingrs, INGR_VOCAB, opts, use_cuda=USE_CUDA)
            # save JSON
            text_file = recipe_path + "/ingredient_attention_json.txt"
            json.dump(attn, open(text_file, "w"))

        # save each sentence structure
        if opts.instAtt:
            h, attn = get_instruction_attention(model, recipe.intrs, INGR_VOCAB, use_cuda=USE_CUDA)
            text_file = recipe_path + "/instruction_attention_json.txt"
            json.dump(attn, open(text_file, "w"))

        if max_samples > 0 and i == max_samples - 1:
            break


if opts.ingrInLayer == "tstsLSTM" or opts.instInLayer == "tstsLSTM" and opts.docInLayer == "tstsLSTM":
    extract_structure(model, partition)

if opts.ingrAtt or opts.instAtt:
    extract_attention(model, partition)
