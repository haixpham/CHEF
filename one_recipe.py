import torch
import cv2
import numpy as np
import copy
import json
import pickle
import os

from nlp_utils import tokenize


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class OneRecipe(object):
    def __init__(self, input, vocab, recipe_id=""):
        self.img = input[0].squeeze(0).permute(1, 2, 0).numpy() #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.is_image_opencv = False
        self.title = self.__get_title(input, vocab)
        self.ingrs = self.__get_ingredients(input, vocab)
        self.intrs = self.__get_instructions(input, vocab)
        self.recipe_id = recipe_id[0]
        #self.recipe_class = None

    @classmethod
    def from_lmdb_sample(cls, recipe_id, sample, img_path, vocab):
        new_obj = object.__new__(OneRecipe)
        new_obj.recipe_id = recipe_id
        # load image
        new_obj.img = cv2.imread(img_path)
        new_obj.is_image_opencv = True
        # load text
        # title
        title = sample['title_word_inds'].astype('int')
        new_obj.title = " ".join([vocab[word] for word in title if word > 0])
        # ingredients
        ingrs = sample["ingrs"].astype("int")
        new_obj.ingrs = [vocab[ingr] for ingr in ingrs if ingr > 0]
        # instructions
        intrs = sample["intrs_novec"].astype("int")
        h, w = intrs.shape
        sentences = []
        for row in range(h):
            if np.sum(intrs[row,:]) == 0:
                break
            sentence = [vocab[intrs[row, column]]for column in range(w) if intrs[row, column] > 0]
            sentences.append(" ".join(sentence))
        new_obj.intrs = sentences
        return new_obj

    def to_json_string(self):
        data = {"id": self.recipe_id}
        data["title"] = self.title
        data["ingredient"] = self.ingrs
        data["instruction"] = self.intrs
        return json.dumps(data)

    @classmethod
    def from_json_string(cls, json_string):
        data = json.loads(json_string)
        new_obj = object.__new__(OneRecipe)
        new_obj.recipe_id = data["id"]
        new_obj.title = data["title"]
        new_obj.ingrs = data["ingredient"]
        new_obj.intrs = data["instruction"]
        new_obj.img = None
        new_obj.is_image_opencv = False
        return new_obj

    def __get_ingredients(self, input, vocab):
        ingrs = input[1].squeeze(0).numpy()
        ingr_ln = input[2].squeeze(0).numpy().item()
        ingrs = [vocab[ingrs[i]] for i in range(ingr_ln)]
        return ingrs

    def __get_title(self, input, vocab):
        title_words = input[3].squeeze(0).numpy()
        title_words_ln = input[4].squeeze(0).numpy().item()
        title = [vocab[title_words[i]] for i in range(title_words_ln)]
        return " ".join(title)

    def __get_instructions(self, input, vocab):
        intrs_novec = input[5].squeeze(0).numpy()
        intrs_novec_ln = input[6].squeeze(0).numpy()
        intrs_nover_num = input[7].squeeze(0).numpy().item()
        intrs = []
        for i in range(intrs_nover_num):
            word_num = intrs_novec_ln[i]
            intr = [vocab[intrs_novec[i][j]] for j in range(word_num)]
            intr = " ".join(intr)
            intrs.append(intr)
        return intrs

    def replace_all_ingredients(self, new_ingrs):
        self.ingrs = new_ingrs

    def get_recipe(self):
        txt = self.title
        txt = txt + "\nIngredients:\n"
        for i in self.ingrs:
            txt += " - " + i + "\n"
        txt += "Instructions:\n"
        for k, i in enumerate(self.intrs):
            txt += f" - {k+1} " + i + "\n"
        return txt

    def get_opencv_image(self):
        if self.is_image_opencv:
            return self.img
        else:
            # un-normalize image:
            img = np.copy(self.img)
            for c in range(3):
                img[:,:,c] = self.img[:,:,c] * (STD[c]*255) + MEAN[c]* 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

    def show_image(self, window_name=""):
        img = self.get_opencv_image()
        cv2.imshow(window_name if window_name else self.title, img)

    def get_recipe_filename(self):
        ret = str(self.recipe_id) + "-" + self.title.replace(".", "").replace("/", "").replace("\\", "").replace("*", "").replace("\"", "").replace(":", "").replace("?", "").replace("|", "").replace("<", "").replace(">", "").replace(" ", "_")
        return ret

    def save_recipe(self, dir_path):
        # save image
        img = self.get_opencv_image()
        cv2.imwrite(dir_path + "/image.jpg", img)
        # save recipe
        txt = self.get_recipe()
        with open(dir_path + "/full_recipe.txt", "w") as f:
            f.write(txt)

    @staticmethod
    def compare_string(str1, str2):
        # if str1 == str2, or str1 is substring is str2, or str2 is substring of str1
        return str1 == str2 or str1.find(str2) >= 0 or str2.find(str1) >= 0

    @staticmethod
    def compare_string_2(str1, str2):
        # if str1 == str2, or str1 is substring is str2, or str2 is substring of str1
        if OneRecipe.compare_string(str1, str2):
            return True

        # more meticulous, in case either ingredient is composite
        if str1.find("_") >= 0:
            str1s = str1.split("_")
        else:
            str1s = [str1]
        if str2.find("_") >= 0:
            str2s = str2.split("_")
        else:
            str2s = [str2]

        if len(str1s) > 1 or len(str2s) > 1:
            for s1 in str1s:
                for s2 in str2s:
                    if OneRecipe.compare_string(s1, s2):
                        return True
        # no match
        return False


    def __is_containing(self, ingr):
        for i in self.ingrs:
            if OneRecipe.compare_string(i, ingr):
                return True
        return False

    def is_containing(self, ingr, all_inclusive=False):
        if type(ingr) is str:
            return self.__is_containing(ingr)
        elif type(ingr) is list or type(ingr) is tuple:
            for i in ingr:
                is_match = self.__is_containing(i)
                if is_match and not all_inclusive:
                    return True
                elif not is_match and all_inclusive:
                    return False
            if all_inclusive:
                return True
            else:
                return False
        else:
            raise ValueError("unsupported input")

    def __replace_ingredient_in_title(self, other, ingr_out, ingr_in, ingr_vocab):
        changed = False
        words = tokenize(self.title)
        for i in range(len(words)):
            if OneRecipe.compare_string(words[i], ingr_out):
                t = words[i].replace(ingr_out, ingr_in)
                if t not in ingr_vocab:
                    t = ingr_in
                words[i] = t
                changed = True
        if changed:
            other.title = " ".join(words)
        return changed

    def __replace_one_ingredient(self, other, ingr_out, ingr_in, ingr_vocab, replace_title=True, replace_ingredient=True):
        # replace word in ingredient:
        changed = False
        if replace_title:
            changed = self.__replace_ingredient_in_title(other, ingr_out, ingr_in, ingr_vocab)
        if replace_ingredient:
            for i in range(len(other.ingrs)):
                if OneRecipe.compare_string_2(ingr_out, other.ingrs[i]):
                    # a match
                    other.ingrs[i] = ingr_in
                    changed = True
        for i in range(len(other.intrs)):
            # for each instruction
            words = tokenize(other.intrs[i])
            for k in range(len(words)):
                if OneRecipe.compare_string_2(words[k], ingr_out):
                    words[k] = ingr_in
                    changed = True
            other.intrs[i] = " ".join(words)
        return changed

    def replace_ingredient(self, ingr_out, ingr_in, ingr_vocab, replace_title=True, replace_ingredient=True):
        if type(ingr_out) is not type(ingr_in):
            return None
        if type(ingr_out) is str:
            other = copy.deepcopy(self)
            if self.__replace_one_ingredient(other, ingr_out, ingr_in, ingr_vocab, replace_title, replace_ingredient):
                return other
            else:
                return None
        elif type(ingr_out) is list:
            if len(ingr_in) != len(ingr_out):
                return None
            changed = False
            others = []
            for i in range(len(ingr_out)):
                other = copy.deepcopy(self)
                changed = self.__replace_one_ingredient(other, ingr_out[i], ingr_in[i], ingr_vocab, replace_title, replace_ingredient)
                others.append(other)
            if changed:
                return others
            else:
                return None
        else:
            return None

    def get_torch_data(self, ingr_vocab, max_seq_len=0):
        if max_seq_len < 1:
            # convert title -> torch tensor
            tokens = tokenize(self.title)
            words = [ingr_vocab[token] for token in tokens]
            title = torch.LongTensor([words])
            title_ln = (title > 0).sum(1)

            # convert ingredient -> torch tensor
            words = [ingr_vocab[i] for i in self.ingrs]
            ingrs = torch.LongTensor([words])
            ingrs_ln = (ingrs > 0).sum(1)

            # convert instructions -> torch tensor
            intrs = [tokenize(i) for i in self.intrs]
            for r in range(len(intrs)):
                for c in range(len(intrs[r])):
                    intrs[r][c] = ingr_vocab[intrs[r][c]]
            sents = []
            for i in range(len(intrs)):
                sent = torch.LongTensor(intrs[i])
                sents.append(sent)
            insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
            insts = insts.unsqueeze(0)
            insts_ln = (insts > 0).sum(2)
            insts_num = (insts_ln > 0).sum(1)
        else:
            title_array = np.zeros((1, max_seq_len), dtype=np.int)
            ingr_array = np.zeros((1, max_seq_len), dtype=np.int)
            inst_array = np.zeros((1, max_seq_len, max_seq_len), dtype=np.int)

            toks = tokenize(self.title)
            for j, tok in enumerate(toks):
                try:
                    title_array[1, j] = ingr_vocab[tok]
                except:
                    title_array[1, j] = ingr_vocab["<UNK>"]
            for j, tok in enumerate(self.ingrs):
                try:
                    ingr_array[1, j] = ingr_vocab[tok]
                except:
                    ingr_array[1, j] = ingr_vocab["<UNK>"]
            for j in range(len(self.intrs)):
                toks = tokenize(self.intrs[j])
                for k in range(len(toks)):
                    try:
                        inst_array[1, j, k] = ingr_vocab[toks[k]]
                    except:
                        inst_array[1, j, k] = ingr_vocab["<UNK>"]
            title = torch.LongTensor(title_array)
            title_ln = (title > 0).sum(1)
            ingrs = torch.LongTensor(ingr_array)
            ingrs_ln = (ingrs > 0).sum(1)
            insts = torch.LongTensor(inst_array)
            insts_ln = (insts > 0).sum(2)
            insts_num = (insts_ln > 0).sum(1)

        return [ingrs, ingrs_ln, title, title_ln, insts, insts_ln, insts_num]

    def get_torch_title(self, ingr_vocab, max_seq_len=0):
        # convert title -> torch tensor
        if max_seq_len < 1:
            tokens = tokenize(self.title)
            words = [ingr_vocab[token] for token in tokens]
            title = torch.LongTensor([words])
        else:
            title_array = np.zeros((1, max_seq_len), dtype=np.int)
            toks = tokenize(self.title)
            for j, tok in enumerate(toks):
                try:
                    title_array[1, j] = ingr_vocab[tok]
                except:
                    title_array[1, j] = ingr_vocab["<UNK>"]
            title = torch.LongTensor(title_array)
        title_ln = (title > 0).sum(1)
        return title, title_ln

    def get_torch_sentences(self, ingr_vocab, create_batch_axis=True, max_seq_len=0):
        N = len(self.intrs)
        intrs = [tokenize(i) for i in self.intrs]
        for r in range(N):
            for c in range(len(intrs[r])):
                intrs[r][c] = ingr_vocab[intrs[r][c]]

        if max_seq_len < 1:
            sents = []
            for i in range(N):
                sent = torch.LongTensor(intrs[i])
                sents.append(sent)
            insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
        else:
            inst_array = np.zeros((N, max_seq_len), dtype=np.int)
            for r in range(N):
                for c in range(len(intrs[r])):
                    try:
                        inst_array[r, c] = ingr_vocab[intrs[r][c]]
                    except:
                        inst_array[r, c] = ingr_vocab["<UNK>"]
            inst = torch.LongTensor(inst_array)

        if create_batch_axis:
            insts = insts.unsqueeze(0)
            insts_ln = (insts > 0).sum(2)
        else:
            insts_ln = (insts > 0).sum(1)
        return insts, insts_ln

    def __str__(self):
        return self.get_recipe()

    def __repr__(self):
        return self.__str__()


# def get_data_loader(img_path, data_path, partition="test", batch_size=100):
#     data = foodSpaceLoader(img_path,
#                                 transforms.Compose([
#                                     PadToSquareResize(resize=256, padding_mode='reflect'),
#                                     transforms.CenterCrop(224),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=MEAN, std=STD)]),
#                                 data_path=data_path,
#                                 partition=partition,
#                                 loader=error_catching_loader,
#                                 maxImgs=1)
#     data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
#     return data_loader


# def count_dish_with_ingredient(partition, ingrs, VOCAB_INGR):
#     metrics = {}
#     for ingr in ingrs:
#         metrics[ingr] = 0
#     data_loader = get_data_loader(IMG_PATH, DATA_PATH, partition=partition, batch_size=1)
#     for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
#         recipe = OneRecipe(input, VOCAB_INGR, id)
#         for ingr in ingrs:
#             if recipe.is_containing(ingr):
#                 metrics[ingr] += 1
#     return metrics


# def get_recipe_with_ingredients(partition, ingrs, VOCAB_INGR, max_count=-1):
#     ret = {}
#     for ingr in ingrs:
#         ret[ingr] = []
#     data_loader = get_data_loader(IMG_PATH, DATA_PATH, partition=partition, batch_size=1)
#     for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
#         recipe = OneRecipe(input, VOCAB_INGR, id)
#         for ingr in ingrs:
#             if recipe.is_containing(ingr):
#                 ret[ingr].append(recipe)
#         cur_len = sum([len(ret[key]) for key in ret])
#         if max_count > 0 and cur_len == max_count:
#             break
#     return ret


# def load_merged_ingeredients():
#     filename = DATA_PATH + "/list_of_merged_ingredients.txt"
#     with open(filename) as f:
#         lines = [line.rstrip("\n").strip() for line in f]
#     return lines


# def test_data_count():
#     # INGR_VOCAB = pickle.load(open(DATA_PATH + "/ingr_vocab.pkl", "rb"))
#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     ingrs = ["chicken", "beef", "fish"]
#     metrics = count_dish_with_ingredient(partition="train", ingrs=ingrs, VOCAB_INGR=VOCAB_INGR)
#     print("In training set:")
#     print(metrics)


# LMDB = {}
# LMDB["train"] = None
# LMDB["val"] = None
# LMDB["test"] = None


def get_recipe_info_from_ID(recipe_id, partition, data_lmdb, VOCAB_INGR, IMG_PATH):
    encoded_id = recipe_id.encode()
    with data_lmdb.begin(write=False) as txn:
        serialized_sample = txn.get(encoded_id)
    try:
        sample = pickle.loads(serialized_sample)
    except:
        # backward compatible
        sample = pickle.loads(serialized_sample, encoding="latin1")
    # image_path = [sample["imgs"][0]["id"][i] for i in range(4)]
    # image_path = os.path.join(*image_path)
    # image_path = os.path.join(IMG_PATH, partition, image_path, sample['imgs'][0]["id"])
    imgs = sample['imgs']
    img_name = imgs[0]
    if isinstance(img_name, dict):
        # backward compatible
        img_name = img_name["id"]
        first_4_chars = [x for x in img_name[:4]]
        img_path = [IMG_PATH, partition] + first_4_chars + [img_name]
        img_path = os.path.join(*img_path)
    else:
        img_path = os.path.join(IMG_PATH, img_name)
    recipe = OneRecipe.from_lmdb_sample(recipe_id, sample, img_path, VOCAB_INGR)
    return recipe
    

def batch_recipes(recipes, ingr_vocab, max_seq_len=20, use_cuda=True):
    N = len(recipes)
    max_sent_num = 0
    for rp in recipes:
        if len(rp.intrs) > max_sent_num:
            max_sent_num = len(rp.intrs)

    title_array = np.zeros((N, max_seq_len), dtype=np.int)
    ingr_array = np.zeros((N, max_seq_len), dtype=np.int)
    inst_array = np.zeros((N, max_sent_num, max_seq_len), dtype=np.int)

    for i, rp in enumerate(recipes):
        toks = tokenize(rp.title)
        for j, tok in enumerate(toks):
            try:
                title_array[i, j] = ingr_vocab[tok]
            except:
                title_array[i, j] = ingr_vocab["<UNK>"]
        for j, tok in enumerate(rp.ingrs):
            try:
                ingr_array[i, j] = ingr_vocab[tok]
            except:
                ingr_array[i, j] = ingr_vocab["<UNK>"]
        for j in range(len(rp.intrs)):
            toks = tokenize(rp.intrs[j])
            for k in range(len(toks)):
                try:
                    inst_array[i, j, k] = ingr_vocab[toks[k]]
                except:
                    inst_array[i, j, k] = ingr_vocab["<UNK>"]

    if use_cuda:
        title_data = torch.cuda.LongTensor(title_array)
        ingr_data = torch.cuda.LongTensor(ingr_array)
        inst_data = torch.cuda.LongTensor(inst_array)
    else:
        title_data = torch.LongTensor(title_array)
        ingr_data = torch.LongTensor(ingr_array)
        inst_data = torch.LongTensor(inst_array)
    title_ln = (title_data > 0).sum(1)
    ingr_ln = (ingr_data > 0).sum(1)
    inst_ln = (inst_data > 0).sum(2)
    inst_num = (inst_ln > 0).sum(1)
    return title_data, title_ln, ingr_data, ingr_ln, inst_data, inst_ln, inst_num


def save_recipe_list(recipe_list, json_filename):
    data = {}
    for rp in recipe_list:
        data[rp.recipe_id] = rp.to_json_string()
    json.dump(data, open(json_filename, "w"))


def load_recipe_list(json_filename):
    data = json.load(open(json_filename))
    recipes = [OneRecipe.from_json_string(data[i]) for i in data.keys()]
    return recipes


# def prepare_recipes_by_ingredient(partition, ingr, VOCAB_INGR):
#     recipes = get_recipe_with_ingredients(partition, [ingr], VOCAB_INGR, max_count=-1)[ingr]
#     save_file = "./recipes_of_{:s}_with_{:s}.json".format(partition, ingr)
#     save_recipe_list(recipes, save_file)


# def main():
#     #test_data_count()
#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     for partition in ["val", "test"]:
#         print(partition)
#         for ingr in ["apple", "apples"]: #["chicken", "beef", "fish", "pork", "spaghetti", "shrimp"]:
#             print("-------- " + ingr + " ---------")
#             prepare_recipes_by_ingredient(partition, ingr, VOCAB_INGR)

#     print("Done!")


# def generate_recipes_from_match_ID():
#     # load the json file:
#     #json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_beef_20191112-174222.json"
#     json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_apples_20191115-134711.json"
#     data = json.load(open(json_file))
#     model_name = "20191022-181011"
#     test_set = data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe/" + model_name
#     make_dir(save_dir)

#     for id_key in test_set:
#         if test_set[id_key]["test"]["correct_img"] == 1 and test_set[id_key]["test"]["correct_txt"] == 1:
#             print("Found: ", id_key)
#             # save it
#             img_id = test_set[id_key]["test"]["matched_img"]
#             txt_id = test_set[id_key]["test"]["matched_txt"]

#             original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#             img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#             txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#             new_save_dir = save_dir + "/" + id_key
#             make_dir(new_save_dir)
#             new_save_img_dir = new_save_dir + "/img_" + img_id
#             new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#             make_dir(new_save_img_dir)
#             make_dir(new_save_txt_dir)

#             original_recipe.save_recipe(new_save_dir)
#             img_recipe.save_recipe(new_save_img_dir)
#             txt_recipe.save_recipe(new_save_txt_dir)

#     print("Done!")


# def generate_recipes_from_match_ID_c2b_only_one_match():
#     # load the json file:
#     json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_beef_20191112-174222.json"
#     data = json.load(open(json_file))
#     model_name = "20191020-161423"
#     test_set = data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe_" + model_name + "_c2b_only_one_match"
#     make_dir(save_dir)

#     for id_key in test_set:
#         if test_set[id_key]["test"]["correct_img"] != test_set[id_key]["test"]["correct_txt"]:
#             print("Found: ", id_key)
#             # save it
#             img_id = test_set[id_key]["test"]["matched_img"]
#             txt_id = test_set[id_key]["test"]["matched_txt"]

#             original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#             img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#             txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#             new_save_dir = save_dir + "/" + id_key
#             make_dir(new_save_dir)
#             if test_set[id_key]["test"]["correct_img"] == 1:
#                 new_save_img_dir = new_save_dir + "/correct_img_" + img_id
#             else:
#                 new_save_img_dir = new_save_dir + "/img_" + img_id
#             if test_set[id_key]["test"]["correct_txt"] == 1:
#                 new_save_txt_dir = new_save_dir + "/correct_txt_" + txt_id
#             else:
#                 new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#             make_dir(new_save_img_dir)
#             make_dir(new_save_txt_dir)

#             original_recipe.save_recipe(new_save_dir)
#             img_recipe.save_recipe(new_save_img_dir)
#             txt_recipe.save_recipe(new_save_txt_dir)

#     print("Done!")


# def generate_recipes_from_match_ID_c2b_no_match():
#     # load the json file:
#     json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_beef_20191112-174222.json"
#     data = json.load(open(json_file))
#     model_name = "20191020-161423"
#     test_set = data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe_" + model_name + "_c2b_no_match"
#     make_dir(save_dir)

#     for id_key in test_set:
#         if test_set[id_key]["test"]["correct_img"] == 0 and test_set[id_key]["test"]["correct_txt"] == 0:
#             print("Found: ", id_key)
#             # save it
#             img_id = test_set[id_key]["test"]["matched_img"]
#             txt_id = test_set[id_key]["test"]["matched_txt"]

#             original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#             img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#             txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#             new_save_dir = save_dir + "/" + id_key
#             make_dir(new_save_dir)
#             new_save_img_dir = new_save_dir + "/img_" + img_id
#             new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#             make_dir(new_save_img_dir)
#             make_dir(new_save_txt_dir)

#             original_recipe.save_recipe(new_save_dir)
#             img_recipe.save_recipe(new_save_img_dir)
#             txt_recipe.save_recipe(new_save_txt_dir)

#     print("Done!")


# def generate_recipes_from_match_ID_c2a_only_one_match():
#     # load the json file:
#     json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_apples_20191115-134711.json"
#     data = json.load(open(json_file))
#     model_name = "20191022-181011"
#     test_set = data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe_" + model_name + "_c2a_only_one_match"
#     make_dir(save_dir)

#     for id_key in test_set:
#         if test_set[id_key]["test"]["correct_img"] != test_set[id_key]["test"]["correct_txt"]:
#             print("Found: ", id_key)
#             # save it
#             img_id = test_set[id_key]["test"]["matched_img"]
#             txt_id = test_set[id_key]["test"]["matched_txt"]

#             original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#             img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#             txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#             new_save_dir = save_dir + "/" + id_key
#             make_dir(new_save_dir)
#             if test_set[id_key]["test"]["correct_img"] == 1:
#                 new_save_img_dir = new_save_dir + "/correct_img_" + img_id
#             else:
#                 new_save_img_dir = new_save_dir + "/img_" + img_id
#             if test_set[id_key]["test"]["correct_txt"] == 1:
#                 new_save_txt_dir = new_save_dir + "/correct_txt_" + txt_id
#             else:
#                 new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#             make_dir(new_save_img_dir)
#             make_dir(new_save_txt_dir)

#             original_recipe.save_recipe(new_save_dir)
#             img_recipe.save_recipe(new_save_img_dir)
#             txt_recipe.save_recipe(new_save_txt_dir)

#     print("Done!")


# def generate_recipes_from_match_ID_c2a_no_match():
#     # load the json file:
#     json_file = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_apples_20191115-134711.json"
#     data = json.load(open(json_file))
#     model_name = "20191022-181011"
#     test_set = data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe_" + model_name + "_c2a_no_match"
#     make_dir(save_dir)

#     for id_key in test_set:
#         if test_set[id_key]["test"]["correct_img"] == 0 and test_set[id_key]["test"]["correct_txt"] == 0:
#             print("Found: ", id_key)
#             # save it
#             img_id = test_set[id_key]["test"]["matched_img"]
#             txt_id = test_set[id_key]["test"]["matched_txt"]

#             original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#             img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#             txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#             new_save_dir = save_dir + "/" + id_key
#             make_dir(new_save_dir)
#             new_save_img_dir = new_save_dir + "/img_" + img_id
#             new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#             make_dir(new_save_img_dir)
#             make_dir(new_save_txt_dir)

#             original_recipe.save_recipe(new_save_dir)
#             img_recipe.save_recipe(new_save_img_dir)
#             txt_recipe.save_recipe(new_save_txt_dir)

#     print("Done!")


# def generate_recipes_from_match_ID_c2b2c():
#     """
#     Generate full circle chicken -> beef -> chicken
#     """
#     # load the json file:
#     json_file_c2b = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_chicken_2_beef_20191112-174222.json"
#     json_file_b2c = "/home/SERILOCAL/hai.xuanpham/Data/im2recipe/data/substitution_recipes_kph_rpttlt_beef_2_chicken_20191113-095030.json"
#     c2b_data = json.load(open(json_file_c2b))
#     b2c_data = json.load(open(json_file_b2c))
#     model_name = "20191020-161423"
#     c2b_test_set = c2b_data[model_name]["test"]
#     b2c_test_set = b2c_data[model_name]["test"]

#     VOCAB_INGR = pickle.load(open(DATA_PATH + "/vocab_ingr.pkl", "rb"))

#     save_dir = "./save_recipe_" + model_name + "_c2b2c_full_match"
#     make_dir(save_dir)

#     for id_key in c2b_test_set:
#         if c2b_test_set[id_key]["test"]["correct_img"] == 1 and c2b_test_set[id_key]["test"]["correct_txt"] == 1:
#             img_id = c2b_test_set[id_key]["test"]["matched_img"]
#             txt_id = c2b_test_set[id_key]["test"]["matched_txt"]

#             if b2c_test_set[img_id]["test"]["correct_img"] == 1 and b2c_test_set[img_id]["test"]["correct_txt"] == 1 and \
#                 b2c_test_set[txt_id]["test"]["correct_img"] == 1 and b2c_test_set[txt_id]["test"]["correct_txt"] == 1:
#                 print("Found: ", id_key)
#                 # save it

#                 original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#                 img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#                 txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#                 c2b2c_img_img_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_img"], "test", VOCAB_INGR)
#                 c2b2c_img_txt_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_txt"], "test", VOCAB_INGR)

#                 c2b2c_txt_img_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_img"], "test", VOCAB_INGR)
#                 c2b2c_txt_txt_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_txt"], "test", VOCAB_INGR)


#                 new_save_dir = save_dir + "/" + id_key
#                 make_dir(new_save_dir)
#                 original_recipe.save_recipe(new_save_dir)

#                 new_save_img_dir = new_save_dir + "/img_" + img_id
#                 new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#                 make_dir(new_save_img_dir)
#                 make_dir(new_save_txt_dir)
#                 img_recipe.save_recipe(new_save_img_dir)
#                 txt_recipe.save_recipe(new_save_txt_dir)

#                 new_save_img_img_dir = new_save_img_dir + "/img_" + b2c_test_set[img_id]["test"]["matched_img"]
#                 new_save_img_txt_dir = new_save_img_dir + "/txt_" + b2c_test_set[img_id]["test"]["matched_txt"]
#                 new_save_txt_img_dir = new_save_txt_dir + "/img_" + b2c_test_set[txt_id]["test"]["matched_img"]
#                 new_save_txt_txt_dir = new_save_txt_dir + "/txt_" + b2c_test_set[txt_id]["test"]["matched_txt"]
#                 make_dir(new_save_img_img_dir)
#                 make_dir(new_save_img_txt_dir)
#                 make_dir(new_save_txt_img_dir)
#                 make_dir(new_save_txt_txt_dir)

#                 c2b2c_img_img_recipe.save_recipe(new_save_img_img_dir)
#                 c2b2c_img_txt_recipe.save_recipe(new_save_img_txt_dir)
#                 c2b2c_txt_img_recipe.save_recipe(new_save_txt_img_dir)
#                 c2b2c_txt_txt_recipe.save_recipe(new_save_txt_txt_dir)

#     # -------------------------------------------------------------------------------------
#     save_dir = "./save_recipe_" + model_name + "_c2b_full_match_b2c_img_full_match"
#     make_dir(save_dir)

#     for id_key in c2b_test_set:
#         if c2b_test_set[id_key]["test"]["correct_img"] == 1 and c2b_test_set[id_key]["test"]["correct_txt"] == 1:
#             img_id = c2b_test_set[id_key]["test"]["matched_img"]
#             txt_id = c2b_test_set[id_key]["test"]["matched_txt"]

#             if b2c_test_set[img_id]["test"]["correct_img"] == 1 and b2c_test_set[img_id]["test"]["correct_txt"] == 1 and \
#                     (b2c_test_set[txt_id]["test"]["correct_img"] == 0 or b2c_test_set[txt_id]["test"]["correct_txt"] == 0):
#                 print("Found: ", id_key)
#                 # save it

#                 original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#                 img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#                 txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#                 c2b2c_img_img_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_img"], "test",
#                                                                VOCAB_INGR)
#                 c2b2c_img_txt_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_txt"], "test",
#                                                                VOCAB_INGR)

#                 c2b2c_txt_img_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_img"], "test",
#                                                                VOCAB_INGR)
#                 c2b2c_txt_txt_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_txt"], "test",
#                                                                VOCAB_INGR)

#                 new_save_dir = save_dir + "/" + id_key
#                 make_dir(new_save_dir)
#                 original_recipe.save_recipe(new_save_dir)

#                 new_save_img_dir = new_save_dir + "/img_" + img_id
#                 new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#                 make_dir(new_save_img_dir)
#                 make_dir(new_save_txt_dir)
#                 img_recipe.save_recipe(new_save_img_dir)
#                 txt_recipe.save_recipe(new_save_txt_dir)

#                 new_save_img_img_dir = new_save_img_dir + "/img_" + b2c_test_set[img_id]["test"]["matched_img"]
#                 new_save_img_txt_dir = new_save_img_dir + "/txt_" + b2c_test_set[img_id]["test"]["matched_txt"]
#                 new_save_txt_img_dir = new_save_txt_dir + "/img_" + b2c_test_set[txt_id]["test"]["matched_img"]
#                 new_save_txt_txt_dir = new_save_txt_dir + "/txt_" + b2c_test_set[txt_id]["test"]["matched_txt"]
#                 make_dir(new_save_img_img_dir)
#                 make_dir(new_save_img_txt_dir)
#                 make_dir(new_save_txt_img_dir)
#                 make_dir(new_save_txt_txt_dir)

#                 c2b2c_img_img_recipe.save_recipe(new_save_img_img_dir)
#                 c2b2c_img_txt_recipe.save_recipe(new_save_img_txt_dir)
#                 c2b2c_txt_img_recipe.save_recipe(new_save_txt_img_dir)
#                 c2b2c_txt_txt_recipe.save_recipe(new_save_txt_txt_dir)
#     # -------------------------------------------------------------------------------------
#     save_dir = "./save_recipe_" + model_name + "_c2b_full_match_b2c_txt_full_match"
#     make_dir(save_dir)

#     for id_key in c2b_test_set:
#         if c2b_test_set[id_key]["test"]["correct_img"] == 1 and c2b_test_set[id_key]["test"]["correct_txt"] == 1:
#             img_id = c2b_test_set[id_key]["test"]["matched_img"]
#             txt_id = c2b_test_set[id_key]["test"]["matched_txt"]

#             if (b2c_test_set[img_id]["test"]["correct_img"] == 0 or b2c_test_set[img_id]["test"]["correct_txt"] == 0) and \
#                     b2c_test_set[txt_id]["test"]["correct_img"] == 1 and b2c_test_set[txt_id]["test"]["correct_txt"] == 1:
#                 print("Found: ", id_key)
#                 # save it

#                 original_recipe = get_recipe_info_from_ID(id_key, "test", VOCAB_INGR)
#                 img_recipe = get_recipe_info_from_ID(img_id, "test", VOCAB_INGR)
#                 txt_recipe = get_recipe_info_from_ID(txt_id, "test", VOCAB_INGR)

#                 c2b2c_img_img_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_img"], "test",
#                                                                VOCAB_INGR)
#                 c2b2c_img_txt_recipe = get_recipe_info_from_ID(b2c_test_set[img_id]["test"]["matched_txt"], "test",
#                                                                VOCAB_INGR)

#                 c2b2c_txt_img_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_img"], "test",
#                                                                VOCAB_INGR)
#                 c2b2c_txt_txt_recipe = get_recipe_info_from_ID(b2c_test_set[txt_id]["test"]["matched_txt"], "test",
#                                                                VOCAB_INGR)

#                 new_save_dir = save_dir + "/" + id_key
#                 make_dir(new_save_dir)
#                 original_recipe.save_recipe(new_save_dir)

#                 new_save_img_dir = new_save_dir + "/img_" + img_id
#                 new_save_txt_dir = new_save_dir + "/txt_" + txt_id
#                 make_dir(new_save_img_dir)
#                 make_dir(new_save_txt_dir)
#                 img_recipe.save_recipe(new_save_img_dir)
#                 txt_recipe.save_recipe(new_save_txt_dir)

#                 new_save_img_img_dir = new_save_img_dir + "/img_" + b2c_test_set[img_id]["test"]["matched_img"]
#                 new_save_img_txt_dir = new_save_img_dir + "/txt_" + b2c_test_set[img_id]["test"]["matched_txt"]
#                 new_save_txt_img_dir = new_save_txt_dir + "/img_" + b2c_test_set[txt_id]["test"]["matched_img"]
#                 new_save_txt_txt_dir = new_save_txt_dir + "/txt_" + b2c_test_set[txt_id]["test"]["matched_txt"]
#                 make_dir(new_save_img_img_dir)
#                 make_dir(new_save_img_txt_dir)
#                 make_dir(new_save_txt_img_dir)
#                 make_dir(new_save_txt_txt_dir)

#                 c2b2c_img_img_recipe.save_recipe(new_save_img_img_dir)
#                 c2b2c_img_txt_recipe.save_recipe(new_save_img_txt_dir)
#                 c2b2c_txt_img_recipe.save_recipe(new_save_txt_img_dir)
#                 c2b2c_txt_txt_recipe.save_recipe(new_save_txt_txt_dir)


#     print("Done!")



# if __name__ == "__main__":
#     main()
#     #generate_recipes_from_match_ID()
#     #generate_recipes_from_match_ID_c2b2c()
#     #generate_recipes_from_match_ID_c2b_only_one_match()
#     #generate_recipes_from_match_ID_c2b_no_match()
#     #generate_recipes_from_match_ID_c2a_only_one_match()
#     #generate_recipes_from_match_ID_c2a_no_match()
#     #pass
