import json
import pickle
import json
from tqdm import tqdm
import os
from utils import tok
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import timeit


LAYER1_FILE = "data/layer1.json"
LAYER2_FILE = "data/layer2.json"
ORG_REPLACEMENT_FILE = "data/det_ingrs.json"
CAN_REPLACEMENT_FILE = "data/replacement_dict.pkl"

OUTPUT_TRAIN = "data/train_samples.pkl" 
OUTPUT_VAL = "data/val_samples.pkl"
OUTPUT_TEST = "data/test_samples.pkl"

TRAIN_VOCAB_FILE = "data/w2v_tokenized_text.txt"
WORD2VEC_FILE = "data/w2v.bin"
WORD2VEC_VECTOR_FILE = "data/vocab.bin"

INGR_VOCAB_FILE = "data/ingr_vocab.pkl"
VOCAB_INGR_FILE = "data/vocab_ingr.pkl"

def read_R1M_conversion():
    with open(ORG_REPLACEMENT_FILE, "r") as fp:
        data = json.load(fp)
    ret = {}
    for x in data:
        id = x["id"]
        if id in ret:
            raise ValueError("duplicated ID")
        valid = x["valid"]
        ingrs = [t["text"] for t in x["ingredients"]]
        ret[id] = {"valid": valid, "ingredients": ingrs}
    return ret

def read_canonical_conversion():
    with open(CAN_REPLACEMENT_FILE, "rb") as fp:
        data = pickle.load(fp)
    ret = {}
    for key in data:
        new_key = key.replace("_", " ")
        ret[new_key] = data[key].replace("_", " ")
    return ret

# def convert_ingredient(ingr_to_convert, raw_ingrs, org_conversion, canonical_conversion):
#     valid = org_conversion["valid"]
#     converted_ingrs = org_conversion["ingredients"]
#     for idx, x in enumerate(raw_ingrs):
#         if x == ingr_to_convert:
#             if valid[idx]:
#                 new_ingr = converted_ingrs[idx]
#                 if new_ingr in canonical_conversion:
#                     new_ingr = canonical_conversion[new_ingr]
#                 return new_ingr
#     return ingr_to_convert


def convert_one_sample(recipe, imgs, org_conversion, can_conversion):
    id = recipe["id"]
    ingrs = [item["text"] for item in recipe["ingredients"]]
    insts = [item["text"] for item in recipe["instructions"]]
    title = recipe["title"]
    partition = recipe["partition"]
    valid = org_conversion["valid"]
    converted_ingrs = org_conversion["ingredients"]
    
    if len(imgs) == 0:
        return None
    
    replace_ingrs = []
    new_ingrs = []
    for i, ingr in enumerate(ingrs):
        if valid[i]:
            new_ingr = converted_ingrs[i]
            if new_ingr in can_conversion:
                can_ingr = can_conversion[new_ingr]
                replace_ingrs.append((ingr, new_ingr, can_ingr, can_ingr.replace(" ", "_")))
                new_ingrs.append(can_ingr.replace(" ", "_"))
    if len(replace_ingrs) == 0:
        return None
    
    new_sample = {"id": id, "title": title, "partition": partition, "ingredients": new_ingrs}
    new_insts = []
    for inst in insts:
        new_inst = inst
        for item in replace_ingrs:
            if new_inst.find(item[0]) >= 0:
                new_inst = new_inst.replace(item[0], item[3])
            if new_inst.find(item[1]) >= 0:
                new_inst = new_inst.replace(item[1], item[3])
            if new_inst.find(item[2]) >= 0:
                new_inst = new_inst.replace(item[2], item[3])
        new_insts.append(new_inst)
    new_sample["instructions"] = new_insts
    new_imgs = []
    for img in imgs:
        first_4_chars = [x for x in img[:4]]
        path = [partition] + first_4_chars + [img]
        path = "/".join(path)
        new_imgs.append(path)
    new_sample["imgs"] = new_imgs
    return new_sample
    
    
    
# START processing

if os.path.exists(OUTPUT_TRAIN) and os.path.exists(OUTPUT_TEST) and os.path.exists(OUTPUT_VAL):
    train_data = pickle.load(open(OUTPUT_TRAIN, "rb"))
    test_data = pickle.load(open(OUTPUT_TEST, "rb"))
    val_data = pickle.load(open(OUTPUT_VAL, "rb"))
else:
    print("Preparing data ...")
    train_data = {}
    val_data = {}
    test_data = {}

    all_recipes = json.load(open(LAYER1_FILE, "r"))
    print("Recipes loaded!")
    all_images = json.load(open(LAYER2_FILE, "r"))
    print("Images metadata loaded!")

    images = {}
    for item in all_images:
        id = item["id"]
        imgs = [x["id"] for x in item["images"]]
        images[id] = imgs
    all_images = images

    org_conversions = read_R1M_conversion()
    can_conversions = read_canonical_conversion()

    print(len(all_recipes))
    print(len(all_images))
    print(len(org_conversions))

    not_used = 0

    for recipe in tqdm(all_recipes, total=len(all_recipes)):
        id = recipe["id"]
        if id not in all_images or len(all_images[id]) == 0:
            not_used += 1
            continue
        sample = convert_one_sample(recipe, all_images[id], org_conversions[id], can_conversions)
        if not sample:
            not_used += 1
            continue
        #id = sample["id"]
        partition = sample["partition"]
        if partition == "train":
            train_data[id] = sample
        elif partition == "test":
            test_data[id] = sample
        elif partition == "val":
            val_data[id] = sample

    with open(OUTPUT_TRAIN, "wb") as fp:
        pickle.dump(train_data, fp)

    with open(OUTPUT_TEST, "wb") as fp:
        pickle.dump(test_data, fp)

    with open(OUTPUT_VAL, "wb") as fp:
        pickle.dump(val_data, fp)

    print("Total number of recipes: ", len(all_recipes))
    print("Total number of recipes with images: ", len(all_images))
    print("Total number of train part: ", len(train_data))
    print("Total number of test part: ", len(test_data))
    print("Total number of val part: ", len(val_data))
    print("Total number of unused recipes: ", not_used)

# prepare data to train word2vec
if not os.path.exists(TRAIN_VOCAB_FILE):
    print("Prepare data to train word2vec ...")
    with open(TRAIN_VOCAB_FILE, "w") as fp:
        for id in tqdm(train_data, total=len(train_data)):
            item = train_data[id]
            ingredients = item["ingredients"]
            instructions = [tok(x) for x in item["instructions"]]
            
            txt = 'Title :\t ' + tok(item["title"]) + ' \t Ingredients : \t ' + ' '.join(ingredients) + ' \t Instructions :\t ' + ' \t '.join(instructions) + "\n"
            fp.write(txt)

# train word2vec
if not os.path.exists(WORD2VEC_FILE):
    print("Train Word2Vec ...")

    class MonitorCallback(CallbackAny2Vec):
        def __init__(self):
            self._iter = 0
            self._current_time = timeit.default_timer()

        def on_epoch_begin(self, model):
            print("Epoch: ", self._iter)
            self._iter += 1

        def on_epoch_end(self, model):
            current_time = timeit.default_timer()
            print("Model loss:", model.get_latest_training_loss())  # print loss
            elapse = current_time - self._current_time
            print("Elapsed time: {:.4f}".format(elapse))
            self._current_time = current_time

    w2v_monitor = MonitorCallback()
    model = Word2Vec(corpus_file=TRAIN_VOCAB_FILE, vector_size=300, window=10, min_count=10, workers=6, sg=1, epochs=10, hs=1, callbacks=[w2v_monitor])
    model.save(WORD2VEC_FILE)
    print("Training finished!")
else:
    model = Word2Vec.load(WORD2VEC_FILE)

# prepare Vocab
wv = model.wv

if not os.path.exists(WORD2VEC_VECTOR_FILE):
    wv.save_word2vec_format(WORD2VEC_VECTOR_FILE, binary=True)

ingr_vocab = {wv.index_to_key[i].rstrip(): i + 4 for i in range(len(wv))}
ingr_vocab['<PAD>'] = 0
ingr_vocab['<UNK>'] = 1
ingr_vocab['<BOS>'] = 2
ingr_vocab['<EOS>'] = 3
vocab_ingr = {i + 4: wv.index_to_key[i].rstrip() for i in range(len(wv))}
vocab_ingr[0] = '<PAD>'
vocab_ingr[1] = '<UNK>'
vocab_ingr[2] = '<BOS>'
vocab_ingr[3] = '<EOS>'

with open(INGR_VOCAB_FILE, 'wb') as f:
    pickle.dump(ingr_vocab, f)
with open(VOCAB_INGR_FILE, 'wb') as f:
    pickle.dump(vocab_ingr, f)
print("Vocab size: ", len(ingr_vocab))
