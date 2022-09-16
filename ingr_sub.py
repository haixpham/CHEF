import pickle
import torch
import torchvision.transforms as transforms
from args import get_parser
from tqdm import tqdm

from data_loader import foodSpaceLoader, error_catching_loader
from utils import PadToSquareResize
from one_recipe import OneRecipe, save_recipe_list

parser = get_parser()
opts = parser.parse_args()

partition = opts.test_split.lower()
if partition not in ["test", "val"]:
    raise ValueError("Test split not specified")

print(" *** Find recipe on {} split *** ".format(partition))


def get_recipe_with_ingredient(data_loader, ingr, VOCAB_INGR, max_count=-1):
    ret = []
    
    for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
        recipe = OneRecipe(input, VOCAB_INGR, id)
        if recipe.is_containing(ingr):
            ret.append(recipe)
        cur_len = len(ret)
        if max_count > 0 and cur_len == max_count:
            break
    return ret


def get_recipe_with_ingredients(data_loader, ingrs, VOCAB_INGR, max_count=-1):
    ret = {}
    for ingr in ingrs:
        ret[ingr] = []
    
    for i, (input, id) in tqdm(enumerate(data_loader), total=len(data_loader)):
        recipe = OneRecipe(input, VOCAB_INGR, id)
        for ingr in ingrs:
            if recipe.is_containing(ingr):
                ret[ingr].append(recipe)
        cur_len = sum([len(ret[key]) for key in ret])
        if max_count > 0 and cur_len == max_count:
            break
    return ret


def prepare_recipes_by_ingredient(data_loader, ingr, VOCAB_INGR):
    recipes = get_recipe_with_ingredient(data_loader, ingr, VOCAB_INGR, max_count=-1)
    save_file = "data/recipes_of_{:s}_with_{:s}.json".format(partition, ingr)
    save_recipe_list(recipes, save_file)
    return recipes


def find_recipes_for_substitution(data_loader, VOCAB_INGR, INGRS = ["apple", "chicken", "beef", "fish", "pork", "spaghetti", "shrimp"]):
    for ingr in INGRS:
        prepare_recipes_by_ingredient(data_loader, ingr, VOCAB_INGR)


if __name__ == "__main__":
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

    find_recipes_for_substitution(data_loader, VOCAB_INGR)