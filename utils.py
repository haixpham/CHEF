import copy
import os
import random
import numbers
import time
import torch
import torchvision.transforms.functional
import numpy as np
import simplejson as json
from html.parser import HTMLParser
from PIL import Image
import pathlib as plb
import shutil
from datetime import datetime


class PadToSquareResize(object):
    def __init__(self, resize=None, fill=0, padding_mode='constant', interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'random']

        self.fill = fill
        self.padding_mode = padding_mode
        self.resize = resize
        self.interpolation = interpolation

    def __call__(self, img):
        if img.size[0] < img.size[1]:
            pad = img.size[1] - img.size[0]
            self.padding = (int(pad/2), 0, int(pad/2 + pad%2), 0)
        elif img.size[0] > img.size[1]:
            pad = img.size[0] - img.size[1]
            self.padding = (0, int(pad/2), 0, int(pad/2 + pad%2))
        else:
            self.padding = (0, 0, 0, 0)

        if self.padding_mode == 'random':
            pad_mode = random.choice(['constant', 'edge', 'reflect', 'symmetric'])
        else:
            pad_mode = self.padding_mode

        if self.resize is None:
            return torchvision.transforms.functional.pad(img, self.padding, self.fill, self.padding_mode)
        else:
            return torchvision.transforms.functional.resize(torchvision.transforms.functional.pad(img, self.padding, self.fill, pad_mode), self.resize, self.interpolation)


    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cosine_similarity(x1, x2=None):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    out = torch.mm(x1, x2.t()) / (w1 * w2.t())
    return out


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def euclidean_distance(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  # import pdb; pdb.set_trace()
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def worker_init_fn(worker_id):
    seed = worker_id
    np.random.seed(seed)


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference


TicToc = TicTocGenerator()


def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)


class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base


REPLACEMENTS = {
    u'\x91':"'", u'\x92':"'", u'\x93':'"', u'\x94':'"', u'\xa9':'',
    u'\xba': ' degrees ', u'\xbc':' 1/4', u'\xbd':' 1/2', u'\xbe':' 3/4',
    u'\xd7':'x', u'\xae': '',
    '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    u'\\u2153':' 1/3', '\\u00bd':' 1/2', '\\u00bc':' 1/4', '\\u00be':' 3/4',
    '\\u2154':' 2/3', '\\u215b':' 1/8', '\\u215c':' 3/8', '\\u215d':' 5/8',
    '\\u215e':' 7/8', '\\u2155':' 1/5', '\\u2156':' 2/5', '\\u2157':' 3/5',
    '\\u2158':' 4/5', '\\u2159':' 1/6', '\\u215a':' 5/6', '\\u2014':'-',
    '\\u0131':'1', '\\u2122':'', '\\u2019':"'", '\\u2013':'-', '\\u2044':'/',
    '\\u201c':'\\"', '\\u2018':"'", '\\u201d':'\\"', '\\u2033': '\\"',
    '\\u2026': '...', '\\u2022': '', '\\u2028': ' ', '\\u02da': ' degrees ',
    '\\uf04a': '', u'\xb0': ' degrees ', '\\u0301': '', '\\u2070': ' degrees ',
    '\\u0302': '', '\\uf0b0': ''
}

parser = HTMLParser()


def tok(text, ts=None):
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']
    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text


def untok(text, ts=False):
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']
    for t in ts:
        text = text.replace(' ' + t + ' ', t)
    return text


def set_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id >= 0 else ""

def get_list_of_files(dir_path):
    root_path = plb.Path(dir_path)
    ret = []
    for filepath in root_path.iterdir():
        if filepath.is_file():
            ret.append(filepath.name)
    return ret

def make_dir(dirname):
    plb.Path(dirname).mkdir(parents=True, exist_ok=True)

def is_dir(filename):
    return plb.Path(filename).is_dir()

def is_file(filename):
    return plb.Path(filename).is_file()

def get_name(filename):
    return plb.Path(filename).stem

def get_extension(filename):
    if not is_file(filename):
        return ""
    else:
        return plb.Path(filename).suffix

def get_parent_path(filename):
    return str(plb.Path(filename).parent)

def exists(filename):
    return plb.Path(filename).exists()


def remove(path):
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def get_HOME_DIR():
    return str(plb.Path().home())


def get_current_time_string():
    return datetime.now().strftime("%Y%m%d-%H%M%S")