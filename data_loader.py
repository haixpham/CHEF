from __future__ import print_function
import os
import pickle
import numpy as np
import lmdb
import torch
from PIL import Image, ExifTags
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torch.utils.data as data

from utils import PadToSquareResize

def get_default_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([PadToSquareResize(resize=256, padding_mode="reflect"),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])


def default_loader(path):
    im = Image.open(path).convert('RGB')
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(im._getexif().items())

        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)
    except:
        pass
    return im


def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)


def collate_fn(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)


def error_catching_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(im._getexif().items())

            if exif[orientation] == 3:
                im = im.rotate(180, expand=True)
            elif exif[orientation] == 6:
                im = im.rotate(270, expand=True)
            elif exif[orientation] == 8:
                im = im.rotate(90, expand=True)
        except:
            pass

        return im
    except:
        # print('bad image: '+path, end =" ")#print(file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class foodSpaceLoader(data.Dataset):
    def __init__(self, img_path, transform=None, loader=error_catching_loader, data_path=None, partition=None, maxInsts=20,
                 maxImgs=5, loadImage=True):
        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        try:
            with open(os.path.join(data_path, partition + '_lmdb/keys.pkl'), 'rb') as fp:
                self.ids = pickle.load(fp)
        except:
            # backward compatible
            with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as fp:
                self.ids = pickle.load(fp)


        self.imgPath = img_path
        self.maxInst = maxInsts
        if transform is None:
            self.transform = get_default_transforms()
        else:
            self.transform = transform
        self.loader = loader
        self.maxImgs = maxImgs
        self.loadImage = loadImage
        
    def __getitem__(self, index):
        # recipe id
        rec_id = self.ids[index]
        # read lmdb
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(rec_id.encode())
        # decode sample
        try:
            sample = pickle.loads(serialized_sample)
        except:
            # backward compatible
            sample = pickle.loads(serialized_sample, encoding='latin1')

        # image
        if self.loadImage:
            imgs = sample['imgs']
            if self.partition == 'train':
                imgIdx = np.random.choice(range(min(self.maxImgs, len(imgs))))
            else:
                imgIdx = 0
            img_name = imgs[imgIdx]
            if isinstance(img_name, dict):
                # backward compatible
                img_name = img_name["id"]
                first_4_chars = [x for x in img_name[:4]]
                img_path = [self.imgPath, self.partition] + first_4_chars + [img_name]
                img_path = os.path.join(*img_path)
            else:
                img_path = os.path.join(self.imgPath, img_name)
            img = self.loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
        
        # ingredients
        if sample["ingrs"][0] == 0:
            sample["ingrs"][0] = 1
        ingrs = torch.LongTensor(sample['ingrs'].astype('int'))
        igr_ln = (ingrs > 0).sum()

        # instructions
        intrs_novec = torch.LongTensor(sample['intrs_novec'].astype('int'))
        intrs_novec_ln = (intrs_novec > 0).sum(1)
        intrs_novec_num = (intrs_novec_ln > 0).sum()

        # title
        title_word_inds = torch.LongTensor(sample['title_word_inds'].astype('int'))
        title_word_inds_ln = (title_word_inds > 0).sum()

        if self.loadImage:
            return [img, ingrs, igr_ln, title_word_inds, title_word_inds_ln, intrs_novec, intrs_novec_ln, intrs_novec_num], rec_id
        else:
            return [ingrs, igr_ln, title_word_inds, title_word_inds_ln, intrs_novec, intrs_novec_ln, intrs_novec_num], rec_id

    def __len__(self):
        return len(self.ids)

