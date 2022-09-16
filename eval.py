import torch
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import random
import os

from utils import AverageMeter, exists, make_dir, PadToSquareResize
from torch_utils import load_torch_model
from test_utils import get_model_signature
import models
from loss_func import calculate_loss
from data_loader import foodSpaceLoader, error_catching_loader


def load_model(model_path, opts):
    if not os.path.exists(model_path):
        return None

    if not opts.no_cuda:
        opts.gpu = list(map(int, opts.gpu.split(',')))
        print('Using GPU(s): ' + ','.join([str(x) for x in opts.gpu]))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in opts.gpu])
    
    model = models.FoodSpaceNet(opts)

    if not load_torch_model(model, model_path):
        return None

    if not opts.no_cuda:
        model.cuda()
        if len(opts.gpu) > 1:
            model = torch.nn.DataParallel(model)
    else:
        model.cpu()

    return model


def run_model(data_loader, model, opts):
    losses = AverageMeter()
    model.eval()
    for i, (input, rec_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            output = model(input, opts)

            if i == 0:
                data0 = output[0].detach().cpu().numpy()
                data1 = output[1].detach().cpu().numpy()
                data2 = rec_ids
            else:
                data0 = np.concatenate((data0, output[0].detach().cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, output[1].detach().cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, rec_ids), axis=0)

            loss = calculate_loss(output, opts)

            losses.update(loss.item(), input[0].size(0))

    return data2, data0, data1, losses


def validate(val_loader, model, opts):
    rec_ids, im_embs, re_embs, losses = run_model(val_loader, model, opts)

    medR, recall, meanR, meanDCG = rank(opts, im_embs, re_embs, rec_ids)

    print('\t* Val medR {medR:.4f}\tRecall {recall}\tVal meanR {meanR:.4f}\tVal meanDCG {meanDCG:.4f}'.format(medR=medR, recall=recall, meanR=meanR, meanDCG=meanDCG))

    return medR, recall, meanR, meanDCG


def rank(opts, img_embeds, rec_embeds, names):
    st = random.getstate()
    random.seed(opts.seed)
    idxs = np.argsort(names)
    names = names[idxs]
    if opts.test_K < 0:
        opts.test_K = 1000
    if opts.test_K == 0:
        opts.test_N_folds = 1
    if opts.test_N_folds < 0:
        opts.test_N_folds = 10
    if opts.test_K > 0:
        idxs = range(opts.test_K)
    else:
        idxs = range(len(names))

    all_rank = []
    glob_rank = []
    dcg = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(opts.test_N_folds):
        if opts.test_K == 0:
            ids = range(len(names))
            img_sub = img_embeds
            rec_sub = rec_embeds
        else:
            ids = random.sample(range(0,len(names)), opts.test_K)
            img_sub = img_embeds[ids,:]
            rec_sub = rec_embeds[ids,:]

        if opts.embtype == 'image':
            sims = np.dot(img_sub,rec_sub.T)# im2recipe
        else:
            sims = np.dot(rec_sub,img_sub.T)# recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}
        for ii in idxs:
            # sort indices in descending order
            sorting = np.argsort(sims[ii,:])[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1] += 1
            if (pos+1) <= 5:
                recall[5] += 1
            if (pos+1) <= 10:
                recall[10] += 1

            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i] = recall[i]/opts.test_K

        med = np.median(med_rank)
        all_rank.append(np.mean(med_rank))
        dcg.append(np.array([1/np.log2(r+1) for r in med_rank]).mean())
        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    random.setstate(st)

    return np.average(glob_rank), glob_recall, np.mean(all_rank), np.mean(dcg)


def extract_partition_embeddings(model, opts, partition, batch_size):
    model_sig = get_model_signature(opts)
    filename = "data/embed_cache/" + model_sig + "_" + partition + ".npz" 
    if exists(filename):
        emb_data = np.load(filename)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        test_data = foodSpaceLoader(opts.img_path,
                                transforms.Compose([
                                    PadToSquareResize(resize=256, padding_mode='reflect'),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize]),
                                data_path=opts.data_path,
                                partition=partition,
                                loader=error_catching_loader)
        test_data_loader = torch.utils.data.DataLoader(test_data,
                                                        batch_size=batch_size, shuffle=False,
                                                        num_workers=opts.workers, pin_memory=True)
        rec_ids, img_embs, rec_embs, _ = run_model(test_data_loader, model, opts)
        emb_data = {}
        emb_data["img_embeds"] = img_embs
        emb_data["rec_embeds"] = rec_embs
        emb_data["rec_ids"] = rec_ids
        # save data
        make_dir("data/embed_cache")
        np.savez(filename, img_embeds=img_embs, rec_embeds=rec_embs, rec_ids=rec_ids)
    return emb_data