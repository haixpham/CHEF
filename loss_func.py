import torch
from torch.autograd import Variable
import numpy as np
from utils import cosine_similarity, cosine_distance, euclidean_distance

def calculate_loss(output, opts):
    if opts.ohem:
        if opts.intraClassLoss:
            label = list(range(0, output[0].shape[0]))
            label.extend(label)
            label = np.array(label)
            label = torch.tensor(label).cuda().long()

            feat = torch.cat((output[0], output[1]))
            if opts.lossDistance == 'cosine':
                dist_mat = cosine_distance(feat, feat)
            elif opts.lossDistance == 'euclidean':
                dist_mat = euclidean_distance(feat, feat)
            else:
                Exception('Undefined distance!!!')
            N = dist_mat.size(0)

            # shape [N, N]
            is_pos = label.expand(N, N).eq(label.expand(N, N).t())
            is_neg = label.expand(N, N).ne(label.expand(N, N).t())
        else:
            label = list(range(0, output[0].shape[0]))
            # label.extend(label)
            label = np.array(label)
            label = torch.tensor(label).cuda().long()

            # feat = torch.cat((output[0], output[1]))
            if opts.lossDistance == 'cosine':
                dist_mat = cosine_distance(output[0], output[1])
            elif opts.lossDistance == 'euclidean':
                dist_mat = euclidean_distance(output[0], output[1])
            else:
                Exception('Undefined distance!!!')
            N = dist_mat.size(0)

            # shape [N, N]
            is_pos = label.expand(N, N).eq(label.expand(N, N).t())
            is_neg = label.expand(N, N).ne(label.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive) both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative) both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1)).cuda()
        cos_lossA = torch.nn.MarginRankingLoss(margin=opts.alpha)(dist_an, dist_ap, y)
        cos_loss = cos_lossA
    else:
        if opts.intraClassLoss:
            label = list(range(0, output[0].shape[0]))
            label.extend(label)
            label = np.array(label)
            label = torch.tensor(label).cuda().long()
            feat = torch.cat((output[0], output[1]))
            outA = cosine_similarity(feat, feat)
            N = outA.size(0)
            # shape [N, N]
            is_pos = label.expand(N, N).eq(label.expand(N, N).t())
            is_neg = label.expand(N, N).ne(label.expand(N, N).t())
        else:
            outA = cosine_similarity(output[0], output[1])
            v = torch.ones([output[0].shape[0]]).cuda()  # .cpu()
            a = -torch.ones([output[0].shape[0], output[0].shape[0]]).cuda()  # .cpu()
            mask = torch.diag(torch.ones_like(v))
            target = (mask * torch.diag(v) + (1. - mask) * a)
            is_pos = target == 1
            is_neg = target == -1
            normalizing_factor = (is_pos == 1).sum().float()
            normalizing_factor = 1 if normalizing_factor == 0 else normalizing_factor
            N = normalizing_factor


        cos_lossA = outA.clone()
        cos_lossA[is_pos == 1] = (1 - cos_lossA[is_pos == 1])
        cos_lossA[is_neg == 1] = torch.nn.functional.relu_(cos_lossA[is_neg == 1] - opts.alpha)
        cos_loss = cos_lossA[is_pos | is_neg].sum() / N

    return cos_loss