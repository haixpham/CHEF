import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import gensim
import torch.nn.functional as F
import os
import numpy as np
import math
import pickle
import basic


# attention layer used in instruction encoding
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, with_attention):
        super(AttentionLayer, self).__init__()
        self.with_attention = with_attention

        self.u = torch.nn.Parameter(torch.Tensor(input_dim))  # u = [2*hid_dim]
        torch.nn.init.normal_(self.u, mean=0, std=0.01)
        self.u.requires_grad = True

        self.fc = nn.Linear(input_dim, input_dim)
        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, x):
        # x = [BS, max_len, 2*hid_dim]
        # a trick used to find the mask for the softmax
        mask = (x != 0)
        mask = mask[:, :, 0]
        h = torch.tanh(self.fc(x))  # h = [BS, max_len, 2*hid_dim]
        if self.with_attention == 1:  # softmax
            scores = h @ self.u  # scores = [BS, max_len], unnormalized importance
            masked_scores = scores.masked_fill(~mask, -1e32)
            alpha = F.softmax(masked_scores, dim=1)  # alpha = [BS, max_len], normalized importance
        elif self.with_attention == 2:  # Transformer
            scores = h @ self.u / math.sqrt(h.shape[-1])  # scores = [BS, max_len], unnormalized importance
            masked_scores = scores.masked_fill((1 - mask).byte(), -1e32)
            alpha = F.softmax(masked_scores, dim=1)  # alpha = [BS, max_len], normalized importance

        alpha = alpha.unsqueeze(-1)  # alpha = [BS, max_len, 1]
        out = x * alpha  # out = [BS, max_len, 2*hid_dim]
        out = out.sum(dim=1)  # out = [BS, 2*hid_dim]
        return out, alpha.squeeze(-1)


# encode individual instructions
class SentenceEncoder(nn.Module):
    def __init__(self, opts, embs, emb_dim, hid_dim):
        super(SentenceEncoder, self).__init__()
        self.with_attention = opts.instAtt
        self.embed_layer = embs
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        if self.with_attention:
            self.atten_layer = AttentionLayer(2 * hid_dim, self.with_attention)

    def forward(self, sent_list, lengths):
        x = self.embed_layer(sent_list)
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)

        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            y, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=20)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous()
            feat, alpha = self.atten_layer(output)
            return feat, alpha
        else:
            _, (h, _) = self.rnn(packed_seq)
            h = h.transpose(0, 1)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous()
            feat = output.view(output.size(0), output.size(1) * output.size(2))
            return feat


# encode sequence of instructions' vectors
class DocEncoder(nn.Module):
    def __init__(self, opts, sent_encoder, hid_dim):
        super(DocEncoder, self).__init__()
        self.sent_encoder = sent_encoder
        self.with_attention = opts.instAtt
        self.rnn = nn.LSTM(
            input_size=2 * hid_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)
        if self.with_attention:
            self.atten_layer_sent = AttentionLayer(2 * hid_dim, self.with_attention)

    def forward(self, doc_list, n_insts, n_words_each_inst, return_all=False):
        embs = []
        attentions_words_each_inst = []
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i]
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero().shape[0]]
            if self.with_attention:
                emb_doc, alpha = self.sent_encoder(doc[:ln], sent_lns)
                attentions_words_each_inst.append(alpha)
            else:
                emb_doc = self.sent_encoder(doc[:ln], sent_lns)
            embs.append(emb_doc)

        embs = sorted(embs, key=lambda x: -x.shape[0])
        packed_seq = torch.nn.utils.rnn.pack_sequence(embs)
        _, sorted_idx = n_insts.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)

        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            y, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=20)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous()
            out, attentions_each_inst = self.atten_layer_sent(output)
            feat = out
            if return_all:
                # return attention weights
                return feat, attentions_each_inst, attentions_words_each_inst
            else:
                return feat
        else:
            _, (h, _) = self.rnn(packed_seq)
            h = h.transpose(0, 1)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous()
            feat = output.view(output.size(0), output.size(1) * output.size(2))
            return feat

    def forward_with_masks(self, doc_list, n_insts, n_words_each_inst, sent_masks):
        embs = []
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i]
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero().shape[0]]
            emb_doc = self.sent_encoder.forward_with_masks(doc[:ln], sent_lns, sent_masks[i if i < len(n_insts) else 0])
            embs.append(emb_doc)
        embs = sorted(embs, key=lambda x: -x.shape[0])
        packed_seq = torch.nn.utils.rnn.pack_sequence(embs)
        _, sorted_idx = n_insts.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        _, (h, _) = self.rnn(packed_seq)
        h = h.transpose(0, 1)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(h)
        output = h.gather(0, unsorted_idx).contiguous()
        feat = output.view(output.size(0), output.size(1) * output.size(2))
        return feat



def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


# Word embedding
class wordEmbs(nn.Module):
    def __init__(self, opts):
        super(wordEmbs, self).__init__()
        if opts.w2vInit:
            model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(opts.data_path, 'vocab.bin'), binary=True)
            #model = gensim.models.Word2Vec.load(os.path.join(opts.data_path, 'vocab.bin')).wv
            vec = torch.FloatTensor(model.vectors)
            np.random.seed(0)
            unknown = np.random.uniform(low=-0.25, high=0.25, size=[4,opts.w2vDim]) # initialize with random values the vectors of <PAD>, <UNK>, <BOS> and <EOS>
            vec = torch.cat([torch.tensor(unknown).float(), vec])
            self.embs = nn.Embedding.from_pretrained(vec)
        else:
            with open(os.path.join(opts.data_path, 'ingr_vocab.pkl'), 'rb') as f:
                self.ingr_vocab = pickle.load(f)
            vec = torch.tensor(np.random.uniform(low=-0.25, high=0.25, size=[len(self.ingr_vocab), opts.w2vDim])).float()
            self.embs = nn.Embedding.from_pretrained(vec)

    def forward(self, x):
        return self.embs(x)


# Attention layer
class attLayer(nn.Module):
    def __init__(self, attSize):
        super(attLayer, self).__init__()
        self.context_vector_size = [attSize, 1]
        self.w_proj = nn.Linear(in_features=attSize, out_features=attSize)
        torch.nn.init.normal_(self.w_proj.weight, mean=0, std=0.01)

        self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size))
        torch.nn.init.normal_(self.w_context_vector, mean=0, std=0.01)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Hw = torch.tanh(self.w_proj(x))
        w_score = self.softmax(Hw.matmul(self.w_context_vector) / np.power(Hw.size(2), 0.5))
        # w_score = Hw.matmul(self.w_context_vector)
        # w_score = w_score/w_score.sum(1)[:,None]

        x = x.mul(w_score)
        x = torch.sum(x, dim=1)

        return x, w_score


# Attention layer
class visualAttLayer(nn.Module):
    def __init__(self, inSize, attSize=1024):
        super(visualAttLayer, self).__init__()
        self.context_vector_size = [attSize, 1]
        self.w_proj = nn.Linear(in_features=inSize, out_features=attSize)
        torch.nn.init.normal_(self.w_proj.weight, mean=0, std=0.01)

        self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size))
        torch.nn.init.normal_(self.w_context_vector, mean=0, std=0.01)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_size = x.size
        x = x.view(x_size(0), x_size(1), -1)
        x = x.permute(0, 2, 1)  # transpose_(1,2)
        Hw = torch.tanh(self.w_proj(x))
        w_score = self.softmax(Hw.matmul(self.w_context_vector) / np.power(Hw.size(2), 0.5))

        x = x.mul(w_score* (x_size(2)*x_size(3)))
        # x = torch.sum(x, dim=1)
        x = x.permute(0, 2, 1)  # transpose_(1,2)
        x = x.view(x_size(0), x_size(1), x_size(2), x_size(3))

        return x, w_score


# Title embedding
class titleNet(nn.Module):
    def __init__(self, embs, opts):
        super(titleNet, self).__init__()
        self.embs = embs
        self.title_GRU = nn.GRU(input_size=opts.w2vDim,
                               hidden_size=opts.rnnDim,
                               bidirectional=True,
                               batch_first=True)
        # self.sent_encoder = SentenceEncoder(embs, opts, emb_dim=opts.w2vDim, hid_dim=opts.rnnDim)

    def forward(self, x, sq_lengths, opts):
        x_emb = self.embs(x.long())
        lengths, perm_index = sq_lengths.sort(0, descending=True)
        input = x_emb[perm_index]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, list(lengths.data), batch_first=True)
        output, hidden = self.title_GRU(packed_input)
        _, original_idx = perm_index.sort(0, descending=False)
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden)
        output = hidden.gather(1,unsorted_idx).transpose(0,1).contiguous()
        x_emb = output.view(output.size(0),output.size(1)*output.size(2))
        w_score = torch.ones((x_emb.shape[0], sq_lengths.max().item(), 1))
        # x_emb = self.sent_encoder(x, sq_lengths)

        return x_emb

# Ingredients embedding
# class ingrNet(nn.Module):
#     def __init__(self, embs,opts):
#         super(ingrNet, self).__init__()
#         self.embs = embs
#         self.ingr_GRU = nn.GRU(input_size=opts.w2vDim,
#                                hidden_size=opts.rnnDim,
#                                bidirectional=True,
#                                batch_first=True)
#         # self.sent_encoder = SentenceEncoder(embs, opts, emb_dim=opts.w2vDim, hid_dim=opts.rnnDim)
#
#     def forward(self, x, sq_lengths, opts):
#         x_emb = self.embs(x.long())
#         lengths, perm_index = sq_lengths.sort(0, descending=True)
#         input = x_emb[perm_index]
#         packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, list(lengths.data), batch_first=True)
#         output, hidden = self.ingr_GRU(packed_input)
#         _, original_idx = perm_index.sort(0, descending=False)
#         unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden)
#         output = hidden.gather(1,unsorted_idx).transpose(0,1).contiguous()
#         x_emb = output.view(output.size(0),output.size(1)*output.size(2))
#         w_score = torch.ones((x_emb.shape[0], sq_lengths.max().item(), 1))
#         # x_emb = self.sent_encoder(x, sq_lengths)
#
#         return x_emb
# Ingredients embedding
class ingrNet(nn.Module):
    def __init__(self, embs, opts):
        super(ingrNet, self).__init__()
        self.embs = embs
        if opts.ingrInLayer == 'RNN':
            self.ingr_GRU = nn.GRU(input_size=opts.w2vDim,
                                   hidden_size=opts.rnnDim,
                                   bidirectional=True,
                                   batch_first=True)
        elif opts.ingrInLayer == 'dense':
            self.linear = nn.Linear(opts.w2vDim, 2 * opts.rnnDim)
            nn.init.normal_(self.linear.weight, mean=0, std=np.sqrt(2.0 / (2 * opts.w2vDim)))
        elif opts.ingrInLayer == 'tstsLSTM':
            self.ingr_LSTM = tstsModel(opts.w2vDim, embs, hidden_dim=opts.rnnDim*2, use_leaf_rnn=False, intra_attention=False,
                                    dropout_prob=0.2, bidirectional=False)
        else:
            raise Exception('Only \'RNN\' or \'dense\' are valid options for ingrInLayer')
        if opts.ingrInLayer == 'dense' or opts.ingrInLayer == 'none' or opts.ingrAtt:
            self.attention = attLayer(2 * opts.rnnDim)

    def forward(self, x, sq_lengths, opts, return_all=False):
        x_vec = self.embs(x.long())
        if opts.ingrInLayer == 'RNN':
            x_emb = x_vec[:]
        elif opts.ingrInLayer == 'dense':
            x_emb = self.linear(x_vec)
        elif opts.ingrInLayer == 'tstsLSTM':
            x_emb = self.ingr_LSTM(x, sq_lengths)
            return x_emb

        lengths, perm_index = sq_lengths.sort(0, descending=True)
        input = x_emb[perm_index]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, list(lengths.data), batch_first=True)
        if opts.ingrInLayer == 'RNN':
            output, hidden = self.ingr_GRU(packed_input)
            output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=opts.maxSeqlen)[0]
        else:
            output = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True, total_length=opts.maxSeqlen)[0]

        if opts.ingrAtt:
            _, odx = perm_index.sort(0)
            odx = odx.view(-1, 1).unsqueeze(1).expand(output.size(0), output.size(1), output.size(2))
            x_emb = output.gather(0, odx)
            x_emb, w_score = self.attention(x_emb)
        elif opts.ingrInLayer == "dense":
            x_emb = torch.sum(x_emb, dim=1)
        elif opts.ingrInLayer == 'RNN':
            _, original_idx = perm_index.sort(0, descending=False)
            unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden)
            output = hidden.gather(1,unsorted_idx).transpose(0,1).contiguous()
            x_emb = output.view(output.size(0),output.size(1)*output.size(2))
            w_score = torch.ones((x_emb.shape[0], sq_lengths.max().item(), 1))
        else:
            raise Exception('Option for ingrInLayer and ingrAtt are not compatible')
        if return_all and opts.ingrAtt:
            return x_emb, w_score
        else:
            return x_emb


# Instructions embedding
class instNet(nn.Module):
    def __init__(self, embs, opts):
        super(instNet, self).__init__()
        self.opts = opts
        if opts.instInLayer == 'LSTM':
            sen_encoder = SentenceEncoder(opts, embs, emb_dim=opts.w2vDim, hid_dim=opts.rnnDim)
        elif opts.instInLayer == 'tstsLSTM':
            sen_encoder = tstsModel(opts.w2vDim, embs, hidden_dim=opts.rnnDim*2, use_leaf_rnn=False, intra_attention=False,
                                    dropout_prob=0.2, bidirectional=False)
        else:
            raise Exception("only LSTM and tstsLSTM supported")

        if opts.docInLayer == 'LSTM':
            self.doc_encoder = DocEncoder(opts, sen_encoder, opts.rnnDim)
        elif opts.docInLayer == 'tstsLSTM':
            self.doc_encoder = tstsDocEncoder(opts.rnnDim*2, sen_encoder, bidirectional=False)
        else:
            raise Exception("only LSTM and tstsLSTM supported")

    def forward(self, x, sq_lengths, v2, opts):
        x_emb = self.doc_encoder(x, sq_lengths, v2)
        return x_emb


# Vision embedding, attenton on resnet
class visionMLP(nn.Module):
    def __init__(self, opts):
        super(visionMLP, self).__init__()
        self.opts = opts
        resnet = models.resnet50(pretrained=True)
        if opts.visualmodel == 'simple_14':
            # modules = list(resnet.children())[:-2]
            # modules.remove(modules[-1])
            # self.visionMLP = nn.Sequential(*modules)
            # self.attentionVisual = attLayer(opts.embDim)
            # self.avg = nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
            # self.visual_embedding = nn.Sequential(
            #     nn.BatchNorm1d(opts.embDim),
            #     nn.Linear(opts.embDim, opts.embDim),
            #     nn.Tanh())
            # self.align_img = nn.Sequential(
            #     nn.BatchNorm1d(opts.embDim),
            #     nn.Tanh())
            modulesA = list(resnet.children())[:7]
            attentionVisual14 = [visualAttLayer(inSize=1024)]
            self.sequential14 = nn.Sequential(*(modulesA + attentionVisual14))
            modulesT = list(resnet.children())[7:9]
            self.sequentialT = nn.Sequential(*modulesT)
        elif opts.visualmodel == 'simple_28':
            modules28 = list(resnet.children())[:6]
            attentionVisual28 = [visualAttLayer(inSize=512)]
            self.sequential28 = nn.Sequential(*(modules28 + attentionVisual28))
            modules14 = [list(resnet.children())[6]]
            self.sequential14 = nn.Sequential(*modules14)
            modulesT = list(resnet.children())[7:9]
            self.sequentialT = nn.Sequential(*modulesT)
        elif opts.visualmodel == 'simple_28-14':
            modules28 = list(resnet.children())[:6]
            attentionVisual28 = [visualAttLayer(inSize=512)]
            self.sequential28 = nn.Sequential(*(modules28 + attentionVisual28))
            modules14 = [list(resnet.children())[6]]
            attentionVisual14 = [visualAttLayer(inSize=1024)]
            self.sequential14 = nn.Sequential(*(modules14 + attentionVisual14))
            modulesT = list(resnet.children())[7:9]
            self.sequentialT = nn.Sequential(*modulesT)
        else:
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

    def forward(self, x, opts):
        w_score = []
        if opts.visualmodel == 'simple_14':
            x, w_score14 = self.sequential14(x)
            w_score.append(w_score14)
            x = self.sequentialT(x)
        elif opts.visualmodel == 'simple_28':
            x, w_score28 = self.sequential28(x)
            x = self.sequential14(x)
            w_score.append(w_score28)
            x = self.sequentialT(x)
        elif opts.visualmodel == 'simple_28-14':
            x, w_score28 = self.sequential28(x)
            x, w_score14 = self.sequential14(x)
            w_score.append(w_score14)
            w_score.append(w_score28)
            x = self.sequentialT(x)
        else:
            x = self.visionMLP(x)
        return x, w_score


# Main FoodSpaceNet
class FoodSpaceNet(nn.Module):
    def __init__(self, opts):
        super(FoodSpaceNet, self).__init__()
        self.opts = opts
        self.visionMLP = visionMLP(opts)
        self.visual_embedding = nn.Sequential(
            nn.BatchNorm1d(opts.imfeatDim),
            nn.Linear(opts.imfeatDim, opts.embDim),
            nn.Tanh())
        self.align_img = nn.Sequential(
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh())


        self.embs = wordEmbs(opts)
        self.titleNet_ = titleNet(self.embs,opts)
        self.ingrNet_ = ingrNet(self.embs,opts)
        self.instNet_ = instNet(self.embs,opts)
        self.recipe_embedding = nn.Sequential(
            nn.BatchNorm1d(opts.rnnDim * 6,),
            nn.Linear(opts.rnnDim * 6, opts.embDim, opts.embDim),
            nn.Tanh())
        self.align_rec = nn.Sequential(
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh())

        self.align = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
        )



    def forward(self, input, opts):  # we need to check how the input is going to be provided to the model
        if not opts.no_cuda:
            for i in range(len(input)):
                input[i] = input[i].cuda()
        x, y1, y2, z1, z2, v1, v2, v3 = input

        # visual embedding
        visual_emb, _ = self.visionMLP(x, opts)
        # if opts.visualmodel == 'simple_high1':
        #     visual_emb = visual_emb.view(visual_emb.size(0),visual_emb.size(1), -1)
        #     visual_emb = visual_emb.permute(0,2,1)#transpose_(1,2)
        #     visual_emb, visual_scoreAtt = self.attentionVisual(visual_emb)
        # else:
        #     visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = self.align(visual_emb)
        visual_emb = self.align_img(visual_emb)
        visual_emb = norm(visual_emb)


        # recipe embedding
        ingrNet_out = self.ingrNet_(y1, y2, opts)
        titleNet_out = self.titleNet_(z1, z2, opts)
        instNet_out = self.instNet_(v1, v3, v2, opts)
        recipe_emb = torch.cat([instNet_out, ingrNet_out, titleNet_out], 1)  # joining on the last dim
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = self.align(recipe_emb)
        recipe_emb = self.align_rec(recipe_emb)
        recipe_emb = norm(recipe_emb)

        return [visual_emb, recipe_emb]

    def forward_img(self, input, opts):  # we need to check how the input is going to be provided to the model
        x = input.cuda()

        # visual embedding
        visual_emb, w_score = self.visionMLP(x, opts)
        # if opts.visualmodel == 'simple_high1':
        #     visual_emb = visual_emb.view(visual_emb.size(0),visual_emb.size(1), -1)
        #     visual_emb = visual_emb.permute(0,2,1)#transpose_(1,2)
        #     visual_emb, visual_scoreAtt = self.attentionVisual(visual_emb)
        # else:
        #     visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = self.align(visual_emb)
        visual_emb = self.align_img(visual_emb)
        visual_emb = norm(visual_emb)


        return visual_emb


##########################################
##### task-specific tree structures ######
##########################################
class BinaryTreeLSTMLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.comp_linear.weight.data)
        torch.nn.init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """
        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class BinaryTreeLSTM(nn.Module):
    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, intra_attention,
                 gumbel_temperature, bidirectional):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature
        self.bidirectional = bidirectional

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = BinaryTreeLSTMLayer(2 * hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
        else:
            self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            torch.nn.init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            torch.nn.init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            torch.nn.init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            torch.nn.init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                torch.nn.init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                torch.nn.init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                torch.nn.init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                torch.nn.init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            torch.nn.init.kaiming_normal_(self.word_linear.weight.data)
            torch.nn.init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        torch.nn.init.normal_(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = (self.comp_query * new_h).sum(-1)
        comp_weights = comp_weights / math.sqrt(self.hidden_dim)
        if self.training:
            select_mask = basic.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = basic.greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, return_select_masks=False):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        select_masks = []

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new_zeros(batch_size, self.hidden_dim)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = self.word_linear(input)
            state = state.chunk(chunks=2, dim=2)
        nodes = []
        if self.intra_attention:
            nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i+1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i+1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)n_insts
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = basic.masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        assert h.size(1) == 1 and c.size(1) == 1
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1)
        else:
            return h.squeeze(1), c.squeeze(1), select_masks

    def forward_with_fixed_masks(self, input, length, select_masks):
        max_depth = input.size(1)
        length_mask = basic.sequence_mask(sequence_length=length,
                                          max_length=max_depth)
        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new_zeros(batch_size, self.hidden_dim)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = self.word_linear(input)
            state = state.chunk(chunks=2, dim=2)

        nodes = []
        if self.intra_attention:
            nodes.append(state[0])

        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                #new_h, new_c, select_mask, selected_h = self.select_composition(
                #    old_state=state, new_state=new_state,
                #    mask=length_mask[:, i + 1:])
                select_mask = select_masks[i]
                old_h_left, old_c_left = l
                old_h_right, old_c_right = r

                new_h, new_c = new_state

                select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
                select_mask_cumsum = select_mask.cumsum(1)
                left_mask = 1 - select_mask_cumsum
                left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
                right_mask = select_mask_cumsum - select_mask
                right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
                new_h = (select_mask_expand * new_h
                         + left_mask_expand * old_h_left
                         + right_mask_expand * old_h_right)
                new_c = (select_mask_expand * new_c
                         + left_mask_expand * old_c_left
                         + right_mask_expand * old_c_right)
                selected_h = (select_mask_expand * new_h).sum(1)

                new_state = (new_h, new_c)

                if self.intra_attention:
                    nodes.append(selected_h)

            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = basic.masked_softmax(
                logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)

        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1)


class tstsModel(nn.Module):

    def __init__(self, input_dim, embs, hidden_dim=600, use_leaf_rnn=True, intra_attention=False,
                 dropout_prob=0.2, bidirectional=False):
        super(tstsModel, self).__init__()
        self.word_embedding = embs
        self.encoder = BinaryTreeLSTM(word_dim=input_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      intra_attention=intra_attention,
                                      gumbel_temperature=1,
                                      bidirectional=bidirectional)


        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.normal_(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        # self.classifier.reset_parameters()

    def forward(self, pre, pre_length):
        pre_embeddings = self.word_embedding(pre)
        pre_embeddings = self.dropout(pre_embeddings)
        pre_h, _ = self.encoder(input=pre_embeddings, length=pre_length)
        return pre_h

    def forward_return_masks(self, pre, pre_length):
        pre_embeddings = self.word_embedding(pre)
        pre_embeddings = self.dropout(pre_embeddings)
        pre_h, _, masks = self.encoder(input=pre_embeddings, length=pre_length, return_select_masks=True)
        return pre_h, masks

    def forward_with_masks(self, pre, pre_length, select_masks):
        pre_embeddings = self.word_embedding(pre)
        pre_embeddings = self.dropout(pre_embeddings)
        pre_h, _ = self.encoder.forward_with_fixed_masks(input=pre_embeddings, length=pre_length, select_masks=select_masks)
        return pre_h


class tstsDocEncoder(nn.Module):
    def __init__(self, hid_dim, sent_encoder, bidirectional=False):
        super(tstsDocEncoder, self).__init__()
        self.sent_encoder = sent_encoder
        self.doc_encoder = BinaryTreeLSTM(
            word_dim=hid_dim,
            hidden_dim=hid_dim,
            use_leaf_rnn=False,
            intra_attention=False,
            gumbel_temperature=1,
            bidirectional=bidirectional)
        self.doc_encoder.reset_parameters()
        
    def forward(self, doc_list, n_insts, n_words_each_inst):
        # encode all sentences
        embs = []
        # iterate through all recipes in this batch
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i]
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero().shape[0]]
            emb_doc = self.sent_encoder(doc[:ln], sent_lns)
            embs.append(emb_doc)

        embs = sorted(embs, key=lambda x: -x.shape[0])
        sorted_n_ists, sorted_idx = n_insts.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        packed_seq = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)

        h, _ = self.doc_encoder(input=packed_seq.data, length=sorted_n_ists)
        unsorted_idx = original_idx.view(-1, 1).expand_as(h)
        output = h.gather(0, unsorted_idx).contiguous()
        feat = output.view(output.size(0), output.size(1))
        return feat

    def forward_with_inst_masks(self, doc_list, n_insts, n_words_each_inst, inst_masks):
        # encode all sentences
        embs = []
        # iterate through all recipes in this batch
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i]
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero().shape[0]]
            emb_doc = self.sent_encoder(doc[:ln], sent_lns)
            embs.append(emb_doc)

        embs = sorted(embs, key=lambda x: -x.shape[0])
        sorted_n_ists, sorted_idx = n_insts.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        packed_seq = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)

        h, _ = self.doc_encoder.forward_with_fixed_masks(input=packed_seq.data, length=sorted_n_ists, select_masks=inst_masks)
        unsorted_idx = original_idx.view(-1, 1).expand_as(h)
        output = h.gather(0, unsorted_idx).contiguous()
        feat = output.view(output.size(0), output.size(1))
        return feat

    def forward_with_both_masks(self, doc_list, n_insts, n_words_each_inst, sent_masks, inst_masks):
        # encode all sentences
        embs = []
        # iterate through all recipes in this batch
        for i in range(len(n_insts)):
            doc = doc_list[i]
            ln = n_insts[i]
            sent_lns = n_words_each_inst[i, :n_words_each_inst[i].nonzero().shape[0]]
            emb_doc = self.sent_encoder.forward_with_masks(doc[:ln], sent_lns, sent_masks[i if i < len(n_insts) else 0])
            embs.append(emb_doc)

        embs = sorted(embs, key=lambda x: -x.shape[0])
        sorted_n_ists, sorted_idx = n_insts.sort(0, descending=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        packed_seq = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)

        h, _ = self.doc_encoder.forward_with_fixed_masks(input=packed_seq.data, length=sorted_n_ists, select_masks=inst_masks)
        unsorted_idx = original_idx.view(-1, 1).expand_as(h)
        output = h.gather(0, unsorted_idx).contiguous()
        feat = output.view(output.size(0), output.size(1))
        return feat