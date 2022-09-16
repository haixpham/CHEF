import nltk
import torch
from nlp_utils import tokenize

def to_nltk_tree(selections, sentence, show_order=False):
    ingredients = tokenize(sentence)
    return to_nltk_tree_from_list(selections, ingredients, show_order)

def to_nltk_tree_from_list(selections, ingredients_list, show_order=False):
    if not selections:
        return None

    ss = [s[0].max(0)[1] for s in selections] + [0]
    nodes = [i.lower() for i in ingredients_list]
    for k, i in enumerate(ss):
        if show_order:
            new_node = nltk.tree.Tree(f'(-{k+1}-)', [nodes[i], nodes[i+1]])
        else:
            new_node = nltk.tree.Tree('o', [nodes[i], nodes[i + 1]])
        del nodes[i:i + 2]
        nodes.insert(i, new_node)
    return nodes[0]


def get_ingredient_structure(model, ingredients_list, ingr_vocab, return_embedding=False, use_cuda=False, copy_to_cpu=True):
    """
    Return the inferred structure and final embedding of a list of ingredients

    :param model:
    :param ingredients_list:
    :param ingr_vocab:
    :param return_embedding:
    :return:  selection masks /and the embedding
    """
    # get word embedder
    word_emb = model.embs
    # get tree encoder
    tree_encoder = model.ingrNet_.ingr_LSTM.encoder

    # embed words
    ingrs = [ingr_vocab[i] for i in ingredients_list]
    ingrs_input = torch.LongTensor([ingrs])
    ingrs_len = (ingrs_input > 0).sum(1)

    if use_cuda:
        ingrs_input = ingrs_input.cuda()
        ingrs_len = ingrs_len.cuda()

    ingrs_emb = word_emb.forward(ingrs_input)

    # pass it to encoder
    h, _, select_masks = tree_encoder.forward(ingrs_emb, ingrs_len, return_select_masks=True)

    if use_cuda and copy_to_cpu:
        #h = h.cpu()
        #c = c.cpu()
        select_masks = [s.cpu() for s in select_masks]

    if return_embedding:
        if use_cuda and copy_to_cpu:
            h = h.cpu()
        return select_masks, h.squeeze(0)
    else:
        return select_masks


def get_ingredient_embedding(model, ingredients, ingr_vocab, opts, use_cuda=False):
    """
    Return the embedding of a list of ingredients

    :param model:
    :param ingredients:
    :param ingr_vocab:
    :param opts:
    :return:
    """
    # embed words
    ingrs = [ingr_vocab[i] for i in ingredients]
    ingrs_input = torch.LongTensor([ingrs])
    ingrs_len = (ingrs_input > 0).sum(1)

    if use_cuda:
        ingrs = ingrs.cuda()
        ingrs_len = ingrs_len.cuda()

    embs = model.ingrNet_.forward(ingrs_input, ingrs_len, opts)

    if use_cuda:
        embs = embs.cpu()
    return embs.squeeze(0)


def get_sentence_structure(model, sentence, ingr_vocab, return_embedding=False, use_cuda=False, copy_to_cpu=True):
    """
    return the structure of a sentence and its final embedding

    :param model:
    :param sentence:
    :param ingr_vocab:
    :param return_embedding:
    :return:
    """

    # get word embedder
    word_encoder = model.embs
    # get sentence encoder
    sent_encoder = model.instNet_.doc_encoder.sent_encoder.encoder

    instruction = tokenize(sentence)
    instruction = [ingr_vocab[i] for i in instruction]
    sent = torch.LongTensor([instruction])
    sent_ln = (sent > 0).sum(1)

    if use_cuda:
        sent = sent.cuda()
        sent_ln = sent_ln.cuda()

    word_emb = word_encoder.forward(sent)
    h, _, select_masks = sent_encoder.forward(word_emb, sent_ln, return_select_masks=True)

    if use_cuda and copy_to_cpu:
        select_masks = [s.cpu() for s in select_masks]

    if return_embedding:
        if use_cuda and copy_to_cpu:
            h = h.cpu()
        return select_masks, h.squeeze(0)
    else:
        return select_masks


def get_sentence_embedding(model, sentence, ingr_vocab, use_cuda=False):
    """
    return the final embedding of the sentence

    :param model:
    :param sentence:
    :param ingr_vocab:
    :param return_embedding:
    :return:
    """

    # get sentence encoder
    sent_encoder = model.instNet_.doc_encoder.sent_encoder

    instruction = tokenize(sentence)
    instruction = [ingr_vocab[i] for i in instruction]
    sent = torch.LongTensor([instruction])
    sent_ln = (sent > 0).sum(1)

    if use_cuda:
        sent = sent.cuda()
        sent_ln = sent_ln.cuda()

    h = sent_encoder.forward(sent, sent_ln)

    if type(h) is tuple or type(h) is list:
        h = h[0]

    if use_cuda:
        h = h.cpu()

    return h.squeeze(0)


def get_instruction_structure(model, instructions, ingr_vocab, return_embedding=False, use_cuda=False, copy_to_cpu=True):
    """
    Return the structure of sentences (at the whole instruction-level), and final embedding

    :param model:
    :param instructions:
    :param ingr_vocab:
    :param return_embedding:
    :return:
    """

    intrs = [tokenize(i) for i in instructions]
    for r in range(len(intrs)):
        for c in range(len(intrs[r])):
            intrs[r][c] = ingr_vocab[intrs[r][c]]
    #word_encoder = model.embs
    sent_encoder = model.instNet_.doc_encoder.sent_encoder
    doc_encoder = model.instNet_.doc_encoder.doc_encoder

    # get embedding for each sentence
    if use_cuda:
        sents = []
        for i in range(len(intrs)):
            sent = torch.cuda.LongTensor([intrs[i]])
            sent_ln = (sent > 0).sum(1)
            h = sent_encoder.forward(sent, sent_ln)
            sents.append(h)

        # get embedding for the whole instructions
        sent_num = torch.cuda.LongTensor([len(sents)])
        sent_input = torch.cat(sents, dim=0)
        sent_input = sent_input.unsqueeze(0)

        h, _, select_masks = doc_encoder.forward(sent_input.data, sent_num, return_select_masks=True)
        if copy_to_cpu:
            select_masks = [s.cpu() for s in select_masks]
        if return_embedding:
            if copy_to_cpu:
                h = h.cpu()
            return select_masks, h.squeeze(0)
        else:
            return select_masks
    else:
        sents = []
        for i in range(len(intrs)):
            sent = torch.LongTensor([intrs[i]])
            sent_ln = (sent > 0).sum(1)
            h = sent_encoder.forward(sent, sent_ln)
            sents.append(h)

        # get embedding for the whole instructions
        sent_num = torch.LongTensor([len(sents)])
        sent_input = torch.cat(sents, dim=0)
        sent_input = sent_input.unsqueeze(0)

        h, _, select_masks = doc_encoder.forward(sent_input.data, sent_num, return_select_masks=True)
        if return_embedding:
            return select_masks, h.squeeze(0)
        else:
            return select_masks

def get_instruction_structure_full(model, instructions, ingr_vocab, return_embedding=False, use_cuda=False, copy_to_cpu=True):
    """
    Return the structures at sentence-level of all sentences, structure of instruction, and embeddings

    :param model:
    :param instructions:
    :param ingr_vocab:
    :param return_embedding:
    :return:
    """
    intrs = [tokenize(i) for i in instructions]
    for r in range(len(intrs)):
        for c in range(len(intrs[r])):
            intrs[r][c] = ingr_vocab[intrs[r][c]]
    word_encoder = model.embs
    sent_encoder = model.instNet_.doc_encoder.sent_encoder.encoder
    doc_encoder = model.instNet_.doc_encoder.doc_encoder

    if use_cuda:
        # get embedding for each sentence
        sents = []
        for i in range(len(intrs)):
            sent = torch.cuda.LongTensor([intrs[i]])
            sent_ln = (sent > 0).sum(1)
            word_emb = word_encoder.forward(sent)
            h, _, select_masks = sent_encoder.forward(word_emb, sent_ln, return_select_masks=True)
            if copy_to_cpu:
                select_masks = [s.cpu() for s in select_masks]
            sents.append((h, select_masks))
        # get embedding for the whole instructions
        sent_num = torch.cuda.LongTensor([len(sents)])
        sent_input = [sent[0] for sent in sents]
        sent_input = torch.cat(sent_input, dim=0)
        sent_input = sent_input.unsqueeze(0)
        h, _, select_masks = doc_encoder.forward(sent_input.data, sent_num, return_select_masks=True)
        if copy_to_cpu:
            select_masks = [s.cpu() for s in select_masks]
        if return_embedding:
            if copy_to_cpu:
                sents = ((s[0].cpu(), s[1]) for s in sents)
            return sents, h.squeeze(0), select_masks
        else:
            sentence_masks = [sent[1] for sent in sents]
            return sentence_masks, select_masks
    else:
        # get embedding for each sentence
        sents = []
        for i in range(len(intrs)):
            sent = torch.LongTensor([intrs[i]])
            sent_ln = (sent > 0).sum(1)
            word_emb = word_encoder.forward(sent)
            h, _, select_masks = sent_encoder.forward(word_emb, sent_ln, return_select_masks=True)
            sents.append((h, select_masks))
        # get embedding for the whole instructions
        sent_num = torch.LongTensor([len(sents)])
        sent_input = [sent[0] for sent in sents]
        sent_input = torch.cat(sent_input, dim=0)
        sent_input = sent_input.unsqueeze(0)
        h, _, select_masks = doc_encoder.forward(sent_input.data, sent_num, return_select_masks=True)
        if return_embedding:
            return sents, h.squeeze(0), select_masks
        else:
            sentence_masks = [sent[1] for sent in sents]
            return sentence_masks, select_masks

def get_instruction_embedding(model, instructions, ingr_vocab, use_cuda=False):
    """
    Return final embedding of instruction

    :param model:
    :param instructions:
    :param ingr_vocab:
    :return:
    """

    intrs = [tokenize(i) for i in instructions]
    for r in range(len(intrs)):
        for c in range(len(intrs[r])):
            intrs[r][c] = ingr_vocab[intrs[r][c]]
    doc_encoder = model.instNet_.doc_encoder

    # get embedding for each sentence
    if use_cuda:
        sents = []
        for i in range(len(intrs)):
            sent = torch.cuda.LongTensor(intrs[i])
            sents.append(sent)
        insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
        insts = insts.unsqueeze(0)
        insts_ln = (insts > 0).sum(2)
        insts_num = (insts_ln > 0).sum(1)
        # insts = insts.unsqueeze(0)
        # insts_ln = insts_ln.unsqueeze(0)
        # insts_num = insts_num.unsqueeze(0)

        # get embedding for the whole instructions
        h = doc_encoder.forward(insts, insts_num, insts_ln)
        h = h.cpu()
        return h.squeeze(0)
    else:
        sents = []
        for i in range(len(intrs)):
            sent = torch.LongTensor(intrs[i])
            sents.append(sent)
        insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
        insts = insts.unsqueeze(0)
        insts_ln = (insts > 0).sum(2)
        insts_num = (insts_ln > 0).sum(1)
        #insts = insts.unsqueeze(0)
        #insts_ln = insts_ln.unsqueeze(0)
        #insts_num = insts_num.unsqueeze(0)

        # get embedding for the whole instructions
        h = doc_encoder.forward(insts, insts_num, insts_ln)
        return h.squeeze(0)


def get_ingredient_attention(model, ingredients, ingr_vocab, opts, use_cuda=False):
    # embed words
    ingrs = [ingr_vocab[i] for i in ingredients]
    ingrs_input = torch.LongTensor([ingrs])
    ingrs_len = (ingrs_input > 0).sum(1)

    if use_cuda:
        ingrs_input = ingrs_input.cuda()
        ingrs_len = ingrs_len.cuda()

    embs, attn_scores = model.ingrNet_.forward(ingrs_input, ingrs_len, opts, return_all=True)

    if use_cuda:
        embs = embs.cpu()
        attn_scores = attn_scores.cpu()
    embs = embs.squeeze(0)
    attn_scores = attn_scores.squeeze(0).detach().numpy()

    # create a json object
    obj = {}
    sum = 0.0
    for i, ingr in enumerate(ingredients):
        obj[ingr] = attn_scores[i].item()
        sum = sum + obj[ingr]
    # scale weights so they sum to 1
    for ingr in ingredients:
        obj[ingr] = obj[ingr] / sum
    return embs, obj


def get_instruction_attention(model, instructions, ingr_vocab, use_cuda=False):
    """
    Return final embedding of instruction

    :param model:
    :param instructions:
    :param ingr_vocab:
    :return:
    """

    intrs = [tokenize(i) for i in instructions]
    for r in range(len(intrs)):
        for c in range(len(intrs[r])):
            intrs[r][c] = ingr_vocab[intrs[r][c]]
    doc_encoder = model.instNet_.doc_encoder

    # get embedding for each sentence
    if use_cuda:
        sents = []
        for i in range(len(intrs)):
            sent = torch.cuda.LongTensor(intrs[i])
            sents.append(sent)
        insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
        insts = insts.unsqueeze(0)
        insts_ln = (insts > 0).sum(2)
        insts_num = (insts_ln > 0).sum(1)
        # insts = insts.unsqueeze(0)
        # insts_ln = insts_ln.unsqueeze(0)
        # insts_num = insts_num.unsqueeze(0)

        # get embedding for the whole instructions
        h, inst_scores, sentence_scores = doc_encoder.forward(insts, insts_num, insts_ln, return_all=True)
        h = h.cpu().squeeze(0)
        inst_scores = inst_scores.cpu().squeeze(0).detach().numpy()
        sentence_scores = [s.cpu().squeeze(0).detach().numpy() for s in sentence_scores[0]]
    else:
        sents = []
        for i in range(len(intrs)):
            sent = torch.LongTensor(intrs[i])
            sents.append(sent)
        insts = torch.nn.utils.rnn.pad_sequence(sents, batch_first=True)
        insts = insts.unsqueeze(0)
        insts_ln = (insts > 0).sum(2)
        insts_num = (insts_ln > 0).sum(1)
        #insts = insts.unsqueeze(0)
        #insts_ln = insts_ln.unsqueeze(0)
        #insts_num = insts_num.unsqueeze(0)

        # get embedding for the whole instructions
        h, inst_scores, sentence_scores = doc_encoder.forward(insts, insts_num, insts_ln, return_all=True)
        h = h.squeeze(0)
        inst_scores = inst_scores.squeeze(0).detach().numpy()
        sentence_scores = [s.squeeze(0).detach().numpy() for s in sentence_scores[0]]

    obj = {}
    #inst_scores = inst_scores.tolist()
    for i in range(len(instructions)):
        obj[str(i)] = {}
        obj[str(i)]["sent_score"] = float(inst_scores[i])
        obj[str(i)]["each_sent_score"] = []
        tokens = tokenize(instructions[i])
        for k, word in enumerate(tokens):
            obj[str(i)]["each_sent_score"].append({word: float(sentence_scores[i][k])})
    return h, obj


def expand_structure_along_batch_axis(masks, N):
    new_masks = [mask.expand(N, -1) for mask in masks]
    return new_masks


def get_ingredient_structure_batch(model, ingrs, ingrs_ln, return_embedding=False, use_cuda=True, copy_to_cpu=False):
    # get word embedder
    word_emb = model.embs
    # get tree encoder
    tree_encoder = model.ingrNet_.ingr_LSTM.encoder

    # embed words
    if use_cuda and (not ingrs.is_cuda):
        ingrs = ingrs.cuda()
        ingrs_ln = ingrs_ln.cuda()

    ingrs_emb = word_emb.forward(ingrs)

    # pass it to encoder
    h, _, select_masks = tree_encoder.forward(ingrs_emb, ingrs_ln, return_select_masks=True)

    if use_cuda and copy_to_cpu:
        select_masks = [s.cpu() for s in select_masks]

    if return_embedding:
        if use_cuda and copy_to_cpu:
            h = h.cpu()
        return select_masks, h.squeeze(0)
    else:
        return select_masks


def get_sentence_structure_batch(model, inst, inst_ln, inst_num, return_embedding=True, use_cuda=True, copy_to_cpu=False):
    # get sentence encoder
    sent_encoder = model.instNet_.doc_encoder.sent_encoder

    if use_cuda and (not inst.is_cuda):
        inst = inst.cuda()
        inst_ln = inst_ln.cuda()
    Num = len(inst_num)
    sent_masks = []
    sent_embs = []
    for i in range(Num):    # for each recipe
        doc = inst[i]
        ln = inst_num[i]
        sent_ln = inst_ln[i, :inst_ln[i].nonzero().shape[0]]
        h, mask = sent_encoder.forward_return_masks(doc[:ln], sent_ln)
        if use_cuda and copy_to_cpu:
            mask = [m.cpu() for m in mask]
            if return_embedding:
                h = h.cpu()
        sent_masks.append(mask)
        if return_embedding:
            sent_embs.append(h)
    if return_embedding:
        return sent_masks, sent_embs
    else:
        return sent_masks


def get_instruction_structure_batch(model, inst_embs, inst_num, use_cuda=True, copy_to_cpu=False):
    Num = len(inst_num)
    if use_cuda and (not inst_num.is_cuda):
        inst_num = inst_num.cuda()
        for i in range(Num):
            inst_embs[i] = inst_embs[i].cuda()

    doc_encoder = model.instNet_.doc_encoder.doc_encoder

    sent_embs = sorted(inst_embs, key=lambda x: -x.shape[0])
    sorted_n_ists, sorted_idx = inst_num.sort(0, descending=True)
    _, original_idx = sorted_idx.sort(0, descending=False)
    packed_seq = torch.nn.utils.rnn.pad_sequence(sent_embs, batch_first=True)

    _, _, select_masks = doc_encoder.forward(input=packed_seq.data, length=sorted_n_ists, return_select_masks=True)
    #original_idx = original_idx.tolist()
    #select_masks = masks[original_idx]
    # resort them to original order
    unsorted_masks = []
    for mask in select_masks:
        unsorted_idx = original_idx.view(-1, 1).expand_as(mask)
        output = mask.gather(0, unsorted_idx).contiguous()
        output = output.view(output.size(0), output.size(1))
        unsorted_masks.append(output)
    select_masks = unsorted_masks
    if use_cuda and copy_to_cpu:
        select_masks = [mask.cpu() for mask in select_masks]
    return select_masks



def get_instruction_structure_batch_2(model, inst, inst_ln, inst_num, use_cuda=True, copy_to_cpu=False):
    # get sentence encoder
    sent_encoder = model.instNet_.doc_encoder.sent_encoder
    #doc_encoder = model.instNet_.doc_encoder.doc_encoder

    if use_cuda and (not inst.is_cuda):
        inst = inst.cuda()
        inst_ln = inst_ln.cuda()
        inst_num = inst_num.cuda()
    Num = len(inst_num)
    sent_embs = []
    for i in range(Num):    # for each recipe
        doc = inst[i]
        ln = inst_num[i]
        sent_ln = inst_ln[i, :inst_ln[i].nonzero().shape[0]]
        h = sent_encoder.forward(doc[:ln], sent_ln)
        sent_embs.append(h)

    select_masks = get_instruction_structure_batch(model, sent_embs, inst_num, use_cuda, copy_to_cpu)
    return select_masks