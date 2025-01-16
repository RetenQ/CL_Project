import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from biglm import BIGLM
from data import Vocab, DataLoader, s2t, s2xy_polish
# 加载模型状态：从.ckpt文件中加载模型权重和配置参数。
# 构建词汇表：从.txt文件中读取词汇信息，构建词汇表对象。
# 初始化并加载模型：使用词汇表和配置参数初始化BIGLM模型，并将加载的权重应用到模型上。
# 设置模型为评估模式：将模型设置为eval模式，这在进行模型评估或推理时是必要的。
gpu = 0
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

m_path = "./songci.ckpt"
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "/home/data/GYX/llm/model/vocab.txt")


MAX_LEN = 300
k = 32
def top_k_inc(enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
    start = time.time()
    incremental_state = None
    inp_y, m = s2t(s, lm_vocab)
    inp_y = inp_y.cuda(gpu)
    res = []
    for l in range(inp_ys_tpl.size(0)):
        probs, pred, incremental_state = lm_model.work_incremental(enc, src_padding_mask, \
                                         inp_y, inp_ys_tpl[0:l+1,:], inp_ys_seg[0:l+1,:], inp_ys_pos[0:l+1,:],\
                                         incremental_state)
        next_tk = []
        for i in range(len(s)):
            ctk = lm_vocab.idx2token(inp_ys_tpl[l,i].item())
            if ctk != "<c1>" and ctk != "<c2>" and ctk != "<c0>":
                next_tk.append(ctk)
                continue
            
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
            else:
                logits = probs[0, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        inp_y, m = s2t(s, lm_vocab)
        inp_y = inp_y.cuda(gpu)
        bidx = torch.BoolTensor(bidx).cuda(gpu)
        incremental_state["bidx"] = bidx
    res += s_
        
    #for i in res:
    #    print(''.join(i))
    print(time.time()-start)
    return res

def top_k(enc, src_padding_mask, inp_ys_tpl, inp_ys_seg, inp_ys_pos, s):
    inp_y, m = s2t(s, lm_vocab)
    inp_y = inp_y.cuda(gpu)

    start = time.time()
    res = []
    for l in range(inp_ys_tpl.size(0)):
        probs, pred = lm_model.work(enc, src_padding_mask, inp_y, inp_ys_tpl[0:l+1,:], inp_ys_seg[0:l+1,:], inp_ys_pos[0:l+1,:])
        next_tk = []
        for i in range(len(s)):
            ctk = lm_vocab.idx2token(inp_ys_tpl[l,i].item())
            if ctk != "<c1>" and ctk != "<c2>" and ctk != "<c0>":
                next_tk.append(ctk)
                continue
            logits = probs[len(s[i]) - 1, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        for sent, t in zip(s, next_tk):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        inp_y, m = s2t(s, lm_vocab)
        inp_y = inp_y.cuda(gpu)

    res += s_
        
    #for i in res:
    #    print(''.join(i))

    #print(time.time()-start)
    return res
   



ds = []
with open("/home/data/GYX/llm/data/polish_tpl2.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            ds.append(line)
print(len(ds))

local_rank = gpu
batch_size = 1
cp_size = 1
batches = round(len(ds) / batch_size)

for i in range(5):
    fo = open("/home/data/GYX/llm/results/out"+str(i+1)+".txt", "w")
    idx = 0
    while idx < len(ds):
        lb = ds[idx:idx + batch_size]
        cplb = []
        for line in lb:
            cplb += [line for i in range(cp_size)]
        print(cplb) 
        xs_tpl, xs_seg, xs_pos, \
        ys_truth, ys_inp, \
        ys_tpl, ys_seg, ys_pos, msk = s2xy_polish(cplb, lm_vocab, lm_args.max_len,2)

        xs_tpl = xs_tpl.cuda(local_rank)
        xs_seg = xs_seg.cuda(local_rank)
        xs_pos = xs_pos.cuda(local_rank)
        ys_tpl = ys_tpl.cuda(local_rank)
        ys_seg = ys_seg.cuda(local_rank)
        ys_pos = ys_pos.cuda(local_rank)

        enc, src_padding_mask = lm_model.encode(xs_tpl, xs_seg, xs_pos)
        s = [['<bos>']] * batch_size * cp_size   
        res = top_k_inc(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos, s)

        for i, line in enumerate(cplb):
            r = ''.join(res[i])
            print(line)
            print(r)
            fo.write(line + "\t" + r + "\n")  

        idx += batch_size
    fo.close()
