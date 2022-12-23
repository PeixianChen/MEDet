import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)

    # print("attn:")
    # print(attn[0])
    if weight is not None:
      attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out

def cosine_similarity_a2a(x1, x2, dim=1, eps=1e-8):
    #x1: (B, n, d) x2: (B, m, d)
    w12 = torch.bmm(x1, x2.transpose(1,2))
    #w12: (B, n, m)

    w1 = torch.norm(x1, 2, dim).unsqueeze(2)
    w2 = torch.norm(x2, 2, dim).unsqueeze(1)

    #w1: (B, n, 1) w2: (B, 1, m)
    w12_norm = torch.bmm(w1, w2).clamp(min=eps)
    return w12 / w12_norm

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))

class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc_local = nn.Linear(img_dim, embed_size)
        #self.fc_global = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_local.in_features +
                                  self.fc_local.out_features)
        self.fc_local.weight.data.uniform_(-r, r)
        self.fc_local.bias.data.fill_(0)

        #self.fc_global.weight.data.uniform_(-r, r)
        #self.fc_global.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        #img_global = images.mean(1)
        #feat_global = self.fc_global(img_global)
        feat_local = self.fc_local(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            feat_local = l2norm(feat_local, dim=-1)
            #feat_global = l2norm(feat_global, dim=-1)

        return feat_local 

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)

class IMRAM(nn.Module):
    def __init__(self, embed_size, iteration_step, raw_feature_norm, lambda_softmax, no_IMRAM_norm):
        super(IMRAM, self).__init__()
        # Build Models
        print("*********using gate to fusion information**************")
        self.linear_t2i = nn.Linear(embed_size * 2, embed_size)
        self.gate_t2i = nn.Linear(embed_size * 2, embed_size)
        self.linear_i2t = nn.Linear(embed_size * 2, embed_size)
        self.gate_i2t = nn.Linear(embed_size * 2, embed_size)

        self.iteration_step = iteration_step
        self.raw_feature_norm = raw_feature_norm
        self.lambda_softmax = lambda_softmax
        self.no_IMRAM_norm = no_IMRAM_norm

        # self.encoder_image = EncoderImagePrecomp(512, embed_size)

    def gated_memory_t2i(self, input_0, input_1):
        
        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_t2i(input_cat))
        gate = torch.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output
    
    def gated_memory_i2t(self, input_0, input_1):

        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_i2t(input_cat))
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def forward_score(self, img_emb, cap_emb, cap_lens, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # compute image-sentence score matrix
        # if self.opt.model_mode == "full_IMRAM":
        # img_emb = self.encoder_image(img_emb)
        scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
        scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
        scores_t2i = torch.stack(scores_t2i, 0).sum(0)
        scores_i2t = torch.stack(scores_i2t, 0).sum(0)
        score = scores_t2i + scores_i2t
        # elif self.opt.model_mode == "image_IMRAM":
        #     scores_i2t = self.xattn_score_Image_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
        #     scores_i2t = torch.stack(scores_i2t, 0).sum(0)
        #     score = scores_i2t
        # elif self.opt.model_mode == "text_IMRAM":
        #     scores_t2i = self.xattn_score_Text_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
        #     scores_t2i = torch.stack(scores_t2i, 0).sum(0)
        #     score = scores_t2i
        return score
        
    def forward(self, img_emb, cap_emb, cap_lens, *args):
        """One training step given images and captions.
        """
        # compute the embeddings
        # print("pos_labels:")
        # print(pos_labels)
        scores = self.forward_score(img_emb, cap_emb, cap_lens)
        return scores
    
    def xattn_score_Text_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = len(captions_all)
        images = images.float()
        # captions_all = captions_all.float()
        for i in range(n_caption):
            # same words: captions_all == captions_all[i]
            same_word = []
            for cap in captions_all:
                if torch.equal(cap, captions_all[i]):
                    same_word.append(1)
                else:
                    same_word.append(0)
            same_word[i] = 0

            # Get the i-th text description
            n_word = cap_lens[i]
            # cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i = captions_all[i].unsqueeze(0).contiguous().float()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            
            query = cap_i_expand
            context = images
            weight = 0
            for j in range(self.iteration_step):
                # "feature_update" by default:
                attn_feat, _ = func_attention(query, context, raw_feature_norm=self.raw_feature_norm, smooth=self.lambda_softmax)
                row_sim = cosine_similarity(cap_i_expand, attn_feat, dim=2)
                # clear neg pairs:
                # row_sim *= (1 - torch.tensor(same_word).cuda()).unsqueeze(1)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                row_sim[torch.tensor(same_word)==1, ...] = -1
                similarities[j].append(row_sim)

                # query = attn_feat
                query = self.gated_memory_t2i(query, attn_feat)

                if not self.no_IMRAM_norm:
                    query = l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)
        return new_similarities

    def xattn_score_Image_IMRAM(self, images, captions_all, cap_lens):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = len(captions_all)
        n_region = images.size(1)
        images = images.float()
        # captions_all = captions_all.float()
        for i in range(n_caption):
            # same words: captions_all == captions_all[i]
            same_word = []
            for cap in captions_all:
                if torch.equal(cap, captions_all[i]):
                    same_word.append(1)
                else:
                    same_word.append(0)
            same_word[i] = 0
            # Get the i-th text description
            n_word = cap_lens[i]
            # cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i = captions_all[i].unsqueeze(0).contiguous().float()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            
            query = images
            context = cap_i_expand
            weight = 0
            for j in range(self.iteration_step):
                attn_feat, _ = func_attention(query, context, raw_feature_norm=self.raw_feature_norm, smooth=self.lambda_softmax)
                row_sim = cosine_similarity(images, attn_feat, dim=2)
                # row_sim * same word, clear neg pairs:
                # row_sim *= (1 - torch.tensor(same_word).cuda()).unsqueeze(1)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                row_sim[torch.tensor(same_word)==1, ...] = -1
                similarities[j].append(row_sim)

                # query = attn_feat
                query = self.gated_memory_i2t(query, attn_feat)

                if not self.no_IMRAM_norm:
                    query = l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)

        return new_similarities