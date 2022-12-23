import torch
from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
import itertools


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)



class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x, text, H, W):
        return x + self.m(x, text, H, W)


class Attentioncat(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd
        self.kv = Linear_BN(dim, h)
        self.q = Linear_BN(dim, nh_kd)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh+1000, dim, bn_weight_init=0))

        points = list(itertools.product(range(100), range(100)))
        N = len(points)
        self.attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    self.attention_offsets[offset] = len(self.attention_offsets)
                idxs.append(self.attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(self.attention_offsets)))

    def forward(self, x, text, H, W):  # x (B,N,C)
        B, N, C = x.shape
        kv = self.kv(x)
        k, v = kv.view(B, N, self.num_heads, -
                          1).split([self.key_dim, self.d], dim=3)


        NT, _ = text.shape
        q = self.q(text.unsqueeze(0)).view(NT, self.num_heads, -1).contiguous().unsqueeze(0).repeat(B,1,1,1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.contiguous().view(B, N, -1).contiguous()


        points = list(itertools.product(range(H), range(W)))
        row, col = int((NT - 0.5) / 100), (NT-1) % 100
        pointst = list(itertools.product(range(row+1), range(100)))

        idxs = []
        for iid, p1 in enumerate(pointst):
            if iid == NT:
                break
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    print(H, W, offset)
                idxs.append(self.attention_offsets[offset])
        attention_bias_idxs = torch.LongTensor(idxs).view(NT, N).cuda()

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, attention_bias_idxs])
        )
        attn = attn.softmax(dim=-1)
        v = torch.cat((v, attn.view(B, -1, N).permute(0, 2, 1).contiguous()), -1)
        comp = torch.zeros(B, N, 1000-NT*self.num_heads).cuda()
        v = torch.cat((v, comp), -1)
        x = self.proj(v)
        return x




class Attentioninv(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd
        self.kv = Linear_BN(dim, h)
        self.q = Linear_BN(dim, nh_kd)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(100), range(100)))
        N = len(points)
        self.attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    self.attention_offsets[offset] = len(self.attention_offsets)
                idxs.append(self.attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(self.attention_offsets)))


    def forward(self, text, x, H, W):  # x (B,N,C)
        B, N, C = x.shape
        kv = self.kv(x)
        k, v = kv.view(B, N, self.num_heads, -
                          1).split([self.key_dim, self.d], dim=3)


        NT, _ = text.shape
        q = self.q(text.unsqueeze(0)).view(NT, self.num_heads, -1).contiguous().unsqueeze(0).repeat(B,1,1,1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)


        points = list(itertools.product(range(H), range(W)))
        row, col = int((NT - 0.5) / 100), (NT-1) % 100
        pointst = list(itertools.product(range(row+1), range(100)))

        idxs = []
        for iid, p1 in enumerate(pointst):
            if iid == NT:
                break
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    assert(offset in self.attention_offsets)
                idxs.append(self.attention_offsets[offset])
        attention_bias_idxs = torch.LongTensor(idxs).view(NT, N).cuda()

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, attention_bias_idxs])
        )
        attn = attn.softmax(dim=-1)
        text = (attn @ v).transpose(1, 2).reshape(B, NT, self.dh)
        text = self.proj(text)
        
        return text




class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd
        self.kv = Linear_BN(dim, h)
        self.q = Linear_BN(dim, nh_kd)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(100), range(100)))
        N = len(points)
        self.attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    self.attention_offsets[offset] = len(self.attention_offsets)
                idxs.append(self.attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(self.attention_offsets)))
        


    def forward(self, x, text, H, W):  # x (B,N,C)
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.num_heads, -1).contiguous()


        BT, NT, CT = -1, -1, -1
        kv = None
        if isinstance(text, list):
            textun = [text[i].unsqueeze(0) for i in range(len(text))]
            text = torch.cat(textun, 0)
            BT, NT, CT = text.shape
            kv = self.kv(text).view(BT, NT, self.num_heads, -1).contiguous()
        else:
            NT, _ = text.shape
            kv = self.kv(text.unsqueeze(0)).view(NT, self.num_heads, -1).contiguous().unsqueeze(0).repeat(B,1,1,1)
        k, v = kv.split([self.key_dim, self.d], dim=3)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)


        points = list(itertools.product(range(H), range(W)))
        row, col = int((NT - 0.5) / 100), (NT-1) % 100
        pointst = list(itertools.product(range(row+1), range(100)))
        idxs = []
        for p1 in points:
            for iid, p2 in enumerate(pointst):
                if iid == NT:
                    break
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in self.attention_offsets:
                    assert(offset in self.attention_offsets)
                idxs.append(self.attention_offsets[offset])
        attention_bias_idxs = torch.LongTensor(idxs).view(N, NT).cuda()

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, attention_bias_idxs])
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x






class CROSSTRANS(torch.nn.Module):
    def __init__(self, in_chans=1024, embed_dim=512, key_dim=64, mlp_activation=torch.nn.Hardswish, attention_activation=torch.nn.Hardswish, nh=8, ar=2, drop_path=0, mr=2):
        super().__init__()
        
        self.de_chan = torch.nn.Sequential(
                             Linear_BN(in_chans, embed_dim),
                             mlp_activation(),)
        self.de_chan_text = None
        self.embed_dim = embed_dim
        if embed_dim < 512:
            self.de_chan_text = torch.nn.Sequential(
                             Linear_BN(512, embed_dim),
                             mlp_activation(),)
        

        self.use_word = True
        self.resa, self.ffn = None, None
        if not self.use_word:
            self.resa = Residual(Attention(
                        embed_dim, key_dim, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                    ), drop_path)
            h = int(embed_dim * mr)
            self.ffn = torch.nn.Sequential(
                            Linear_BN(embed_dim, h),
                            mlp_activation(),)
        
        self.text, self.ffntext = None, None
        if self.use_word: 
            self.text = Residual(Attentioninv(
                        embed_dim, key_dim, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                    ), drop_path)
            h = int(embed_dim * mr)
            if embed_dim < 512:
                self.ffntext = torch.nn.Sequential(
                            Linear_BN(embed_dim, h),
                            mlp_activation(),)
            else:
                self.ffntext = torch.nn.Sequential(
                            Linear_BN(embed_dim, h),
                            mlp_activation(),
                            Linear_BN(h, embed_dim, bn_weight_init=0),
                        )


    def forward(self, im_emd, text_emd):
        x = im_emd
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        xd = self.de_chan(x)  

        if self.use_word:
            text_emd2 = text_emd
            if self.de_chan_text is not None:
                text_emd2 = self.de_chan_text(text_emd.unsqueeze(0)) 
                text_emd2 = torch.squeeze(text_emd2)

            textd = self.text(text_emd2, xd, H, W)
            if self.embed_dim < 512:
                textd = self.ffntext(textd)[0,:,:].clone() + text_emd.detach()
                return textd
            else:
                textd = self.ffntext(textd) + textd
                return textd[0,:,:].clone()
        else:
            xd2 = self.resa(xd, text_emd, H, W)
            xd2 = x + self.ffn(xd2)
            im_emd['res4'] = xd2.permute(0, 2, 1).view(B, C, H, W).contiguous()
            return im_emd

     
