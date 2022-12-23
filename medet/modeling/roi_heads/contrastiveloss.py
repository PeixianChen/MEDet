import torch 
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, hardcase=False, diagonal=None):
        pos_index = scores.shape[0] // 2

        # if not hardcase:
        #     diagonal = scores.diag().view(-1, 1)
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        if not hardcase:
            I = torch.eye(scores.size(0)) > .5
            if torch.cuda.is_available():
                I = I.cuda()
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)
        # I_false = torch.zeros_like(I[pos_index:, ...]) > 0
        # I[pos_index:, ...] = I_false
        
        # if hardcase:
        #     # I_true = torch.ones_like(I[..., pos_index:]) > 0
        #     # I[..., pos_index:] = I_true
        #     # cost_s = cost_s.masked_fill_(I, 0)[:, :pos_index]
        #     # cost_im = cost_im.masked_fill_(I, 0)[:pos_index, :]
        #     I[..., pos_index:]  = I[..., :pos_index] 
        # # else:

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum(), diagonal

