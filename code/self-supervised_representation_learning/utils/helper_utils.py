import torch
from torch import nn


#Adapted from https://github.com/rasbt/stat453-deep-learning-ss21/
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        low_bound = [(x.shape[-1]-self.shape[0])//2, (x.shape[-1]-self.shape[1])//2]
        return x[:, :, low_bound[0] : low_bound[0] + self.shape[0], low_bound[1] : low_bound[1] + self.shape[1]]
    

#Adapted from https://github.com/wutong16/Density_aware_Chamfer_Distance/
class ChamferDistance(nn.Module):

    def __init__(self, variant='cd-t'):
        super().__init__()
        self.variant = variant

    def forward(self, x, y, return_raw=False):
        """
        :param a: Pointclouds Batch x dim x n
        :param b:  Pointclouds Batch x dim x m
        :return:
        -closest point on b of points from a
        -closest point on a of points from b
        -idx of closest point on b of points from a
        -idx of closest point on a of points from b
        Works for pointcloud of any dimension
        """
        x, y = x.double(), y.double()
        bs, dim, num_points_x = x.size()
        bs, dim, num_points_y = y.size()

        xx = torch.pow(x, 2).sum(1)
        yy = torch.pow(y, 2).sum(1)
        zz = torch.bmm(x.transpose(1, 2), y)

        rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
        ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
        P = rx.transpose(2, 1) + ry - 2 * zz
        
        min_for_each_x_i, idx1 = P.min(dim=2)  # shape [b, n]
        min_for_each_y_j, idx2 = P.min(dim=1)  # shape [b, m]
        
        if return_raw:
            return min_for_each_x_i, min_for_each_y_j, idx1.int(), idx2.int()
        
        Ls1s2 = min_for_each_x_i.sum(1)/num_points_x
        Ls2s1 = min_for_each_y_j.sum(1)/num_points_y          

        match self.variant:
            case 'cd-t':
                distance =  Ls1s2 + Ls2s1
            case 'cd-p':
                distance = (torch.sqrt(Ls1s2) + torch.sqrt(Ls2s1))/2
            case _:
                raise NameError('Chamfer distance variant not specified')

        return distance.mean(0)
    

#Adapted from https://github.com/wutong16/Density_aware_Chamfer_Distance/
class DensityAwareChamferDistance(nn.Module):
    
    def __init__(self, alpha = 10, n_lambda=1, non_reg = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.non_reg = non_reg
        
    def forward(self, x, gt):
        x = x.float()
        gt = gt.float()
        _, _, n_x = x.shape
        _, _, n_gt = gt.shape
        assert x.size(0) == gt.size(0)

        if self.non_reg:
            frac_12 = max(1, n_x / n_gt)
            frac_21 = max(1, n_gt / n_x)
        else:
            frac_12 = n_x / n_gt
            frac_21 = n_gt / n_x

        cd = ChamferDistance()
        
        dist1, dist2, idx1, idx2 = cd(x, gt, return_raw=True)
        # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
        # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
        # dist2 and idx2: vice versa
        exp_dist1, exp_dist2 = torch.exp(-dist1 * self.alpha), torch.exp(-dist2 * self.alpha)

        count1 = torch.zeros_like(idx2)
        count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
        weight1 = count1.gather(1, idx1.long()).float().detach() ** self.n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

        count2 = torch.zeros_like(idx1)
        count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
        weight2 = count2.gather(1, idx2.long()).float().detach() ** self.n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

        loss = (loss1 + loss2) / 2        #bounded between 0 and 1
        
        # loss = torch.tan(torch.pi*loss.double()/2)  #bounded between 0 and +inf
                        
        return loss.mean()


#Credit: @Renly_Hou in https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/10
class JSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):

        p, q = p.view(-1, p.size(1)), q.view(-1, q.size(1))
        p, q = torch.clamp(p,1e-10), torch.clamp(q,1e-10)
        m = (0.5 * (p + q)).log2()
        jsd_loss =  0.5 * (self.kl(m, p.log2()) + self.kl(m, q.log2()))

        return jsd_loss


def init_weights(func):
    def inner(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            m.weight.data = func(m.weight,gain=50)
    return inner