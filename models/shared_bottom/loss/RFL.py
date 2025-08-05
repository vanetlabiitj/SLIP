import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class RFL(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, gamma_class_ng=1.2, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, **kwargs):
        super(RFL, self).__init__()
        #print("RFL kwargs received:", locals())  # or vars(self) to see assigned values

        self.gamma_neg = gamma_neg
        self.gamma_class_ng=gamma_class_ng
        self.gamma_class_pos=1
        self.gamma_pos = 0
        self.clip = clip
        self.disable_torch_grad_focal_loss = True
        self.eps = eps
        self.city = str(kwargs.get('city'))
        self.num_regions = int(kwargs.get('num_nodes', 1))
        self.num_outputs = int(kwargs.get('output_dim', 1))
        if self.city == 'LA':
           self.distribution_path = "./dataset/LA/crime_counts.npy"
        else:
           self.distribution_path = "./dataset/CHI/crime_counts.npy"

    def _create_class_weight(self, distribution_path):
        crime_counts = np.load(distribution_path)  # shape: [num_rows, num_classes]
        # Compute row-wise probabilities
        row_sums = crime_counts.sum(axis=1, keepdims=True)  # shape: [num_rows, 1]
        prob = crime_counts / row_sums  # each row sums to 1
        # Convert to torch tensor
        prob = torch.FloatTensor(prob)
        prob = prob.clamp(min=1e-6)
        # Compute weights
        weight = (-prob.log() + 1)
        return weight


    def _create_co_occurrence_matrix(self, y):
        co_occurrence_matrix = np.dot(y.T, y)  # shape: [num_classes, num_classes]
        col_sum = co_occurrence_matrix.sum(axis=0)  # shape: [num_classes]
        # Prevent division by zero by adding a small epsilon where col_sum is 0
        col_sum = np.where(col_sum == 0, 1e-6, col_sum)
        norm_matrix = co_occurrence_matrix / col_sum  # safe column-wise normalization
        co_matrix = torch.tensor(norm_matrix, dtype=torch.float32)
        return co_matrix


    def forward(self, x, y):

        # postive -
        x_sigmoid = torch.pow(torch.sigmoid(x),1)
        xs_pos = (x_sigmoid ) * self.gamma_class_pos
        xs_neg = 1 - (x_sigmoid)

        # negtive +
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # weight calculation
        weight = self._create_class_weight(self.distribution_path)
        expanded_weights = weight.view(-1)
        targets_nonzero = y.gt(0).float()
        log_factorial = torch.lgamma(y + 1)

        #nb calculation
        nb_mean = torch.exp(x)  # avoid log(0)
        nb_disp = F.softplus(x)
        targets = y.view(-1,  self.num_regions, 8)
        zero_mask = (targets == 0)
        zero_ratio_per_region = zero_mask.sum(dim=2) / 8  # 8 labels per region
        sparse_regions_mask = zero_ratio_per_region > 0.5
        num_sparse_regions = sparse_regions_mask.sum(dim=1)
        trigger_mask = num_sparse_regions > (0.1 * self.num_regions)  # [15] boolean
        trigger_mask_expanded = trigger_mask.view(-1, 1, 1).expand(-1, self.num_regions, 8)
        trigger = trigger_mask_expanded.reshape(-1, self.num_outputs)

        # Basic CE calculation

        los_pos_nb = targets_nonzero * ( torch.lgamma(y + nb_disp) - torch.lgamma(nb_disp)
                     - log_factorial + nb_disp * (torch.log(nb_disp) - torch.log(nb_disp + nb_mean)) + y * (torch.log(nb_mean) - torch.log(nb_disp + nb_mean)))

        los_pos_bce = y * torch.log(xs_pos.clamp(min=self.eps))

        los_pos = torch.where(trigger, los_pos_nb, los_pos_bce)

        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss =  expanded_weights * los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = (self.gamma_pos + expanded_weights ) * y + (self.gamma_neg + expanded_weights) * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        loss=-loss.mean()

        return loss
