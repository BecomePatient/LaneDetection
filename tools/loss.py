import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import time
import numpy as np

def compute_binary_loss(binary_label,logits,weights=None):

    # binary_values, binary_indices = torch.unique(binary_label, return_inverse=True)
    # binary_indices = binary_indices.view(binary_label.shape)
    # binary_label_onehot = F.one_hot(binary_indices, num_classes=num_classes)
    # binary_label_onehot = binary_label_onehot.permute(0,3,1,2) 

    loss = F.cross_entropy(input=logits, target=binary_label.float(), weight=weights)   
    return loss

def compute_border_loss(binary_label,logits,weights=None):

    h, w = binary_label.size(2), binary_label.size(3)
    ph, pw = logits.size(2), logits.size(3)
    if ph != h or pw != w:
        logits = F.interpolate(logits, size=(
            h, w), mode='bilinear', align_corners=True)
    # print(f"binary_label ={binary_label.shape},logits = {logits.shape}")
    loss = F.cross_entropy(input=logits, target=binary_label.float(), weight=weights)   
    return loss

class EndLossCalculator:
    def __init__(self, device):
        self.max_n_clusters = 4
        self.total_classes = torch.tensor([0.0000, 0.1176, 0.4902]).to(device) 
    def compute_loss(self, instance_label, pixelcls,weights):
        h, w = instance_label.shape[1], instance_label.shape[2]
        ph, pw = pixelcls.shape[1], pixelcls.shape[2]
        
        if ph != h or pw != w:
            try: 
                pixelcls = F.interpolate(pixelcls.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True)
            except Exception as e:
                print(f"Error in resizing pixelcls: {e}")
        pixelcls = pixelcls.squeeze(1)
        target_flat = instance_label.flatten().unsqueeze(1)  # (N, 1)
        total_classes_expanded = self.total_classes.unsqueeze(0)  # (1, C)
        
        diff = torch.abs(target_flat - total_classes_expanded)
        min_diff, mapped_indices = torch.min(diff, dim=1)
        mapped_indices = mapped_indices.view(instance_label.shape).float()
        squared_diff = (pixelcls - mapped_indices) ** 2

        weight_matrix = torch.where(mapped_indices == 0, weights[0], weights[1])
        squared_diff *= weight_matrix
        loss = squared_diff.mean()
        return loss
    def compute_loss_V2(self, instance_label, pixelcls,weights):
        
        WeiT = []
        WeiT.append(weights[0])
        for _ in range(1,self.max_n_clusters):
            WeiT.append(weights[1])
        weights = torch.tensor(WeiT, dtype=torch.float32).to(weights.device)

        # print(f"weights = {weights}")
        h, w = instance_label.shape[1], instance_label.shape[2]
        ph, pw = pixelcls.shape[2], pixelcls.shape[3]
        if ph != h or pw != w:
            try: 
                pixelcls_expand = F.interpolate(pixelcls, size=(h, w), mode='bilinear', align_corners=True)
            except Exception as e:
                print(f"Error in resizing pixelcls: {e}")

        target_flat = instance_label.flatten().unsqueeze(1)  # (N, 1)
        total_classes_expanded = self.total_classes.unsqueeze(0)  # (1, C)
        diff = torch.abs(target_flat - total_classes_expanded)
        min_diff, mapped_indices = torch.min(diff, dim=1)
        mapped_indices = mapped_indices.view(instance_label.shape)

        # unique_values = np.unique(mapped_indices.cpu())
        # print(f"unique_values = {unique_values}")

        target = F.one_hot(mapped_indices, num_classes=self.max_n_clusters).permute(0, 3, 1, 2)
        loss = F.cross_entropy(input=pixelcls_expand, target=target.float(), weight=weights)        
        return loss

class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=3.0,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True, reduction='mean'):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        self.max_n_clusters = 5
        self.total_classes = torch.tensor([0.0000, 0.1176, 0.4902]).to(usegpu)
        self.tolerance = 1e-3
        self.device = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, instance_label,n_clusters):
        return self._discriminative_loss(input, instance_label, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        start_time = time.time()
        bs, n_features, height, width = input.size()

        target_flat = target.flatten().unsqueeze(1)  # (N, 1)
        total_classes_expanded = self.total_classes.unsqueeze(0)  # (1, C)
        diff = torch.abs(target_flat - total_classes_expanded)
        min_diff, mapped_indices = torch.min(diff, dim=1)
        mapped_indices = mapped_indices.view(target.shape)
        target = F.one_hot(mapped_indices, num_classes=self.max_n_clusters).to(target.device)
        target = target.permute(0,3,1,2) 

        max_n_clusters = target.size(1)
        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)
        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)
        # print(f"l_var = {l_var},l_dist = {l_dist},l_reg = {l_reg}")
        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # print(f"target_sample = {target.shape}")
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :,n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / target_sample.sum(2)
            # print(f"mean_sample = {mean_sample}")
            # print(f"input_sample.sum(2) = {input_sample.sum(2)}")
            # print(f"target_sample.sum(2) = {target_sample.sum(2)}")
            # padding
            n_pad_clusters = max_n_clusters - len(n_clusters[i])
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters)
                pad_sample = Variable(pad_sample)
                if self.usegpu != "cpu":
                    pad_sample = pad_sample.to(self.device)
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)
        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target
        var_term = 0
        for i in range(bs):
            # n_clusters, n_loc
            var_sample = var[i, n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, n_clusters[i]]
            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            var_term += c_var.sum() /len(n_clusters[i])
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if len(n_clusters[i]) <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, len(n_clusters[i]), len(n_clusters[i]))
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(len(n_clusters[i])))
            margin = Variable(margin)
            if self.usegpu != "cpu":
                margin = margin.to(self.device)
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * len(n_clusters[i]) * (len(n_clusters[i]) - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term

    
class MeanIoU:
    def __init__(self, num_classes,device):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64).to(device)

    def reset(self):
        self.confusion_matrix.zero_()  # Use in-place zeroing to avoid creating new tensors

    def compute_iou(self, pred, target):
        self.reset()
        pred = pred.view(-1)
        target = torch.argmax(target, dim=1).view(-1)
        
        # Create a mask to ignore a specific class (assumed to be background or ignored)
        mask = (target != 0)

        pred = pred[mask]
        target = target[mask]

        with torch.no_grad():
            indices = self.num_classes * target + pred 
            confusion_matrix_update = torch.bincount(indices, minlength=self.num_classes**2)
            self.confusion_matrix += confusion_matrix_update.view(self.num_classes, self.num_classes)

        print(self.confusion_matrix)

        # Calculate Intersection over Union (IoU)
        intersection = self.confusion_matrix.diag()
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection
        iou = intersection.float() / union.float()

        # Ignore NaN values in mean calculation (e.g., division by zero)
        mean_iou = iou[iou == iou].mean().item()

        return mean_iou


if __name__ == "__main__":
    L = EndLossCalculator(device="cpu")
    label = torch.ones((8,640,384))
    pred = torch.zeros((8,80,48))
    loss = L.compute_loss(label,pred)
    
    print(f"loss = {loss}")