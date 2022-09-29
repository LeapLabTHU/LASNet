import math
import torch.nn as nn
import torch


class SparsityCriterion_gradually(nn.Module):
    def __init__(self, sparsity_target, num_epochs, full_flops):
        super(SparsityCriterion_gradually, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops

    def forward(self, epoch, sparsity_list, sparsity_list_dil, flops_perc_list, flops, gradually=False, is_eval=False):
        p = epoch / (self.num_epochs/3)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (0.9 - progress*(0.9 - self.sparsity_target))
        lower_bound = progress*(self.sparsity_target-0.15)+0.15
        
        flops_target = self.sparsity_target
        loss_block_bounds = 0
        for i in range(len(sparsity_list)):
            loss_block_bounds += max(0, sparsity_list_dil[i] - upper_bound)**2
            loss_block_bounds += max(0, lower_bound - sparsity_list[i])**2
            
            loss_block_bounds += max(0, sparsity_list_dil[i] - sparsity_list[i] - 0.2)**2
        loss_block_bounds /= len(sparsity_list)
        
        loss_sparsity = (flops/self.full_flops - flops_target)**2
        
        return  loss_block_bounds + loss_sparsity


class SparsityCriterion_bounds(nn.Module):
    def __init__(self, sparsity_target, num_epochs, full_flops):
        super(SparsityCriterion_bounds, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops

    def forward(self, epoch, sparsity_list, flops):
        
        loss_block_bounds = 0.0
        p = epoch / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (1 - progress*(1-self.sparsity_target))
        lower_bound = progress*(self.sparsity_target)
            
        for i in range(len(sparsity_list)):
            loss_block_bounds += max(0, sparsity_list[i] - upper_bound)**2
            
            loss_block_bounds += max(0, lower_bound - sparsity_list[i])**2
        
        loss_block_bounds /= len(sparsity_list)
        loss_sparsity = (flops/self.full_flops - self.sparsity_target)**2

        return  loss_block_bounds + loss_sparsity


class SparsityCriterion_block(nn.Module):
    def __init__(self, sparsity_target, num_epochs, full_flops, margin=0.2, add_all_loss=False):
        super(SparsityCriterion_block, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops
    def forward(self, epoch, sparsity_list, flops):
        loss_block_bounds = torch.mean((sparsity_list - self.sparsity_target)**2)
        return loss_block_bounds


class SparsityCriterion_per_block(nn.Module):
    def __init__(self, sparsity_target, target_per_block, num_epochs, full_flops):
        super(SparsityCriterion_per_block, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.full_flops = full_flops
        self.target_per_block = target_per_block
    def forward(self, epoch, sparsity_list, flops):
        loss_block_bounds = torch.mean((sparsity_list - self.target_per_block)**2)
        return loss_block_bounds
