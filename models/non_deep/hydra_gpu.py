# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://doi.org/10.1007/s10618-023-00939-3

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class HydraGPU(nn.Module):

    def __init__(self, input_length, k = 8, g = 64, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        W = W - W.mean(-1, keepdims = True)
        W = W / W.abs().sum(-1, keepdims = True)

        self.register_buffer("W", W)
        
        # self.num_features_ = self.num_dilations * self.divisor * self.k * self.h * 2
        self.num_features = self.num_dilations * self.divisor * self.k * self.h * 2

    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X if diff_index == 0 else diff_X, self.W[dilation_index, diff_index], dilation = d, padding = p) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device = X.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device = X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        # return Z
        return Z.clamp(0).sqrt()

class HydraMultivariateGPU(nn.Module):

    def __init__(self, input_length, num_channels, k = 8, g = 64, max_num_channels = 8, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        W = W - W.mean(-1, keepdims = True)
        W = W / W.abs().sum(-1, keepdims = True)

        self.register_buffer("W", W)

        # self.num_features_ = self.num_dilations * self.divisor * self.k * self.h * 2
        self.num_features = self.num_dilations * self.divisor * self.k * self.h * 2

        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        self.I = torch.randint(0, num_channels, (self.num_dilations, self.divisor, self.h, num_channels_per))

    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X[:, self.I[dilation_index, diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index, diff_index]].sum(2),
                              self.W[dilation_index, diff_index], dilation = d, padding = p,
                              groups = self.h) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k, device = X.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k, device = X.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z.clamp(0).sqrt()