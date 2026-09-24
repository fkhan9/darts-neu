import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cell import Cell
from operations import PRIMITIVES
from genotypes import Genotype


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, self.multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(2 + i for i in range(self.steps))
        num_ops = 8
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def genotype(self):
        """
        Discretizes the current alpha weights into a fixed Genotype, following
        the standard DARTS parsing rule: for each intermediate node, keep the
        top-2 incoming edges (ranked by their best non-'none' operation weight),
        then select the argmax operation (excluding 'none') on each kept edge.
        """
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            none_index = PRIMITIVES.index('none')
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(n),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_index)
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != none_index:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat,
        )
        return genotype

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits