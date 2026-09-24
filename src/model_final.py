"""
Fixed-architecture model, built by discretizing a Genotype (see
model_search.py's genotype() method) into an actual, non-mixed network.

Differences from the search-phase model_search.py, deliberately:
  - No MixedOp / no alpha weighting -- each edge uses exactly the operation
    the genotype selected, nothing else.
  - affine=True on all ops (search phase used affine=False throughout, the
    standard DARTS convention for search-time BatchNorm stability -- final
    training does not need that constraint).
  - Typically trained with more channels (C) than the search phase, since
    memory is no longer split across 8 candidate operations per edge --
    only the compute cost of a single ordinary CNN remains.
"""

import torch
import torch.nn as nn
from operations import OPS, FactorizedReduce, ReLUConvBN


class FixedCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)

        gene = genotype.reduce if reduction else genotype.normal
        concat = genotype.reduce_concat if reduction else genotype.normal_concat

        self._compile(C, gene, concat, reduction)

    def _compile(self, C, gene, concat, reduction):
        assert len(gene) % 2 == 0
        self._steps = len(gene) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._indices = []
        for op_name, index in gene:
            stride = 2 if reduction and index < 2 else 1
            op = OPS[op_name](C, stride, True)
            self._ops.append(op)
            self._indices.append(index)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            s = op1(h1) + op2(h2)
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkFinal(nn.Module):
    def __init__(self, C, num_classes, layers, genotype, stem_multiplier=3):
        super().__init__()
        self.layers = layers

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
            cell = FixedCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits