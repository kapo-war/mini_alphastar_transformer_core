#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Core."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP

__author__ = "Ruo-Ze Liu"

debug = False

class Coreblock(nn.Module):
    def __init__(self, dmodel, dimff, dropout):
        super(Coreblock, self).__init__()
        self.transformerblock = nn.TransformerEncoderLayer(d_model=dmodel, dim_feedforward = dimff, 
                                                           dropout=dropout, batch_first=True, nhead=8)
    def forward(self, input):
        return self.transformerblock(input)
    
class Core(nn.Module):
    '''
    Inputs: embedded_entity, embedded_spatial, embedded_scalar
    Outputs: transformer_out - The output of the transformer
    '''

    def __init__(self, embedding_dim=AHP.original_1024, batch_size=AHP.batch_size,
                 sequence_length=AHP.sequence_length,
                 n_layers=AHP.core_trans_layers, drop_prob=AHP.core_drop_prob, dim_ff=AHP.core_dimff):
        super(Core, self).__init__()
        self.n_layers = n_layers

        self.transformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer_layers.append(Coreblock(embedding_dim, dim_ff, drop_prob))

        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial, 
                batch_size=None, sequence_length=None):
        # note: the input_shape[0] is batch_seq_size, we only transfrom it to [batch_size, seq_size, ...]
        # before input it into the transformer
        # shapes of embedded_entity, embedded_spatial, embedded_scalar are all [batch_seq_size x embedded_size]
        batch_seq_size = embedded_scalar.shape[0]

        batch_size = batch_size if batch_size is not None else self.batch_size
        sequence_length = sequence_length if sequence_length is not None else self.sequence_length
        input_tensor = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=-1)
        del embedded_scalar, embedded_entity, embedded_spatial

        # note, before input to the transformer
        # we transform the shape from [batch_seq_size, embedding_size] 
        # to the actual [batch_size, seq_size, embedding_size] 
        embedding_size = input_tensor.shape[-1]
        transformer_out = input_tensor.reshape(batch_size, sequence_length, embedding_size)

        for i in range(self.n_layers):
            transformer_out = self.transformer_layers[i](transformer_out)
        transformer_out = transformer_out.reshape(batch_size * sequence_length, embedding_size)
        del input_tensor

        return transformer_out

def test():

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
