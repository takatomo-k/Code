"""Define RNN-based encoders."""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.init import orthogonal_, xavier_uniform_
from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules.memory import Memory

class MEMEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 nr_cells=16,read_heads=4,cell_size=32,
                 independent_linears=False,share_memory=True,clip=20
                ):
        super(MEMEncoder, self).__init__()
        assert embeddings is not None
        bidirectional=False #Not impliment
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size=hidden_size
        self.num_hidden_layer=1
        self.dropout=dropout
        self.embeddings = embeddings
        self.num_layers=num_layers
        self.rnn_type=rnn_type
        self.cell_size=cell_size
        self.clip=clip
        self.read_heads=read_heads
        nn_input_size = self.embeddings.embedding_size + (cell_size*read_heads)
        nn_output_size= hidden_size + (cell_size*read_heads)
        
        self.rnns=[]
        for layer in range(self.num_layers):
            if self.rnn_type.lower() == 'rnn':
                self.rnns.append(nn.RNN((nn_input_size if layer == 0 else nn_output_size), hidden_size, dropout=dropout, batch_first=True))
            elif self.rnn_type.lower() == 'gru':
                self.rnns.append(nn.GRU((nn_input_size if layer == 0 else nn_output_size),
                                        output_size, dropout=dropout, batch_first=True))
            if self.rnn_type.lower() == 'lstm':
                self.rnns.append(nn.LSTM((nn_input_size if layer == 0 else nn_output_size),
                                        hidden_size, dropout=dropout, batch_first=True))
            setattr(self, rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])
        
        self.memories = []
        for layer in range(self.num_layers):
            self.memories.append(
                Memory(
                input_size=hidden_size,
                mem_size=nr_cells,
                cell_size=cell_size,
                read_heads=read_heads,
                independent_linears=independent_linears
                ).cuda()
            )
            # only one memory shared by all layers
            if share_memory:
                break
            # memories for each layer
            setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])
         # final output layer
        self.output = nn.Linear(nn_output_size, embeddings.embedding_size)
        orthogonal_(self.output.weight)  
    
    def forward(self, src, lengths=None,hx=(None, None, None), reset_experience=False, pass_through_memory=True):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()
        """
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)
        is_packed = type(emb) is PackedSequence
        if is_packed:
            x, lengths = pad(emb)
            max_length = lengths[0]
        else:
        """
        max_length = emb.size(0)
        lengths    = [emb.size(0)] * max_length

        batch_size = emb.size(1)

        emb= emb.transpose(0, 1)
        controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)
        # concat xwith last read (or padding) vectors
        #import pdb; pdb.set_trace()
        inputs = [torch.cat([emb[:, x, :], last_read], 1) for x in range(max_length)]
        # batched forward pass per element / word / etc
        outs = [None] * max_length
        read_vectors = None

        # pass through time
        for time in range(max_length):
        # pass thorugh layers
            for layer in range(self.num_layers):
                # this layer's hidden states
                chx = controller_hidden[layer]
                m = mem_hidden if self.share_memory else mem_hidden[layer]
                # pass through controller
                outs[time], (chx, m, read_vectors) = \
                self._layer_forward(inputs[time], layer, (chx, m), pass_through_memory)
                # store the memory back (per layer or shared)
                if self.share_memory:
                    mem_hidden = m
                else:
                    mem_hidden[layer] = m
                    controller_hidden[layer] = chx

                if read_vectors is not None:
                # the controller output + read vectors go into next layer
                    outs[time] = torch.cat([outs[time], read_vectors], 1)
                else:
                    outs[time] = torch.cat([outs[time], last_read], 1)
                inputs[time] = outs[time]
        # pass through final output layer
        inputs = [self.output(i) for i in inputs]
        outputs = torch.stack(inputs)
        #if is_packed:
        #    outputs = pack(output, lengths)
        return (controller_hidden,mem_hidden,read_vectors),outputs,lengths

    def _init_hidden(self, hx, batch_size, reset_experience):
    # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # initialize hidden state of the controller RNN
        if chx is None:
            h = torch.zeros(self.num_hidden_layer, batch_size, self.hidden_size).cuda()
            xavier_uniform_(h)
            chx = [ (h, h) if self.rnn_type.lower() == 'lstm' else h for x in range(self.num_layers)]
        # Last read vectors
        if last_read is None:
            last_read = torch.zeros(batch_size, self.cell_size * self.read_heads).cuda()
        # memory states
        if mhx is None:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, mhx, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

        return chx, mhx, last_read


    def _layer_forward(self, x, layer, hx=(None, None), pass_through_memory=True):
        (chx, mhx) = hx
        #import pdb; pdb.set_trace()
        # pass through the controller layer
        x, chx = self.rnns[layer](x.unsqueeze(1), chx)
        #import pdb; pdb.set_trace()
        x = F.dropout(x.squeeze(1),self.dropout,self.training)

        # clip the controller output
        if self.clip != 0:
            output = torch.clamp(x, -self.clip, self.clip)
        else:
            output = x

        # the interface vector
        xi = output

        # pass through memory
        if pass_through_memory:
            if self.share_memory:
                read_vecs, mhx = self.memories[0](xi, mhx)
            else:
                read_vecs, mhx = self.memories[layer](xi, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, self.cell_size * self.read_heads)
        else:
            read_vectors = None

        return output, (chx, mhx, read_vectors)

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from .util import *
from .memory import *

from torch.nn.init import orthogonal_, xavier_uniform_


class DNC(nn.Module):

  def __init__(
      self,
      input_size,
      hidden_size,
      rnn_type='lstm',
      num_layers=1,
      num_hidden_layers=2,
      bias=True,
      batch_first=True,
      dropout=0,
      bidirectional=False,
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=0,
      independent_linears=False,
      share_memory=True,
      debug=False,
      clip=20
  ):
    super(DNC, self).__init__()
    # todo: separate weights and RNNs for the interface and output vectors

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.rnn_type = rnn_type
    self.num_layers = num_layers
    self.num_hidden_layers = num_hidden_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id
    self.independent_linears = independent_linears
    self.share_memory = share_memory
    self.debug = debug
    self.clip = clip

    self.cell_size = self.cell_size
    self.read_heads = self.read_heads

    self.read_vectors_size = self.read_heads * self.cell_size
    self.output_size = self.hidden_size

    self.nn_input_size = self.input_size + self.read_vectors_size
    self.nn_output_size = self.output_size + self.read_vectors_size

    self.rnns = []
    self.memories = []

    for layer in range(self.num_layers):
      if self.rnn_type.lower() == 'rnn':
        self.rnns.append(nn.RNN((self.nn_input_size if layer == 0 else self.nn_output_size), self.output_size,
                                bias=self.bias, nonlinearity=self.nonlinearity, batch_first=True, dropout=self.dropout, num_layers=self.num_hidden_layers))
      elif self.rnn_type.lower() == 'gru':
        self.rnns.append(nn.GRU((self.nn_input_size if layer == 0 else self.nn_output_size),
                                self.output_size, bias=self.bias, batch_first=True, dropout=self.dropout, num_layers=self.num_hidden_layers))
      if self.rnn_type.lower() == 'lstm':
        self.rnns.append(nn.LSTM((self.nn_input_size if layer == 0 else self.nn_output_size),
                                 self.output_size, bias=self.bias, batch_first=True, dropout=self.dropout, num_layers=self.num_hidden_layers))
      setattr(self, self.rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])

      # memories for each layer
      if not self.share_memory:
        self.memories.append(
            Memory(
                input_size=self.output_size,
                mem_size=self.nr_cells,
                cell_size=self.cell_size,
                read_heads=self.read_heads,
                gpu_id=self.gpu_id,
                independent_linears=self.independent_linears
            )
        )
        setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

    # only one memory shared by all layers
    if self.share_memory:
      self.memories.append(
          Memory(
              input_size=self.output_size,
              mem_size=self.nr_cells,
              cell_size=self.cell_size,
              read_heads=self.read_heads,
              gpu_id=self.gpu_id,
              independent_linears=self.independent_linears
          )
      )
      setattr(self, 'rnn_layer_memory_shared', self.memories[0])

    # final output layer
    
    self.output = nn.Linear(self.nn_output_size, self.input_size)
    orthogonal_(self.output.weight)
    
    if self.gpu_id != -1:
      [x.cuda(self.gpu_id) for x in self.rnns]
      [x.cuda(self.gpu_id) for x in self.memories]
      self.output.cuda()
    
  def _init_hidden(self, hx, batch_size, reset_experience,memories=None):
    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (chx, mhx, last_read) = hx

    # initialize hidden state of the controller RNN
    if chx is None:
      h = cuda(torch.zeros(self.num_hidden_layers, batch_size, self.output_size), gpu_id=self.gpu_id)
      xavier_uniform_(h)

      chx = [ (h, h) if self.rnn_type.lower() == 'lstm' else h for x in range(self.num_layers)]

    # Last read vectors
    if last_read is None:
      last_read = cuda(torch.zeros(batch_size, self.cell_size * self.read_heads), gpu_id=self.gpu_id)
    if memories is not None:
      self.memories=memories
    # memory states
    if mhx is None:
      if self.share_memory:
        mhx = self.memories[0].reset(batch_size, erase=reset_experience)
      else:
        mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
    else:
      if self.share_memory:
        mhx = self.memories[0].reset(batch_size, mhx, erase=reset_experience)
      else:
        mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

    return chx, mhx, last_read

  def _debug(self, mhx, debug_obj):
    if not debug_obj:
      debug_obj = {
          'memory': [],
          'link_matrix': [],
          'precedence': [],
          'read_weights': [],
          'write_weights': [],
          'usage_vector': [],
      }

    debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
    debug_obj['link_matrix'].append(mhx['link_matrix'][0][0].data.cpu().numpy())
    debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
    debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().numpy())
    debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu().numpy())
    debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())
    return debug_obj

  def _layer_forward(self, x, layer, hx=(None, None), pass_through_memory=True):
    (chx, mhx) = hx

    # pass through the controller layer
    x, chx = self.rnns[layer](input.unsqueeze(1), chx)
    x= x.squeeze(1)

    # clip the controller output
    if self.clip != 0:
      output = torch.clamp(x, -self.clip, self.clip)
    else:
      output = input

    # the interface vector
    xi = output

    # pass through memory
    if pass_through_memory:
      if self.share_memory:
        read_vecs, mhx = self.memories[0](xi, mhx)
      else:
        read_vecs, mhx = self.memories[layer](xi, mhx)
      # the read vectors
      read_vectors = read_vecs.view(-1, self.cell_size * self.read_heads)
    else:
      read_vectors = None

    return output, (chx, mhx, read_vectors)

  def forward(self, x, hx=(None, None, None), reset_experience=False, pass_through_memory=True,memories=None):
    # handle packed data
    is_packed = type(input) is PackedSequence
    if is_packed:
      x, lengths = pad(input)
      max_length = lengths[0]
    else:
      max_length = input.size(1) if self.batch_first else input.size(0)
      lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

    batch_size = input.size(0) if self.batch_first else input.size(1)

    #if not self.batch_first:
    #  x= input.transpose(0, 1)
    # make the data time-first

    controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience,memories)

    # concat xwith last read (or padding) vectors
    inputs = [torch.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

    # batched forward pass per element / word / etc
    if self.debug:
      viz = None

    outs = [None] * max_length
    read_vectors = None

    # pass through time
    for time in range(max_length):
      # pass thorugh layers
      for layer in range(self.num_layers):
        # this layer's hidden states
        chx = controller_hidden[layer]
        m = mem_hidden if self.share_memory else mem_hidden[layer]
        # pass through controller
        outs[time], (chx, m, read_vectors) = \
          self._layer_forward(inputs[time], layer, (chx, m), pass_through_memory)

        # debug memory
        if self.debug:
          viz = self._debug(m, viz)

        # store the memory back (per layer or shared)
        if self.share_memory:
          mem_hidden = m
        else:
          mem_hidden[layer] = m
        controller_hidden[layer] = chx

        if read_vectors is not None:
          # the controller output + read vectors go into next layer
          outs[time] = torch.cat([outs[time], read_vectors], 1)
        else:
          outs[time] = torch.cat([outs[time], last_read], 1)
        inputs[time] = outs[time]

    if self.debug:
      viz = {k: np.array(v) for k, v in viz.items()}
      viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}

    # pass through final output layer
    inputs = [self.output(i) for i in inputs]
    outputs = torch.stack(inputs, 1 if self.batch_first else 0)

    if is_packed:
      outputs = pack(output, lengths)

    if self.debug:
      return outputs, (controller_hidden, mem_hidden, read_vectors), viz, self.memories
    else:
      return outputs, (controller_hidden, mem_hidden, read_vectors), self.memories

  def decode(self, x, hx=(None, None, None), reset_experience=False, pass_through_memory=True):
      # handle packed data
      is_packed = type(input) is PackedSequence
      if is_packed:
        x, lengths = pad(input)
        max_length = lengths[0]
      else:
        max_length = input.size(1) if self.batch_first else input.size(0)
        lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

      batch_size = input.size(0) if self.batch_first else input.size(1)

      #if not self.batch_first:
      #  x= input.transpose(0, 1)
      # make the data time-first

      controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience,memories)

      # concat xwith last read (or padding) vectors
      inputs = [torch.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

      # batched forward pass per element / word / etc
      if self.debug:
        viz = None

      outs = [None] * (max_length-1)
      read_vectors = None
      inp=input[:,0]
      # pass through time
      for time in range(max_length):
        # pass thorugh layers
        for layer in range(self.num_layers):
          # this layer's hidden states
          chx = controller_hidden[layer]
          m = mem_hidden if self.share_memory else mem_hidden[layer]
          # pass through controller
          inp, (chx, m, read_vectors) = \
            self._layer_forward(inp, layer, (chx, m), pass_through_memory)
          outs[time]=inp
          # debug memory
          if self.debug:
            viz = self._debug(m, viz)

          # store the memory back (per layer or shared)
          if self.share_memory:
            mem_hidden = m
          else:
            mem_hidden[layer] = m
          controller_hidden[layer] = chx

          if read_vectors is not None:
            # the controller output + read vectors go into next layer
            outs[time] = torch.cat([outs[time], read_vectors], 1)
          else:
            outs[time] = torch.cat([outs[time], last_read], 1)
         

      if self.debug:
        viz = {k: np.array(v) for k, v in viz.items()}
        viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}

      # pass through final output layer
      outs = [self.output(i) for i in outs]
      outputs = torch.stack(outs, 1 if self.batch_first else 0)

      if is_packed:
        outputs = pack(output, lengths)

      if self.debug:
        return outputs, (controller_hidden, mem_hidden, read_vectors), viz, self.memories
      else:
        return outputs, (controller_hidden, mem_hidden, read_vectors), self.memories
"""