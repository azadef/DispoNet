#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from simsg.layers import build_mlp
import numpy as np
import torch.nn.functional as fn


"""
PyTorch modules for dealing with graphs.
"""

def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim_obj
    self.input_dim_obj = input_dim_obj
    self.input_dim_pred = input_dim_pred
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling

    self.pooling = pooling
    net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)


  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
    - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
    Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (num_triples,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()
    
    # Get current vectors for subjects and objects; these have shape (num_triples, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]
    
    # Get current vectors for triples; shape is (num_triples, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
    # p vecs have shape (num_triples, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]
 
    # Allocate space for pooled object vectors of shape (num_objs, H)
    pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (num_triples, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    # print(pooled_obj_vecs.shape, o_idx_exp.shape)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      #print("here i am, would you send me an angel")
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
      ones = torch.ones(num_triples, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)
  
      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (num_objs, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim_obj, input_dim_pred, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim_obj': input_dim_obj,
      'input_dim_pred': input_dim_pred,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs



class DisenTripletGCN(nn.Module):
    def __init__(self, args):
      super(DisenTripletGCN, self).__init__()
      self.Din_obj = args.input_dim_obj
      self.Din_pred = args.input_dim_pred
      self.H = args.hidden_dim
      self.Dout = args.out_dim

      self.net1 = DisenGCN(2 * args.input_dim_obj + args.input_dim_pred, 2 * args.hidden_dim + args.out_dim, args, split_mlp=False)
      self.net2 = DisenGCN(args.hidden_dim, args.out_dim, args, split_mlp=False)

    def extract_info(self, obj_vecs, pred_vecs, edges):
      self.dtype, self.device = obj_vecs.dtype, obj_vecs.device
      self.O, self.T = obj_vecs.size(0), pred_vecs.size(0)
      # Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

      # Break apart indices for subjects and objects; these have shape (T,)
      self.s_idx = edges[:, 0].contiguous()
      self.o_idx = edges[:, 1].contiguous()

      # Get current vectors for subjects and objects; these have shape (T, Din)
      cur_s_vecs = obj_vecs[self.s_idx]
      cur_o_vecs = obj_vecs[self.o_idx]

      # Get current vectors for triples; shape is (T, 3 * Din)
      # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
      cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
      return cur_t_vecs

    def pool_objs(self, out_feats):
      # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
      # p vecs have shape (T, Dout)
      dtype, device = self.dtype, self.device
      new_s_vecs = out_feats[:, :self.H]
      pred_vecs = out_feats[:, self.H:(self.H + self.Dout)]
      new_o_vecs = out_feats[:, (self.H + self.Dout):(2 * self.H + self.Dout)]

      # Allocate space for pooled object vectors of shape (O, H)
      pooled_obj_vecs = torch.zeros(self.O, self.H, dtype=dtype, device=device)

      # Use scatter_add to sum vectors for objects that appear in multiple triples;
      # we first need to expand the indices to have shape (T, D)
      s_idx_exp = self.s_idx.view(-1, 1).expand_as(new_s_vecs)
      o_idx_exp = self.o_idx.view(-1, 1).expand_as(new_o_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
      pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(self.O, dtype=dtype, device=device)
      ones = torch.ones(self.T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, self.s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, self.o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

      return pooled_obj_vecs, pred_vecs

    def forward(self, obj_vecs, pred_vecs, edges):
      curr_t_vecs = self.extract_info(obj_vecs, pred_vecs, edges)
      out_feats = self.net1(curr_t_vecs, edges)

      so_vecs, pred_vecs = self.pool_objs(out_feats)
      obj_vecs = self.net2(so_vecs, edges)

      return obj_vecs, pred_vecs


class NeibRoutLayer(nn.Module):
  def __init__(self, num_caps, niter, tau=1.0):
    super(NeibRoutLayer, self).__init__()
    self.k = num_caps
    self.niter = niter
    self.tau = tau

  #
  # x \in R^{n \times d}: d-dimensional node representations.
  #    It can be node features, output of the previous layer,
  #    or even rows of the adjacency matrix.
  #
  # src_trg \in R^{2 \times m}: a list that contains m edges.
  #    src means the source nodes of the edges, and
  #    trg means the target nodes of the edges.
  #
  def forward(self, x, src_trg):
    m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
    n, d = x.shape
    k, delta_d = self.k, d // self.k
    x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
    z = x[src].view(m, k, delta_d)
    u = x
    scatter_idx = trg.view(m, 1).expand(m, d)
    for clus_iter in range(self.niter):
      p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
      p = fn.softmax(p / self.tau, dim=1)
      scatter_src = (z * p.view(m, k, 1)).view(m, d)
      u = torch.zeros(n, d, device=x.device)
      u.scatter_add_(0, scatter_idx, scatter_src)
      u += x
      # noinspection PyArgumentList
      u = fn.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
    # p: m * k, m is #edges
    return u  # , p


class DisenGCN(nn.Module):
  #
  # nfeat: dimension of a node's input feature
  # nclass: the number of target classes
  # hyperpm: the hyper-parameter configuration
  #    ncaps: the number of capsules/channels/factors per layer
  #    routit: routing iterations
  #
  def __init__(self, nfeat, nclass, hyperpm, split_mlp=False):
    super(DisenGCN, self).__init__()
    self.input_dim = nfeat
    self.output_dim = nclass
    self.pca = SparseInputLinear(nfeat, hyperpm.ncaps * hyperpm.nhidden)
    self.bn = nn.BatchNorm1d(nclass)
    conv_ls = []
    for i in range(hyperpm.nlayer):
      conv = NeibRoutLayer(hyperpm.ncaps, hyperpm.routit)
      self.add_module('conv_%d' % i, conv)
      conv_ls.append(conv)
    self.conv_ls = conv_ls
    if split_mlp:
      self.clf = SplitMLP(nclass, hyperpm.nhidden * hyperpm.ncaps,
                          nclass)
    else:
      self.clf = nn.Linear(hyperpm.nhidden * hyperpm.ncaps, nclass)
    self.dropout = hyperpm.dropout

  def _dropout(self, x):
    return fn.dropout(x, self.dropout, training=self.training)

  def forward(self, obj_vec, edges):
    obj_vec = self.pca(obj_vec)
    obj_vec = self._dropout(fn.leaky_relu(obj_vec))
    for conv in self.conv_ls:
      obj_vec = conv(obj_vec, edges)
      # obj_vec = self._dropout(fn.leaky_relu(obj_vec))
    obj_vec = self.clf(obj_vec)
    obj_vec = self.bn(obj_vec)
    obj_vec = fn.leaky_relu(obj_vec)
    return obj_vec


# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
  def __init__(self, inp_dim, out_dim):
    super(SparseInputLinear, self).__init__()
    weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
    weight = nn.Parameter(torch.from_numpy(weight))
    bias = np.zeros(out_dim, dtype=np.float32)
    bias = nn.Parameter(torch.from_numpy(bias))
    self.inp_dim, self.out_dim = inp_dim, out_dim
    self.weight, self.bias = weight, bias
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / np.sqrt(self.weight.shape[1])
    self.weight.data.uniform_(-stdv, stdv)
    self.bias.data.uniform_(-stdv, stdv)

  def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
    return torch.mm(x, self.weight) + self.bias


class SplitMLP(nn.Module):
  def __init__(self, n_mlp, d_inp, d_out):
    super(SplitMLP, self).__init__()
    assert d_inp >= n_mlp and d_inp % n_mlp == 0
    assert d_out >= n_mlp and d_out % n_mlp == 0
    self.mlps = nn.Conv1d(in_channels=n_mlp, out_channels=d_out,
                          kernel_size=d_inp // n_mlp, groups=n_mlp)
    self.n_mlp = n_mlp

  def forward(self, x):
    n = x.shape[0]
    x = x.view(n, self.n_mlp, -1)
    x = self.mlps(x)
    x = x.view(n, -1)
    return x
