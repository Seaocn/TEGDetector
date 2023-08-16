import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from set2set import Set2Set


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0,batch_size = 10, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
            self.dropout_layer1 = nn.Dropout(p=0.5)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def preprocess_support(self, adj):
        In =  torch.ones((adj.size(1),))
        adj_normalized = self.nomalize_adj(torch.add(adj, torch.diag(In).cuda()))
        return adj_normalized

    def nomalize_adj(self, adj):
        D = torch.sum(adj[1], 1)
        D1 = torch.diag(torch.pow(D, D.new_full((adj.size(1),),-0.5))).cuda()
        return torch.matmul(torch.matmul(D1, adj), D1)

    def forward_na(self, x, adj):
        adj = self.preprocess_support(adj)
        # x = torch.cat([torch.t(x[i]).unsqueeze(0) for i in range(batch_size)])
        if self.dropout > 0.001:
            x = self.dropout_layer1(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

    def forward(self, x, adj):
        # adj = self.preprocess_support(adj)
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y

    def concat_forward(self, x, adj):
        # adj = self.preprocess_support(adj)
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = y.tolist()
        y = torch.Tensor([torch.t(torch.tensor(y[i])).tolist() for i in range(self.batch_size)])


        y = torch.matmul(y.cuda(), self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            print(y[0][0])
        y = y.squeeze()
        return y



class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.dropout = dropout

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.score_layer_1 = GraphConv(input_dim=hidden_dim, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.score_layer_2 = GraphConv(input_dim=hidden_dim, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.score_layer_3 = GraphConv(input_dim=hidden_dim, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=False, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def read_forward(self, x, adj, conv_first, conv_block, conv_last, batch_size):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first.forward_na(x, adj)
        x = self.act(x)
        x1 = x
        if self.bn:
            x = self.apply_bn(x)

        for i in range(len(conv_block)):
            x = conv_block[i].forward_na(x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)

        x = conv_last.forward_na(x, adj)

        x_tensor = x

        return x_tensor

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        if self.concat:
            x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            if self.concat:
                x_all.append(x)
        x = conv_last(x, adj)
        if self.concat:
            x_all.append(x)
            # x_tensor: [batch_size x num_nodes x embedding]
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
        # print(x_tensor.shape)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())




class Gcn(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.8, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], gcn_concat=True, pool_concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(Gcn, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=gcn_concat,
                                                    bn=bn, args=args)
        add_self = not gcn_concat
        self.read = False
        self.args = args
        self.ratio = 0.15
        self.hidden = 64
        self.batch_num = args.batch_size
        self.max_num_nodes = max_num_nodes
        self.assign_ratio = assign_ratio
        self.num_pooling = num_pooling
        self.pool_concat = pool_concat
        self.linkpred = linkpred
        self.assign_ent = True
        self.adj = nn.Parameter(torch.FloatTensor(args.batch_size, max_num_nodes, max_num_nodes).cuda())

        self.concat_gnn_1 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.concat_gnn_2 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.concat_gnn_3 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)


        self.lin1 = torch.nn.Linear(hidden_dim*2 if not self.pool_concat else (hidden_dim*2+embedding_dim)*2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.lin3 = torch.nn.Linear(hidden_dim//2, label_dim)


        # self.pred_model = self.build_pred_layers(
        #     (hidden_dim*2+embedding_dim) if pool_concat else embedding_dim, pred_hidden_dims,
        #     label_dim, num_aggs=self.num_aggs)  #三层
        self.pred_model = self.build_pred_layers(
            (hidden_dim*2+embedding_dim)*2 if pool_concat else embedding_dim, pred_hidden_dims,
            label_dim, num_aggs=self.num_aggs)  #三层

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)



    def get_mask(self, score, mask, batch_size, batch_num_nodes):


        score = [score[i].split([batch_num_nodes[i], self.max_num_nodes - batch_num_nodes[i]], dim=0)[0] for i in
                       range(self.batch_num )]
        pad1 = [nn.ZeroPad2d(padding=(0, self.max_num_nodes - int(batch_num_nodes[i]), 0, 0)) for i in range(batch_size)]
        score = [pad1[i](score[i].unsqueeze(0)) for i in range(batch_size)]
        score = torch.cat([score[i] for i in range(batch_size)])
        score =score+(-10000)*(1-mask)
        score_ori = score
        score = score.sort(dim=1, descending=True)[0]

        k = (self.assign_ratio * batch_num_nodes.to(torch.float)).ceil().to(torch.long)

        num = [score[i][int(k[i])].unsqueeze(0).unsqueeze(1) for i in range(batch_size)]


        num_ = torch.cat([num[i] for i in range(self.batch_num )])

        score_ori = score_ori - num_
        m_x_ = F.relu(score_ori).unsqueeze(2)
        score_mask =torch.sign( F.relu(score_ori))
        m_x = [torch.t(score_mask[i].repeat(self.args.hidden_dim,1)) for i in range(batch_size)]
        m_x = torch.cat([m_x[i].unsqueeze(0) for i in range(batch_size)])

        m_adj = [torch.matmul(torch.t(score_mask[i].unsqueeze(0)), score_mask[i].unsqueeze(0)) for i in range(batch_size)]
        m_adj = torch.cat([m_adj[i].unsqueeze(0) for i in range(batch_size)])

        # m_x_ = torch.cat([score_mask[i].unsqueeze(0) for i in range(batch_size)])

        return m_x, m_x_, m_adj, score_mask



    def Pool(self,x, adj, pool_layer, batch_num_nodes, m, batch_size):
        score = pool_layer(x,adj).squeeze()
        batch_num_nodes = torch.tensor(batch_num_nodes)

        m_x, m_x_, m_adj, mask_next = self.get_mask(score, m, batch_size, batch_num_nodes)

        x = torch.mul(x,m_x)
        x = torch.mul(x,torch.tanh(m_x_))
        adj = torch.mul(adj,m_adj)



        return x, adj, mask_next

    def get_fea_A(self, x, batch_size, ratio):

        x_sim = torch.matmul(torch.cat([torch.t(x[i]).unsqueeze(0) for i in range(batch_size)]),
                             torch.cat([F.normalize(x[i]).unsqueeze(0) for i in range(batch_size)]))
        x_sim = torch.cat([((torch.t(x_sim[i]) + x_sim[i]) / 2).unsqueeze(0) for i in range(batch_size)])

        node_num = int(x_sim.size(2) * x_sim.size(2) * ratio)
        x_sim_sort = [x_sim[i].view(-1).sort(dim=0, descending=True)[0] for i in range(batch_size)]
        num = [x_sim_sort[i][node_num] for i in range(batch_size)]
        fea_A = torch.cat([torch.sign(F.relu(x_sim[i] - num[i])).unsqueeze(0) for i in range(batch_size)])
        return fea_A

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        # self.adj.data = adj
        # self.A = adj

        # mask
        max_num_nodes = self.adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.out_all = []
        self.read_out = []

        # if x is not None:
        x = self.conv_first(x, adj)
        x = self.act(x)
        self.x0 = x
        # self.out, _ = torch.max(x, dim=1)
        self.batch_num_nodes = torch.tensor(batch_num_nodes)
        self.index = [torch.arange(1, self.batch_num_nodes[i], dtype=torch.long, device=x.device).unsqueeze(0) for i in range(self.batch_num )]
        self.index1 = [self.index[i].cpu().numpy() for i in range(self.batch_num )]
        self.pad = [nn.ZeroPad2d(padding=(0,self.max_num_nodes - int(batch_num_nodes[i]),0,0)) for i in range(self.batch_num )]
        self.index =[self.pad[i](self.index[i]) for i in range(self.batch_num )]
        self.index = torch.cat([self.index[i] for i in range(self.batch_num )], dim=0)
        self.mask_0 = torch.zeros(self.batch_num , self.max_num_nodes).cuda().scatter_(1, self.index, 1)


        self.x_pool1, self.adj_pool1, self.mask_1 = self.Pool(self.x0, adj, self.score_layer_1, batch_num_nodes, self.mask_0, self.batch_num )

        if self.pool_concat:

            self.out0 = torch.cat([torch.mean(self.x_pool1, dim=1), torch.max(self.x_pool1, dim=1)[0]], dim=1)
            self.out_all.append(self.out0)
        else:
            self.X1 = torch.cat([torch.mean(self.x_pool1,dim=1), torch.max(self.x_pool1, dim=1)[0]], dim=1)


        if self.bn:
            x = self.apply_bn(x)

        #
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](self.x_pool1.cuda(), self.adj_pool1.cuda())
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            #
            self.x1 = x
            out, _ = torch.max(x, dim=1)
            #
        batch_num_nodes_1 = torch.tensor(batch_num_nodes)
        self.batch_num_nodes_1 = (self.assign_ratio * batch_num_nodes_1.to(torch.float)).ceil().to(torch.long)
        self.x_pool2, self.adj_pool2, self.mask_2 = self.Pool(self.x1, self.adj_pool1, self.score_layer_2, self.batch_num_nodes_1 ,self.mask_1, self.batch_num )




        if self.pool_concat:

            # self.out1 = self.concat_gnn_2.concat_forward(self.x_pool2.cuda(), self.adj_pool2.cuda())
            self.out1 = torch.cat([torch.mean(self.x_pool2, dim=1), torch.max(self.x_pool2, dim=1)[0]], dim=1)
            self.out_all.append(self.out1)
        else:
            self.X2 = torch.cat([torch.mean(self.x_pool2, dim=1), torch.max(self.x_pool2, dim=1)[0]], dim=1)



        x = self.conv_last(self.x_pool2.cuda(), self.adj_pool2.cuda())
        self.x2 = x
        self.batch_num_nodes_2 = (self.assign_ratio * self.batch_num_nodes_1.to(torch.float)).ceil().to(torch.long)
        self.x_pool3, self.adj_pool3, self.mask_3 = self.Pool(self.x2, self.adj_pool2, self.score_layer_3,
                                                              self.batch_num_nodes_2, self.mask_2, self.batch_num )


        if self.pool_concat:

            # self.out2 = self.concat_gnn_3.concat_forward(self.x_pool3.cuda(), self.adj_pool3.cuda())
            self.out2 = torch.cat([torch.mean(self.x_pool3, dim=1), torch.max(self.x_pool3, dim=1)[0]], dim=1)
            self.out_all.append(self.out2)
        else:
            self.X3 = torch.cat([torch.mean(self.x_pool3, dim=1), torch.max(self.x_pool3, dim=1)[0]], dim=1)

        if self.pool_concat:
            if self.read:
                # self.X = torch.cat(self.read_out, dim=1)
                self.X = self.read_out3
            else:
                self.X = torch.cat(self.out_all, dim=1)
            ypred = self.pred_model(self.X)
        else:

            self.X = self.X1.cuda()+self.X2.cuda()+self.X3.cuda()
            #
            self.x = F.relu(self.lin1(self.X.cuda()))
            self.x = F.dropout(self.x.cuda(), p=self.dropout, training=self.training)
            self.x = F.relu(self.lin2(self.x.cuda()))
            self.x = F.log_softmax(self.lin3(self.x.cuda()), dim=-1)
            ypred = self.x

        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(Gcn, self).loss(pred, label)

        return loss


    def loss_node(self, pred, label, type='softmax'):
        # softmax + CE
        # if type == 'softmax':
        #     return -1 * F.cross_entropy(pred.long(), label.long(), size_average=True)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -torch.sum(log_prob.cuda() * label.cuda()) / self.batch_num
        return loss

    def update_adj(self):
        return self.adj




class SAGEPool_EGC(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.8, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], gcn_concat=True, pool_concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SAGEPool_EGC, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=gcn_concat,
                                                    bn=bn, args=args)
        add_self = not gcn_concat
        self.read = False
        self.args = args
        self.ratio = 0.15
        self.hidden = 64
        self.batch_num = args.batch_size
        self.max_num_nodes = max_num_nodes
        self.assign_ratio = assign_ratio
        self.num_pooling = num_pooling
        self.pool_concat = pool_concat
        self.linkpred = linkpred
        self.assign_ent = True
        self.adj = nn.Parameter(torch.FloatTensor(args.batch_size, max_num_nodes, max_num_nodes).cuda())

        self.concat_gnn_1 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.concat_gnn_2 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)
        self.concat_gnn_3 = GraphConv(input_dim=self.max_num_nodes, output_dim=1, add_self=False,
                               normalize_embedding=False, bias=True)


        self.lin1 = torch.nn.Linear(hidden_dim*2 if not self.pool_concat else (hidden_dim*2+embedding_dim)*2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.lin3 = torch.nn.Linear(hidden_dim//2, label_dim)

        assign_dim = int(max_num_nodes * assign_ratio)
        #read_GC
        self.read_first, self.read_block, self.read_last = self.build_conv_layers(
            self.args.max_nodes, self.hidden, 1, num_layers,
            add_self, normalize=False, dropout=dropout)
        self.read_first_after_pool, self.read_block_after_pool, self.read_last_after_pool = self.build_conv_layers(
            self.args.max_nodes, self.hidden, 1, num_layers,
            add_self, normalize=False, dropout=dropout)


        # self.pred_model = self.build_pred_layers(
        #     (hidden_dim*2+embedding_dim) if pool_concat else embedding_dim, pred_hidden_dims,
        #     label_dim, num_aggs=self.num_aggs)  #三层
        self.pred_model = self.build_pred_layers(
            hidden_dim*3, pred_hidden_dims,
            label_dim, num_aggs=self.num_aggs)  #三层

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)



    def get_mask(self, score, mask, batch_size, batch_num_nodes):


        score = [score[i].split([batch_num_nodes[i], self.max_num_nodes - batch_num_nodes[i]], dim=0)[0] for i in
                       range(self.batch_num )]
        pad1 = [nn.ZeroPad2d(padding=(0, self.max_num_nodes - int(batch_num_nodes[i]), 0, 0)) for i in range(batch_size)]
        score = [pad1[i](score[i].unsqueeze(0)) for i in range(batch_size)]
        score = torch.cat([score[i] for i in range(batch_size)])
        score =score+(-10000)*(1-mask)
        score_ori = score
        score = score.sort(dim=1, descending=True)[0]

        k = (self.assign_ratio * batch_num_nodes.to(torch.float)).ceil().to(torch.long)

        num = [score[i][int(k[i])].unsqueeze(0).unsqueeze(1) for i in range(batch_size)]


        num_ = torch.cat([num[i] for i in range(self.batch_num )])

        score_ori = score_ori - num_
        m_x_ = F.relu(score_ori).unsqueeze(2)
        score_mask =torch.sign( F.relu(score_ori))
        m_x = [torch.t(score_mask[i].repeat(self.args.hidden_dim,1)) for i in range(batch_size)]
        m_x = torch.cat([m_x[i].unsqueeze(0) for i in range(batch_size)])

        m_adj = [torch.matmul(torch.t(score_mask[i].unsqueeze(0)), score_mask[i].unsqueeze(0)) for i in range(batch_size)]
        m_adj = torch.cat([m_adj[i].unsqueeze(0) for i in range(batch_size)])

        # m_x_ = torch.cat([score_mask[i].unsqueeze(0) for i in range(batch_size)])

        return m_x, m_x_, m_adj, score_mask



    def Pool(self,x, adj, pool_layer, batch_num_nodes, m, batch_size):
        score = pool_layer(x,adj).squeeze()
        batch_num_nodes = torch.tensor(batch_num_nodes)

        m_x, m_x_, m_adj, mask_next = self.get_mask(score, m, batch_size, batch_num_nodes)

        x = torch.mul(x,m_x)
        x = torch.mul(x,torch.tanh(m_x_))
        adj = torch.mul(adj,m_adj)

        return x, adj, mask_next

    def get_fea_A(self, x, batch_size, ratio):

        x_sim = torch.matmul(torch.cat([torch.t(x[i]).unsqueeze(0) for i in range(batch_size)]),
                             torch.cat([F.normalize(x[i]).unsqueeze(0) for i in range(batch_size)]))
        x_sim = torch.cat([((torch.t(x_sim[i]) + x_sim[i]) / 2).unsqueeze(0) for i in range(batch_size)])

        node_num = int(x_sim.size(2) * x_sim.size(2) * ratio)
        x_sim_sort = [x_sim[i].view(-1).sort(dim=0, descending=True)[0] for i in range(batch_size)]
        num = [x_sim_sort[i][node_num] for i in range(batch_size)]
        fea_A = torch.cat([torch.sign(F.relu(x_sim[i] - num[i])).unsqueeze(0) for i in range(batch_size)])
        return fea_A

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        # self.adj.data = adj
        # self.A = adj

        # mask
        max_num_nodes = self.adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.out_all = []
        self.read_out = []

        # if x is not None:
        x = self.conv_first(x, adj)
        x = self.act(x)
        self.x0 = x
        # self.out, _ = torch.max(x, dim=1)
        self.batch_num_nodes = torch.tensor(batch_num_nodes)
        self.index = [torch.arange(1, self.batch_num_nodes[i], dtype=torch.long, device=x.device).unsqueeze(0) for i in range(self.batch_num )]
        self.index1 = [self.index[i].cpu().numpy() for i in range(self.batch_num )]
        self.pad = [nn.ZeroPad2d(padding=(0,self.max_num_nodes - int(batch_num_nodes[i]),0,0)) for i in range(self.batch_num )]
        self.index =[self.pad[i](self.index[i]) for i in range(self.batch_num )]
        self.index = torch.cat([self.index[i] for i in range(self.batch_num )], dim=0)
        self.mask_0 = torch.zeros(self.batch_num , self.max_num_nodes).cuda().scatter_(1, self.index, 1)


        self.x_pool1, self.adj_pool1, self.mask_1 = self.Pool(self.x0, adj, self.score_layer_1, batch_num_nodes, self.mask_0, self.batch_num )
        self.x1_ = torch.cat([torch.t(self.x_pool1[i]).unsqueeze(0) for i in range(self.batch_num)])

        self.fea_A1 = self.get_fea_A(self.x_pool1, self.args.batch_size, self.ratio)
        # self.x1_111 = [self.x1_[i].cpu().detach().numpy() for i in range(self.batch_size)]
        # self.fea_A11 = [self.fea_A1[i].cpu().detach().numpy() for i in range(self.batch_size)]
        self.out1 = self.read_forward(self.x1_, self.fea_A1, self.read_first, self.read_block, self.read_last, embedding_mask).squeeze(2)
        self.out_all.append(self.out1)


        # if self.pool_concat:
        #     self.out0 = torch.cat([torch.mean(self.x_pool1, dim=1), torch.max(self.x_pool1, dim=1)[0]], dim=1)
        #     self.out_all.append(self.out0)
        # else:
        #     self.X1 = torch.cat([torch.mean(self.x_pool1,dim=1), torch.max(self.x_pool1, dim=1)[0]], dim=1)


        if self.bn:
            x = self.apply_bn(x)

        #
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](self.x_pool1.cuda(), self.adj_pool1.cuda())
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            #
            self.x1 = x
            out, _ = torch.max(x, dim=1)
            #
        batch_num_nodes_1 = torch.tensor(batch_num_nodes)
        self.batch_num_nodes_1 = (self.assign_ratio * batch_num_nodes_1.to(torch.float)).ceil().to(torch.long)
        self.x_pool2, self.adj_pool2, self.mask_2 = self.Pool(self.x1, self.adj_pool1, self.score_layer_2, self.batch_num_nodes_1 ,self.mask_1, self.batch_num )

        self.x2_ = torch.cat([torch.t(self.x_pool2[i]).unsqueeze(0) for i in range(self.batch_num)])
        # self.a21 = [self.x2[i].cpu().detach().numpy() for i in range(10)]
        # self.a2 = [self.x2_[i].cpu().detach().numpy() for i in range(10)]

        self.fea_A2 = self.get_fea_A(self.x_pool2, self.args.batch_size, self.ratio)

        # self.a22 = [self.fea_A2[i].cpu().detach().numpy() for i in range(10)]
        self.out2 = self.read_forward(self.x2_, self.fea_A2, self.read_first_after_pool, self.read_block_after_pool,
                                      self.read_last_after_pool,
                                      embedding_mask).squeeze(2)
        # self.out2_ = [self.out2[i].cpu().detach().numpy() for i in range(10)]
        self.out_all.append(self.out2)



        # if self.pool_concat:
        #     # self.out1 = self.concat_gnn_2.concat_forward(self.x_pool2.cuda(), self.adj_pool2.cuda())
        #     self.out1 = torch.cat([torch.mean(self.x_pool2, dim=1), torch.max(self.x_pool2, dim=1)[0]], dim=1)
        #     self.out_all.append(self.out1)
        # else:
        #     self.X2 = torch.cat([torch.mean(self.x_pool2, dim=1), torch.max(self.x_pool2, dim=1)[0]], dim=1)

        x = self.conv_last(self.x_pool2.cuda(), self.adj_pool2.cuda())
        self.x2 = x
        self.batch_num_nodes_2 = (self.assign_ratio * self.batch_num_nodes_1.to(torch.float)).ceil().to(torch.long)
        self.x_pool3, self.adj_pool3, self.mask_3 = self.Pool(self.x2, self.adj_pool2, self.score_layer_3,
                                                              self.batch_num_nodes_2, self.mask_2, self.batch_num )
        out_3,_ = torch.max(self.x_pool3, dim=1)
        self.out_all.append(out_3)


        # if self.pool_concat:
        #     # self.out2 = self.concat_gnn_3.concat_forward(self.x_pool3.cuda(), self.adj_pool3.cuda())
        #     self.out2 = torch.cat([torch.mean(self.x_pool3, dim=1), torch.max(self.x_pool3, dim=1)[0]], dim=1)
        #     self.out_all.append(self.out2)
        # else:
        #     self.X3 = torch.cat([torch.mean(self.x_pool3, dim=1), torch.max(self.x_pool3, dim=1)[0]], dim=1)


        self.X = torch.cat(self.out_all, dim=1)
        ypred = self.pred_model(self.X)

        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SAGEPool_EGC, self).loss(pred, label)

        return loss


    def loss_node(self, pred, label, type='softmax'):
        # softmax + CE
        # if type == 'softmax':
        #     return -1 * F.cross_entropy(pred.long(), label.long(), size_average=True)
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -torch.sum(log_prob.cuda() * label.cuda()) / self.batch_num
        return loss

    def update_adj(self):
        return self.adj

