import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter
from data import *
import pandas as pd
# from att import att_con_Gset


import copy
import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util


def evaluate(dataset, batch_num, model, args, name='Validation', max_num_examples=None):
    model.eval()
    labels = []
    preds = []
    for batch_idx in range(batch_num):
        adj = Variable(torch.from_numpy(dataset['adj'][batch_idx]), requires_grad=False).cuda()
        h0 = Variable(torch.from_numpy(dataset['fea'][batch_idx]), requires_grad=False).cuda()
        labels.append((Variable(torch.Tensor(dataset['label'][batch_idx])).cuda()).to(torch.int64))
        batch_num_nodes = torch.from_numpy(dataset['batch_num_nodes'][batch_idx]).numpy() if True else None
        assign_input = Variable(torch.from_numpy(dataset['assign_input'][batch_idx]), requires_grad=False).cuda()
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, "  prec",result['prec'], "  recall",result['recall'],  " accuracy:", result['acc'], " F1:", result['F1'])

    return result



def evaluate_dynamic(dataset, batch_num, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx in range(batch_num):
        adj = dataset['adj'][batch_idx]
        h0 = dataset['fea'][batch_idx]
        labels.append((Variable(torch.Tensor(dataset['label'][batch_idx])).cuda()).to(torch.int64))
        batch_num_nodes = dataset['batch_num_nodes'][batch_idx]
        assign_input = dataset['assign_input'][batch_idx]
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, "  prec",result['prec'], "  recall",result['recall'],  " accuracy:", result['acc'], " F1:", result['F1'])
    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method +'_' + str(args.num_nodes) +'_' +str(args.max_links)
    # if args.method == 'diffpool':
    #     name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
    #     name += '_ar' + str(int(args.assign_ratio * 100))
    #     if args.linkpred:
    #         name += '_lp'
    # else:
    #     name += '_l' + str(args.num_gc_layers)
    # name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    # if not args.bias:
    #     name += '_nobias'
    # if len(args.name_suffix) > 0:
    #     name += '_' + args.name_suffix
    return name


def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    # fig = plt.figure(figsize=(8,6), dpi=300)
    # for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    # plt.tight_layout()
    # fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters - 1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed

            # log once per XX epochs
            if epoch % 10 == 0 and batch_idx == len(
                    dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if args.log_graph:
                    log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def train_phishing_detector_dy(train_dataset,model, train_num, val_num, test_num, args, val_dataset = None,test_dataset=None, writer=None ):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    best_test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_epochs = []
    test_accs = []
    val_accs = []
    best_test_epochs = []
    all_time = 0
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        t = time.time()
        print('Epoch: ', epoch)
        for batch_idx in range(train_num):
            begin_time = time.time()
            model.zero_grad()
            # if args.normalize:
            #     adj = Variable(Adj[0].float(), requires_grad=False).cuda()
            # else:
            adj = train_dataset['adj'][batch_idx]
            h0 = train_dataset['fea'][batch_idx]
            label = (Variable(torch.Tensor(train_dataset['label'][batch_idx])).cuda()).to(torch.int64)
            batch_num_nodes = train_dataset['batch_num_nodes'][batch_idx]
            assign_input = train_dataset['assign_input'][batch_idx]
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed
            # print('train_batach_time：',elapsed)

        train_time = time.time() - t
        all_time += train_time
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time, ';   train_time：', train_time)
        result = evaluate_dynamic(train_dataset, train_num, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate_dynamic(val_dataset, val_num, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            if val_result['acc'] > best_val_result['acc'] - 1e-7:
                best_val_result['acc'] = val_result['acc']
                best_val_result['epoch'] = epoch
                best_val_result['loss'] = avg_loss
                best_val_result['pre'] = val_result['prec']
                best_val_result['recall'] = val_result['recall']
                best_val_result['F1'] = val_result['F1']
                print('Best val result: ', best_val_result)
                best_val_epochs.append(best_val_result['epoch'])
        if test_dataset is not None:
            test_result = evaluate_dynamic(test_dataset, test_num, model, args, name='Test')
            test_accs.append(test_result['acc'])
            if test_result['acc'] > best_test_result['acc'] - 1e-7:
                best_test_result['acc'] = test_result['acc']
                best_test_result['epoch'] = epoch
                best_test_result['loss'] = avg_loss
                best_test_result['pre'] = test_result['prec']
                best_test_result['recall'] = test_result['recall']
                best_test_result['F1'] = test_result['F1']





        print('Best Test result: ', best_test_result)
        best_test_epochs.append(best_test_result['epoch'])
        #

        if epoch % 20 == 0 or (test_result['acc'] > best_test_result['acc'] - 1e-7):
            model_state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            mth = os.path.join(args.logdir, gen_prefix(args), str(epoch) + '.pth')
            torch.save(model_state, mth)

    print(all_time / args.num_epochs)
    return  model, val_accs

def train_phishing_detector(train_dataset,model, train_num, val_num, test_num, args, val_dataset = None,test_dataset=None, writer=None ):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    best_test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_epochs = []
    test_accs = []
    val_accs = []
    best_test_epochs = []
    all_time = 0
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        t = time.time()
        print('Epoch: ', epoch)
        for batch_idx in range(train_num):
            begin_time = time.time()
            model.zero_grad()
            # if args.normalize:
            #     adj = Variable(Adj[0].float(), requires_grad=False).cuda()
            # else:

            adj = Variable(torch.from_numpy(train_dataset['adj'][batch_idx]), requires_grad=False).cuda()
            h0 = Variable(torch.from_numpy(train_dataset['fea'][batch_idx]), requires_grad=False).cuda()
            label = (Variable(torch.Tensor(train_dataset['label'][batch_idx])).cuda()).to(torch.int64)
            batch_num_nodes = torch.from_numpy(train_dataset['batch_num_nodes'][batch_idx]).numpy() if True else None
            assign_input = Variable(torch.from_numpy(train_dataset['assign_input'][batch_idx]),
                                    requires_grad=False).cuda()
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed

        train_time = time.time() - t
        all_time += train_time
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time, ';   train_time：', train_time)
        result = evaluate(train_dataset, train_num, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, val_num, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            if val_result['acc'] > best_val_result['acc'] - 1e-7:
                best_val_result['acc'] = val_result['acc']
                best_val_result['epoch'] = epoch
                best_val_result['loss'] = avg_loss
                best_val_result['pre'] = val_result['prec']
                best_val_result['recall'] = val_result['recall']
                best_val_result['F1'] = val_result['F1']
        if test_dataset is not None:
            test_result = evaluate(test_dataset, test_num, model, args, name='Test')
            test_accs.append(test_result['acc'])
            if test_result['acc'] > best_test_result['acc'] - 1e-7:
                best_test_result['acc'] = test_result['acc']
                best_test_result['epoch'] = epoch
                best_test_result['loss'] = avg_loss
                best_test_result['pre'] = test_result['prec']
                best_test_result['recall'] = test_result['recall']
                best_test_result['F1'] = test_result['F1']

                model_state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                mth = os.path.join('/home/NewDisk/zhangdunjie/xhy/TEGDetect/model_param/I_BGNN-static-lr-0.05.pth')
                torch.save(model_state, mth)



        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])

        print('Best Test result: ', best_test_result)
        best_test_epochs.append(best_val_result['epoch'])
        #

        # if epoch % 20 == 0 or (test_result['acc'] > best_test_result['acc'] - 1e-7):
        #     model_state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     mth = os.path.join(args.logdir, gen_prefix(args), str(epoch) + '.pth')
        #     torch.save(model_state, mth)

    print(all_time / args.num_epochs)
    return  model, val_accs

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim



def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = util.node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


#攻击
def is_to_modify(g, link):
    if g > 0 and link == 0:
        modify = 1
    elif g <= 0 and link == 1:
        modify = 0
    else:
        modify = -1

    return modify



def GA_static(gradients, inputs, batch_num_nodes, attack_link_rate):

    _inputs = copy.deepcopy(inputs)

    def construct_mask(max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        # packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes, max_nodes)
        for i in range(batch_size):
                out_tensor[i, :batch_num_nodes[i], :batch_num_nodes[i]] =1
        return out_tensor.cuda()

    gradients_mask =  construct_mask(inputs.shape[2], batch_num_nodes)
    gradients = gradients * gradients_mask


    one_graph = inputs.shape[1] * inputs.shape[2]

    for idx in range(inputs.shape[0]):
        grad_value, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[idx:idx+1]), (-1,)), descending=True)
        # a = torch.sum(torch.where(grad_value>0,torch.cuda.FloatTensor([1]),torch.cuda.FloatTensor([0])))

        modify_link = int(_inputs.shape[2] * attack_link_rate)
        num_iter = 0
        max_value,_ =  torch.max(torch.reshape(_inputs[idx:idx+1], (-1,)),dim=0)
        for i in range(len(sorted_index)):
            row, col = (sorted_index[i] % one_graph) // _inputs.shape[2], (
                    sorted_index[i] % one_graph) % _inputs.shape[2]
            # g = gradients[idx, row, col].item()
            v = inputs[idx, row, col].item()
            if v == 0:
                link_value = random.random() * float(max_value)
                value = link_value if v == 0 else 0
                _inputs[idx, row, col] = torch.cuda.FloatTensor([value])
                num_iter += 1
                if num_iter >= modify_link:
                    break

                # if g == 0:
                #     link_value = random.random() * float(max_value)
                #     link_value = 1
                #     value = link_value if v == 0 else 0
                #     _inputs[idx, row, col] = torch.cuda.FloatTensor([value])
                #     num_iter += 1
                #     if num_iter >= modify_link:
                #         break
                # else:
                #     is_value = is_to_modify(g, v)
                #     #区块链攻击不允许删除交易
                #     if is_value == 1:
                #         link_value = random.random() * float(max_value)
                #         link_value = 1
                #         value = link_value if v == 0 else 0
                #         _inputs[idx, row, col] = torch.cuda.FloatTensor([value])
                #         num_iter += 1
                #         if num_iter >= modify_link:
                #             break

                    # if i >= _inputs.shape[2]*(attack_link_rate/0.1):
                    #     break
    return _inputs

def attack_detector(dataset, batch_num, model, args, modify_rate, max_num_examples=None):
    labels = []
    clean_preds = []
    attack_preds = []
    for batch_idx in range(batch_num):
        adj = Variable(torch.from_numpy(dataset['adj'][batch_idx]), requires_grad=False).cuda()
        h0 = Variable(torch.from_numpy(dataset['fea'][batch_idx]), requires_grad=False).cuda()
        labels.append((Variable(torch.Tensor(dataset['label'][batch_idx])).cuda()).to(torch.int64))
        label=Variable(torch.Tensor(dataset['label'][batch_idx])).cuda().to(torch.int64)
        batch_num_nodes = torch.from_numpy(dataset['batch_num_nodes'][batch_idx]).numpy() if True else None
        assign_input = Variable(torch.from_numpy(dataset['assign_input'][batch_idx]), requires_grad=False).cuda()

        adj = Variable(adj, requires_grad=True)
        adj_clean = copy.deepcopy(adj)
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        torch.cuda.empty_cache()

        if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
        else:
            loss = model.loss(ypred, label, adj, batch_num_nodes)
        grad = torch.autograd.grad(loss, adj)[0].data
        with torch.no_grad():
            torch.cuda.empty_cache()
            # clean data
            _, indices = torch.max(ypred, 1)
            clean_preds.append(indices.cpu().data.numpy())

            adj = GA_static(grad, adj, batch_num_nodes, modify_rate)
            modify_link = torch.sum(torch.abs(adj - adj_clean)).item()
            print('modify_link', modify_link)
            torch.cuda.empty_cache()

            # 重新计算攻击后的值
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            torch.cuda.empty_cache()
            # attack data
            _, indices = torch.max(ypred, 1)
            attack_preds.append(indices.cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * args.batch_size > max_num_examples:
                    break
            print(batch_idx)
            torch.cuda.empty_cache()  # 无用

    labels = np.hstack(labels)
    clean_preds = np.hstack(clean_preds)
    attack_preds = np.hstack(attack_preds)

    clean_result = {'prec': metrics.precision_score(labels, clean_preds, average='macro'),
                    'recall': metrics.recall_score(labels, clean_preds, average='macro'),
                    'acc': metrics.accuracy_score(labels, clean_preds),
                    'F1': metrics.f1_score(labels, clean_preds, average="micro")}

    attack_result = {'prec': metrics.precision_score(labels, attack_preds, average='macro'),
                     'recall': metrics.recall_score(labels, attack_preds, average='macro'),
                     'acc': metrics.accuracy_score(labels, attack_preds),
                     'F1': metrics.f1_score(labels, attack_preds, average="micro")}
    # print(name, " clean_prec",result['prec'], "  recall",result['recall'],  " accuracy:", result['acc'], " F1:", result['F1'])
    # print(att)
    print('clean_test_results', clean_result)
    print('attack_test_results', attack_result)

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-links', dest='max_links', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-node', dest='num_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='bmname',  # syn1v2
                        bmname='Ethereum_1k',
                        num_nodes = 2000,
                        max_links=2000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=10,
                        num_epochs=50,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=32,
                        output_dim=32,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='diffpool',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()

def main():
    args = arg_parse()
    # export scalar data to JSON for external processing
    path = os.path.join(args.logdir, gen_prefix(args))
    # if os.path.isdir(path):
    #     print('Remove existing log dir: ', path)
    #     shutil.rmtree(path)
    writer = SummaryWriter(path)
    # writer = None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.system('nvidia-smi')
    print('CUDA', args.cuda)

    # if prog_args.bmname is not None:
    #     benchmark_task_val(prog_args, writer=writer)
    np.random.seed(0)

    writer.close()


    path = '/home/NewDisk/zhangdunjie/graph/diffpool-master/data/Normal first-order nodes'
    path1 = '/home/NewDisk/zhangdunjie/graph/diffpool-master/data/Phishing first-order nodes'
    # all_node_path = "./graph_nodes.txt"
    all_node_path = "./data/graph_nodes_2000.txt"
    test_node_path = "./data/graph_test_nodes_2000_1.txt"
    data_path = "./data/all_static_data.npz"
    # data_path = "/home/NewDisk/zhangdunjie/chainblock/diffpool-master/data_1000(1k).npz"
    test_data_path = "./data/static_test_data.npz" #_phshing
    # node_list =  get_node(path,path1, args.num_nodes, all_node_path)
    # test_nodes = get_node_test(path,path1, args.num_nodes, node_list, test_node_path)
    #构建交易模式图集合
    print(1)
    # Graph_train_set = con_Gset(all_node_path,data_path, args.batch_size, args.max_links, max_n=2000)
    # max_num_nodes = Graph_train_set['max_num']
    # Graph_test_set = con_Gset(test_node_path,test_data_path, args.batch_size, args.max_links, max_n = max_num_nodes ) #, max_n = max_num_nodes
    # max_num_nodes = Graph_test_set['max_num']
    # f_path = './data/normal_nodes.txt'#'./data/normal_nodes.txt'  phshing_nodes
    # f_nor = open(f_path,'r+')
    # t = 0.6
    print(2)
    t0 = time.time()
    Graph_train_set, adv_n, adv_l = con_Gset(all_node_path, data_path, args.batch_size, args.max_links, max_n=2000)


    #att_con_Gset(all_node_path, test_data_path, args.batch_size, args.max_links, f_nor, t, max_n=1990, att_nor = False)#con_Gset(all_node_path, data_path, args.batch_size, args.max_links, max_n = max_num_nodes)

    #re-build fea
    # Graph_train_set['Fea'] = np.array(np.ones((200,10,2000,2), dtype=np.float32()))



    t1 = time.time()
    print(t1 - t0,'  ',adv_n,'   ',adv_l)
    train_dataset, val_dataset, test_dataset,  train_num, val_num, test_num = split(Graph_train_set)

    # Graph_test_set, adv_n, adv_l = con_Gset(test_node_path, test_data_path, args.batch_size, args.max_links, max_n=2000)
    # test_batch = Graph_test_set['Adj'].shape[0]
    # test_G = split_data(Graph_test_set,0,test_batch)

    # train_num = int(len(Adj)/10*7)
    # val_num = int(len(Adj)/10*2)
    # test_num = int(len(Adj)/10*1)
    # train_dataset = split_data(Graph_train_set, 0, train_num)
    # val_dataset = split_data(Graph_train_set, train_num, val_num)
    # test_dataset = split_data(Graph_train_set, train_num+val_num, test_num)


    input_dim = 2
    assign_input_dim = 2
    max_num_nodes = Graph_train_set['max_num']

    #MCGC
    # model = encoders.SoftPoolingGcnEncoder(
    #     max_num_nodes,
    #     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
    #     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
    #     bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
    #     assign_input_dim=assign_input_dim).cuda()

    #I_BGNN
    model = encoders.I_BGNN(
        max_num_nodes,
        input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        assign_input_dim=assign_input_dim).cuda()


    # _, val_accs = train_phishing_detector(train_dataset, model, train_num, val_num, test_num, args, val_dataset=None, test_dataset=test_dataset,
    #                     writer=writer)


    mth = os.path.join('/home/NewDisk/zhangdunjie/xhy/TEGDetect/model_param/I_BGNN-static-lr-0.05.pth')
    checkpoint = torch.load(mth)
    model.load_state_dict(checkpoint['net'])
    #
    # result = evaluate(test_G, test_batch, model, args, name='Train', max_num_examples=100)
    # print('result',result)

    attack_detector(test_dataset, test_num, model, args, 0.5)


main()
print(1)