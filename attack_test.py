from block import *
from encoders1 import *
from data import *
import copy
import pickle as pkl


def arg_parse_1():
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
                        bmname='Ethereum_dynamic_1k',
                        num_nodes = 1000,
                        max_links=2000,
                        cuda='7',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=10,
                        num_epochs=200,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='diffpool',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()

args = arg_parse_1()
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


writer.close()


path = '/home/NewDisk/zhangdunjie/chainblock/diffpool-master/data/Normal first-order nodes'
path1 = '/home/NewDisk/zhangdunjie/chainblock/diffpool-master/data/Phishing first-order nodes'
# all_node_path = "./graph_nodes.txt"
all_node_path = "./data/graph_nodes_2000.txt"
test_node_path = "./data/graph_test_nodes_2000_1.txt"
data_path = "./data/TEGs(2k)1.npz"
test_data_path = "./data/test_nodes.npz"

node_list = get_node(path, path1, args.num_nodes, all_node_path)
test_nodes = get_node_test(path, path1, args.num_nodes, node_list, test_node_path)
# node_list =  get_node(path,path1, args.num_nodes, all_node_path)
# node = '0x92e14a71b86d4d0c5fa37174b19f1fed5c591863'
# adj1, fea1 = dynamic_G(node, 0, 2000, all_node_path)

Dynamic_Gset = con_dynamic_Gset(all_node_path,data_path, args.batch_size, args.max_links,max_n = 2000 )
train_dataset, val_dataset, test_dataset, train_num, val_num, test_num = split(Dynamic_Gset)
max_num_nodes = Dynamic_Gset['Adj'][0][0][0].shape[0]
# Dynamic_Graph_test = con_dynamic_Gset(test_node_path, test_data_path, args.batch_size, args.max_links, max_n=max_num_nodes)
# test_batch = len(Dynamic_Graph_test['Adj'])
# test_G = split_data(Dynamic_Graph_test, 0, test_batch)
# test_G_num = len(test_G['adj'])



input_dim = 2
assign_input_dim = 2

print(max_num_nodes)
#DEGD
# model = diffpool(
#     max_num_nodes,
#     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
#     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, pool_method = 'diffpool',
#     bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
#     assign_input_dim=assign_input_dim).cuda()


#DEGD-S
model = encoders.diffpool(
    max_num_nodes,
    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
    assign_input_dim=assign_input_dim).cuda()


# _, val_accs = train_phishing_detector_dy(train_dataset, model,train_num, val_num, test_num, args, val_dataset=val_dataset, test_dataset=test_dataset,
#                     writer=writer)

# _, val_accs = train_phishing_detector_dy(train_dataset, model,train_num, val_num, test_G_num, args, val_dataset=val_dataset, test_dataset=test_G,
#                     writer=writer)


#'TEGD-diffpool',TEGD-no-time-coefficient
mth = os.path.join('/home/NewDisk/zhangdunjie/xhy/TEGDetect/model_param/'+'TEGD-no-time-coefficient' + '.pth')
checkpoint = torch.load(mth)
model.load_state_dict(checkpoint['net'])
# evaluate_dynamic(test_G, test_G_num, model, args, name='Test')
# print(1)


def is_to_modify(g, link):
    if g > 0 and link == 0:
        modify = 1
    elif g <= 0 and link == 1:
        modify = 0
    else:
        modify = -1

    return modify


def GA(gradients, inputs, batch_num_nodes, attack_link_rate):

    _inputs = copy.deepcopy(inputs)

    def construct_mask(max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        # packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size,10, max_nodes, max_nodes)
        for i in range(batch_size):
            out_tensor[i,:, :batch_num_nodes[i][0], :batch_num_nodes[i][0]] = 1
        return out_tensor.cuda()

    gradients_mask = construct_mask(inputs.shape[2], batch_num_nodes)
    gradients = gradients * gradients_mask

    one_graph = inputs.shape[2] * inputs.shape[3]

    for idx in range(inputs.shape[0]):
        _, sorted_index = torch.sort(torch.reshape(torch.abs(gradients[idx:idx + 1]), (-1,)), descending=True)
        modify_link = int(_inputs.shape[2] * attack_link_rate)

        max_value, _ = torch.max(torch.reshape(_inputs[idx:idx + 1], (10, -1)), dim=1)

        num_iter = 0
        for i in range(len(sorted_index)):

            ts, row, col = sorted_index[i] // one_graph, (sorted_index[i] % one_graph) // _inputs.shape[2], (
                        sorted_index[i] % one_graph) % _inputs.shape[2]
            # max_value, _ = torch.max(torch.reshape(_inputs[idx:idx + 1,ts:ts+1], (10,-1)), dim=0)
            # g = gradients[idx, ts, row, col].item()
            v = inputs[idx, ts, row, col].item()
            if v == 0:
                link_value = random.random() * float(max_value[idx])
                # link_value = 1
                value = link_value if v == 0 else 0
                _inputs[idx, ts, row, col] = torch.cuda.FloatTensor([value])
                num_iter += 1
                if num_iter >= modify_link:
                    break




            # if g >= 0:
            #     if g == 0:
            #         link_value = random.random() * float(max_value)
            #         value = link_value if v == 0 else 0
            #         _inputs[idx, ts, row, col] = torch.cuda.FloatTensor([value])
            #         num_iter += 1
            #         if num_iter >= modify_link:
            #             break
            #     else:
            #         is_value = is_to_modify(g, v)
            #         if is_value == 1:
            #             link_value = random.random() * float(max_value)
            #             value = link_value if v == 0 else 0
            #             _inputs[idx, ts, row, col] = torch.cuda.FloatTensor([value])
            #             num_iter += 1
            #             if num_iter >= modify_link:
            #                 break




            # if i >= _inputs.shape[2]*(attack_link_rate/0.1):
            #     break
    return _inputs


def attack_detector(dataset, batch_num, model, args,modify_rate, max_num_examples=None):
    labels = []
    clean_preds = []
    attack_preds = []
    for batch_idx in range(batch_num):
        adj = dataset['adj'][batch_idx]
        h0 = dataset['fea'][batch_idx]
        labels.append((Variable(torch.Tensor(dataset['label'][batch_idx])).cuda()).to(torch.int64))
        label = (Variable(torch.Tensor(dataset['label'][batch_idx])).cuda()).to(torch.int64)
        batch_num_nodes = dataset['batch_num_nodes'][batch_idx]
        assign_input = dataset['assign_input'][batch_idx]
        # torch.cuda.empty_cache()

        adj = torch.cat([torch.unsqueeze(
            torch.cat([torch.unsqueeze(torch.from_numpy(adj[i][j].A), 0) for j in range(len(adj[0]))], 0), 0) for i in
            range(len(adj))], 0).cuda()
        adj = Variable(adj,requires_grad=True)
        adj_clean = copy.deepcopy(adj)

        # ypred, att = model(h0, adj, batch_num_nodes, Is_attack = True, assign_x=assign_input) #DEGD
        ypred = model(h0, adj, batch_num_nodes, Is_attack=True, assign_x=assign_input) #DEGD-S
        torch.cuda.empty_cache()
        if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
        else:
            loss = model.loss(ypred, label, adj, batch_num_nodes)
        grad = torch.autograd.grad(loss, adj)[0].data
        with torch.no_grad():
            torch.cuda.empty_cache()
            #clean data
            _, indices = torch.max(ypred, 1)
            clean_preds.append(indices.cpu().data.numpy())

            print('-----')
            adj = GA(grad, adj, batch_num_nodes, modify_rate)
            modify_link =  torch.sum(torch.abs(adj-adj_clean)).item()
            print('modify_link',modify_link)
            print('*******')
            torch.cuda.empty_cache()


            #重新计算攻击后的值
            # ypred, att = model(h0, adj, batch_num_nodes, Is_attack=True, assign_x=assign_input) #DEGD
            ypred = model(h0, adj, batch_num_nodes, Is_attack=True, assign_x=assign_input) #DEGD-S

            torch.cuda.empty_cache()
            # attack data
            _, indices = torch.max(ypred, 1)
            attack_preds.append(indices.cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * args.batch_size > max_num_examples:
                    break
            print(batch_idx)
            torch.cuda.empty_cache() #无用

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


attack_detector(test_dataset, test_num, model, args,0.5)