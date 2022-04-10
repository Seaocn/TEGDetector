from block import *
from data import *
# def main():
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
test_node_path = "./data/graph_test_nodes_2000_4.txt"
data_path = "./data_500.npz"
test_data_path = "./data/test_data_2000_5.npz"
node_list = get_node(path, path1, args.num_nodes, all_node_path)
test_nodes = get_node_test(path, path1, args.num_nodes, node_list, test_node_path)
# 构建交易模式图集合
Graph_train_set = con_Gset(all_node_path, data_path, args.batch_size, args.max_links)
# max_num_nodes = Graph_train_set['max_num']
# Graph_test_set = con_Gset(test_node_path, test_data_path, args.batch_size, args.max_links, max_n=max_num_nodes)
#

train_dataset, val_dataset, test_dataset, train_num, val_num, test_num = split(Graph_train_set)

Graph_test_set = con_Gset(test_node_path, test_data_path, args.batch_size, args.max_links)
# test_batch = Graph_test_set['Adj'].shape[0]
test_batch = len(Graph_test_set['Adj'])
test_G = split_data(Graph_test_set, 0, test_batch)

# train_num = int(len(Adj)/10*7)
# val_num = int(len(Adj)/10*2)
# test_num = int(len(Adj)/10*1)
# train_dataset = split_data(Graph_train_set, 0, train_num)
# val_dataset = split_data(Graph_train_set, train_num, val_num)
# test_dataset = split_data(Graph_train_set, train_num+val_num, test_num)


input_dim = 2
assign_input_dim = 2
max_num_nodes = Graph_test_set['max_num']

model = encoders.SoftPoolingGcnEncoder(
    max_num_nodes,
    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
    assign_input_dim=assign_input_dim).cuda()

_, val_accs = train_phishing_detector(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
                                      writer=writer)

mth = os.path.join(args.logdir, gen_prefix(args), str(67) + '.pth')
checkpoint = torch.load(mth)
model.load_state_dict(checkpoint['net'])
evaluate(test_G, test_batch, model, args, name='Test')
print(1)