import json
import random
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import MessagePassing, PairNorm
from torch_geometric.nn import SAGEConv # sparse matrix
from torch_geometric.nn import GATv2Conv # global attention
from torch_geometric.utils import dropout_edge, negative_sampling

import scipy.sparse as sp
from networkx.readwrite import json_graph
from sklearn.metrics import f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 以下为数据处理部分 ----------------- #
def encode_label(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels


def load_cite_data(path="../data/cora/", dataset="cora", task='node'):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 得到论文单词组成，也就是节点特征
    x = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 得到论文的类别，也就是节点标签。 把字符类型标签映射为类别
    y = encode_label(idx_features_labels[:, -1])
    num_classes = torch.tensor(np.max(y) + 1)

    # 把节点名称映射为下标，方便后面提取边
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))

    # 提取边
    edge_index = [[], []]
    for i in range(edges_unordered.shape[0]):
        try:  # 判断边的两端是否在节点列表中，如果不在就去掉这条边
            start_idx = idx_map[edges_unordered[i, 0]]
            end_idx = idx_map[edges_unordered[i, 1]]
        except KeyError:
            continue
        edge_index[0].append(start_idx)
        edge_index[1].append(end_idx)
    edge_index = np.array(edge_index, dtype=np.int32)
    
    # 转换为 tensor 类型
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    edge_index = torch.LongTensor(edge_index)
    
    data_ = Data(x=x, edge_index=edge_index, y=y)
    if task == 'node':  # 节点分类任务
        # 随机划分训练集、验证集和测试集
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        mask = np.random.permutation(idx.shape[0])
        train_mask = mask[: int(idx.shape[0] * 0.6)]
        val_mask = mask[int(idx.shape[0] * 0.6): int(idx.shape[0] * 0.8)]
        test_mask = mask[int(idx.shape[0] * 0.8):]

        # 把数据转换为合适的类型，并且封装到图数据的类型
        train_mask = torch.LongTensor(train_mask)
        val_mask = torch.LongTensor(val_mask)
        test_mask = torch.LongTensor(test_mask)

        data_.train_mask = train_mask
        data_.val_mask = val_mask
        data_.test_mask = test_mask
        data_.num_classes = num_classes
    else:  # 链路预测任务
        # 随机划分边为训练集，验证集和测试集。训练集无负样本。图为有向图。
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        transform = RandomLinkSplit(is_undirected=False, num_val=0.2, num_test=0.2,
                                    add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data_)
        data_.train_pos_edge_index = train_data.edge_label_index
        data_.val_edge_index = val_data.edge_label_index
        data_.val_edge_label = val_data.edge_label
        data_.test_edge_index = test_data.edge_label_index
        data_.test_edge_label = test_data.edge_label

    return data_


def load_ppi_data(path="../data/ppi/", dataset="ppi", task='node'):
    prefix = path + dataset
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    feats = np.load(prefix + "-feats.npy")
    class_map = json.load(open(prefix + "-class_map.json"))

    x = feats
    x = torch.FloatTensor(x)
    edge_index = np.array(G.edges()).T
    edge_index = torch.LongTensor(edge_index)
    str_nodes = list(map(str, G.nodes))
    y = np.array(list(map(class_map.get, str_nodes)))
    y = torch.FloatTensor(y)  # 这里需要使用 Float 类型，后面计算 BCELoss 时需要输入为 Float 类型
    data_ = Data(x=x, edge_index=edge_index, y=y)
    
    if task == 'node':  # 节点分类任务
        num_classes = torch.tensor(y.size(1))
        train_mask = []
        val_mask = []
        test_mask = []
        # 这里数据集中已经划分好训练集、验证集和测试集。
        for node in G.nodes():
            if G.nodes()[node]['val']:
                val_mask.append(node)
            elif G.nodes()[node]['test']:
                test_mask.append(node)
            else:
                train_mask.append(node)
        train_mask, val_mask = torch.LongTensor(train_mask), torch.LongTensor(val_mask)
        test_mask = torch.LongTensor(test_mask)
        
        data_.train_mask = train_mask
        data_.val_mask = val_mask
        data_.test_mask = test_mask
        data_.num_classes = num_classes
    else:  # 链路预测任务
        # 随机划分边为训练集，验证集和测试集。训练集无负样本。图为无向图。
        # 训练集: 验证集: 测试集 = 0.6: 0.2: 0.2
        transform = RandomLinkSplit(is_undirected=True, num_val=0.2, num_test=0.2,
                                    add_negative_train_samples=False)
        train_data, val_data, test_data = transform(data_)
        data_.train_pos_edge_index = train_data.edge_label_index
        data_.val_edge_index = val_data.edge_label_index
        data_.val_edge_label = val_data.edge_label
        data_.test_edge_index = test_data.edge_label_index
        data_.test_edge_label = test_data.edge_label
    return data_


def load_data(path, dataset, task='node'):
    print('Loading {} dataset...'.format(dataset))
    if dataset != "ppi":
        return load_cite_data(path, dataset, task)
    else:
        return load_ppi_data(path, dataset, task)


# ----------------- 以下为评估指标部分 ----------------- #
def accuracy(output, labels, dataset, task='node'):
    if task == 'node':
        if dataset != "ppi":  # 正确率
            preds = output.max(dim=1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            score = correct.item() / len(labels)
        else:  # Micro F1 score
            predict = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predict = np.where(predict > 0.5, 1, 0)
            score = f1_score(labels, predict, average='micro')
    else:  # AUC 指标
        predict = output.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        score = roc_auc_score(labels, predict)
    return score


def criterion(output, target, dataset, task='node'):
    # print(output.dtype, target.dtype)
    if task == 'node':
        if dataset != 'ppi':
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            output = nn.Sigmoid()(output)
            loss = nn.BCELoss()(output, target)
    else:  # 二分类任务
        loss = nn.functional.binary_cross_entropy_with_logits(output, target)

    return loss


# ----------------- 以下为模型部分 ----------------- #
def _add_self_loops(edge_index, num_nodes):
    """
    添加自环
    """
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


def _degree(index, num_nodes, dtype):
    """
    计算每个节点的度
    """
    out = torch.zeros((num_nodes), dtype=dtype, device=index.device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True):
        super(GCNConv, self).__init__(aggr='mean')  # "mean" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        col_deg = _degree(col, x.size(0), dtype=x.dtype)
        row_deg = _degree(row, x.size(0), dtype=x.dtype)
        col_deg_inv_sqrt = col_deg.pow(-0.5)
        row_deg_inv_sqrt = row_deg.pow(-0.5)
        norm = row_deg_inv_sqrt[row] * col_deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class NodeNet(torch.nn.Module):
    """Network for node classification
    """

    def __init__(self, node_features, hidden_features, num_layers, num_classes, add_self_loops=True,
                 pair_norm=False, drop_edge=False, act_fn='prelu', conv_type="GCNConv"):
        super(NodeNet, self).__init__()
        self.pair_norm = pair_norm
        self.drop_edge = drop_edge
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_features
            out_channels = hidden_features
            if i == 0:
                in_channels = node_features
            
            if i == num_layers - 1:
                out_channels = num_classes
            
            if conv_type == "GCNConv":
                self.convs.append(GCNConv(in_channels=in_channels, out_channels=out_channels,
                                          add_self_loops=add_self_loops))
            elif conv_type == "SAGEConv":
                self.convs.append(SAGEConv(in_channels=in_channels, out_channels=out_channels))
            elif conv_type == "GATv2Conv":
                self.convs.append(GATv2Conv(in_channels=in_channels, out_channels=out_channels))
            else:
                raise NotImplementedError

        if pair_norm:
            self.pn = PairNorm()

        if act_fn == 'prelu':
            self.act_fn = nn.PReLU()
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.pn(x)
            if self.drop_edge:
                edge_index = dropout_edge(edge_index=edge_index, p=0.2)[0]
            # x = self.act_fn(x)
            if i != len(self.convs) - 1:
                x = self.act_fn(x)
        return x


class NodeClassification(object):
    def __init__(self, device="cuda", dataset="cora", path="../data/cora/"):
        super(NodeClassification, self).__init__()
        self.net = None
        self.device = torch.device(device)
        self.data = None
        self.path = path
        self.dataset = dataset

    def train(self, patience=3, epochs=10, lr=2e-5, hidden_features=16, num_layers=2, add_self_loops=True,
              pair_norm=False, drop_edge=False, test=True, act_fn='prelu', scheduler=None, conv_type="GCNConv"):
        self.data = load_data(path=self.path, dataset=self.dataset, task='node')
        num_classes = self.data['num_classes'].item()
        self.data = self.data.to(self.device)
        self.net = NodeNet(node_features=self.data.num_features, hidden_features=hidden_features,
                           num_classes=num_classes, num_layers=num_layers, add_self_loops=add_self_loops,
                           pair_norm=pair_norm, drop_edge=drop_edge, act_fn=act_fn, conv_type=conv_type)
        total_params = sum([param.nelement() for param in self.net.parameters()])
        # print(f">>> total params: {total_params}")
        self.net.to(self.device)
        optimizer = Adam(self.net.parameters(), lr=lr)
        if scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        elif scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        best_model_path = f"./{self.dataset}_node_best.pth"
        
        delay = 0
        best_val_loss = np.inf
        best_val_score = -1
        
        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []
        
        for epoch in range(epochs):
            self.net.train()
            optimizer.zero_grad()
            out = self.net(self.data)
            train_loss = criterion(out[self.data.train_mask], self.data.y[self.data.train_mask], self.dataset)
            train_losses.append(train_loss.item())
            train_score = accuracy(out[self.data.train_mask], self.data.y[self.data.train_mask], self.dataset)
            train_scores.append(train_score)
            train_loss.backward()
            optimizer.step()
            if scheduler:
                lr_scheduler.step()

            with torch.no_grad():
                self.net.eval()
                val_loss = criterion(out[self.data.val_mask], self.data.y[self.data.val_mask], self.dataset).item()
                val_losses.append(val_loss)
                val_score = accuracy(out[self.data.val_mask], self.data.y[self.data.val_mask], self.dataset)
                val_scores.append(val_score)
                if (epoch % 10) == 0:
                    print(f"epoch: {epoch}, train_loss: {train_loss.item():7.5f}, train_score: {train_score:7.5f}, "
                          f"val_loss: {val_loss:7.5f}, val_score: {val_score:7.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = val_score
                torch.save(self.net, best_model_path)
                delay = 0
            else:
                delay += 1
                if delay > patience:
                    break
        print(f"best_val_loss: {best_val_loss:7.4f}, best_val_score: {best_val_score:7.4f} \n"
              f"{best_val_loss:7.4f} | {best_val_score:7.4f} |")
        if test:  # whether test on test dataset
            self.net = torch.load(best_model_path)
            self.net.to(self.device)
            with torch.no_grad():
                self.net.eval()
                out = self.net(self.data)
                test_loss = criterion(out[self.data.test_mask], self.data.y[self.data.test_mask], self.dataset).item()
                test_score = accuracy(out[self.data.test_mask], self.data.y[self.data.test_mask], self.dataset)
                print(f"test_score {test_score:7.4f}")
        
        return train_losses, train_scores, val_losses, val_scores


class LinkNet(torch.nn.Module):
    """Network for link prediction
    """

    def __init__(self, node_features, hidden_features, num_layers, add_self_loops=True, pair_norm=False,
                 drop_edge=False, act_fn='prelu') -> None:
        super(LinkNet, self).__init__()
        self.pair_norm = pair_norm
        self.drop_edge = drop_edge
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GCNConv(in_channels=node_features, out_channels=hidden_features, add_self_loops=add_self_loops))
            else:
                self.convs.append(
                    GCNConv(in_channels=hidden_features, out_channels=hidden_features, add_self_loops=add_self_loops))
        if self.pair_norm:
            self.pn = PairNorm()
        if act_fn == 'prelu':
            self.act_fn = nn.PReLU()
        else:
            self.act_fn = nn.Tanh()

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.pn(x)
            if self.drop_edge:
                edge_index = dropout_edge(edge_index=edge_index, p=0.2)[0]
            if i != len(self.convs) - 1:
                x = self.act_fn(x)
        return x

    def decode(self, z, edge_index):
        # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  #[2, E]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # element-wise 乘法

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    

class LinkPrediction(object):
    def __init__(self, device="cuda", dataset="cora", path="../data/cora/") -> None:
        super(LinkPrediction, self).__init__()
        self.net = None
        self.device = torch.device(device)
        self.data = None
        self.path = path
        self.dataset = dataset

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)
        link_labels[:pos_edge_index.size(1)] = 1
        return link_labels

    def train(self, patience=3, epochs=10, lr=2e-5, hidden_features=16, num_layers=2, add_self_loops=True,
              pair_norm=False, drop_edge=False, test=True, act_fn='prelu'):
        self.data = load_data(self.path, self.dataset, 'link')
        self.data = self.data.to(self.device)
        self.net = LinkNet(self.data.num_features, hidden_features, num_layers, add_self_loops, pair_norm, drop_edge,
                           act_fn=act_fn)
        total_params = sum([param.nelement() for param in self.net.parameters()])
        # print(f">>> total params: {total_params}")
        self.net.to(self.device)
        optimizer = Adam(self.net.parameters(), lr=lr)
        best_model_path = f"./{self.dataset}_link_best.pth"
        
        delay = 0
        best_val_loss = np.inf
        best_val_score = -1
        
        train_losses = []
        train_scores = []
        val_losses = []
        val_scores = []
        
        for epoch in range(epochs):
            neg_edge_index = negative_sampling(edge_index=self.data.train_pos_edge_index,
                                               num_nodes=self.data.num_nodes,
                                               num_neg_samples=self.data.train_pos_edge_index.size(1))
            self.net.train()
            optimizer.zero_grad()
            z = self.net.encode(self.data.x, self.data.train_pos_edge_index)
            edge_index = torch.cat([self.data.train_pos_edge_index, neg_edge_index], dim=-1)
            link_logits = self.net.decode(z, edge_index)
            link_labels = self.get_link_labels(self.data.train_pos_edge_index, neg_edge_index).to(self.data.x.device)

            train_loss = criterion(link_logits, link_labels, dataset=self.data, task='link')
            train_losses.append(train_loss.item())
            train_score = accuracy(link_logits.sigmoid(), link_labels, self.dataset, 'link')
            train_scores.append(train_score)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.net.eval()

                z = self.net.encode(self.data.x, self.data.train_pos_edge_index)
                edge_index = self.data.val_edge_index

                link_logits = self.net.decode(z, edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = self.data.val_edge_label
                val_loss = criterion(link_logits, link_labels, self.dataset, 'link').item()
                val_losses.append(val_loss)
                val_score = accuracy(link_probs, link_labels, self.dataset, 'link')
                val_scores.append(val_score)
                if (epoch % 10) == 0:
                    print(f"epoch: {epoch}, train_loss: {train_loss.item():7.5f}, train_score: {train_score:7.5f}, "
                          f"val_loss: {val_loss:7.5f}, val_score: {val_score:7.5f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_score = val_score
                    torch.save(self.net, best_model_path)
                    delay = 0
                else:
                    delay += 1
                    if delay > patience:
                        break
        print(f"best_val_loss: {best_val_loss:7.4f}, best_val_score: {best_val_score:7.4f} \n"
              f"{best_val_loss:7.4f} | {best_val_score:7.4f} |")
        if test:  # whether test on test dataset
            self.net = torch.load(best_model_path)
            self.net.to(self.device)
            with torch.no_grad():
                self.net.eval()
                z = self.net.encode(self.data.x, self.data.train_pos_edge_index)
                edge_index = self.data.test_edge_index

                link_logits = self.net.decode(z, edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = self.data.test_edge_label
                test_loss = criterion(link_logits, link_labels, self.dataset, 'link')
                test_score = accuracy(link_probs, link_labels, self.dataset, 'link')
                print(f"test_score {test_score:7.4f}")
        
        return train_losses, train_scores, val_losses, val_scores


def train(task, dataset, 
          act_fn, num_layers, 
          add_self_loops, pair_norm, drop_edge, 
          test, device, conv_type="GCNConv"):

    # node classification task
    if task == "node classification":
        model = NodeClassification(device=device, dataset=dataset, path=f"./data/{dataset}/")
        if dataset == "ppi":
            lr = 2e-1
        else:
            lr = 1e-3
        train_losses, train_scores, val_losses, val_scores = model.train(patience=500, epochs=1000, lr=lr, hidden_features=64, num_layers=num_layers,
                    add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
                    act_fn=act_fn, conv_type=conv_type)
    # link prediction task
    elif task == "link prediction":
        model = LinkPrediction(device=device, dataset=dataset, path=f"./data/{dataset}/")
        train_losses, train_scores, val_losses, val_scores = model.train(patience=500, epochs=1000, lr=1e-3, hidden_features=64, num_layers=num_layers,
                    add_self_loops=add_self_loops, pair_norm=pair_norm, drop_edge=drop_edge, test=test,
                    act_fn=act_fn)
    else:
        raise NotImplementedError
    
    return train_losses, train_scores, val_losses, val_scores


def draw(train_losses, train_scores, val_losses, val_scores, task, dataset):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.legend()
    plt.savefig(f"./result/{task}_{dataset}_loss.png")
    plt.figure()
    plt.plot(train_scores, label="train score")
    plt.plot(val_scores, label="val score")
    plt.legend()
    plt.savefig(f"./result/{task}_{dataset}_score.png")


if __name__ == "__main__":
    
    device="cuda:1"
    
    # node classification task
    # cora
    train_losses, train_scores, val_losses, val_scores = train(task="node classification", dataset="cora",
            act_fn='prelu', num_layers=2,
            add_self_loops=True, pair_norm=False, drop_edge=False,
            test=True, device=device)
    draw(train_losses, train_scores, val_losses, val_scores, task="node", dataset="cora")
    # citeseer
    train_losses, train_scores, val_losses, val_scores = train(task="node classification", dataset="citeseer",
            act_fn='prelu', num_layers=1,
            add_self_loops=True, pair_norm=False, drop_edge=True,
            test=True, device=device)
    draw(train_losses, train_scores, val_losses, val_scores, task="node", dataset="citeseer")
    # ppi
    train_losses, train_scores, val_losses, val_scores = train(task="node classification", dataset="ppi",
            act_fn='prelu', num_layers=2,
            add_self_loops=True, pair_norm=False, drop_edge=False,
            test=True, device=device, conv_type="SAGEConv")
    draw(train_losses, train_scores, val_losses, val_scores, task="node", dataset="ppi")
    
    # link prediction task
    # cora
    train_losses, train_scores, val_losses, val_scores = train(task="link prediction", dataset="cora",
            act_fn='prelu', num_layers=1,
            add_self_loops=True, pair_norm=False, drop_edge=False,
            test=True, device=device)
    draw(train_losses, train_scores, val_losses, val_scores, task="link", dataset="cora")
    # citeseer
    train_losses, train_scores, val_losses, val_scores = train(task="link prediction", dataset="citeseer",
            act_fn='prelu', num_layers=1,
            add_self_loops=True, pair_norm=False, drop_edge=False,
            test=True, device=device)
    draw(train_losses, train_scores, val_losses, val_scores, task="link", dataset="citeseer")
    # ppi
    train_losses, train_scores, val_losses, val_scores = train(task="link prediction", dataset="ppi",
            act_fn='prelu', num_layers=1,
            add_self_loops=True, pair_norm=True, drop_edge=False,
            test=True, device=device)
    draw(train_losses, train_scores, val_losses, val_scores, task="link", dataset="ppi")