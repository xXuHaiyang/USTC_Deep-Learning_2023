import os
import torch
import tarfile
import random
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from gensim.models import Word2Vec, KeyedVectors
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import TensorDataset, random_split, RandomSampler, SequentialSampler

# 数据路径
root_data_path = './data/'
compress_data_file = os.path.join(root_data_path, 'aclImdb_v1.tar.gz')

# 探索训练数据量（需要调整的参数）
train_rate = 0.5  # train data 中使用的比例, 可以尝试 0.25, 0.5, 0.75, 1 等
split_rate = 0.2  # 验证集比例

# 训练词向量设置
vector_size = 200  # 词向量维度大小
window = 10  # 使用前 window 个词预测后面一个词
embedding_file_path = os.path.join(root_data_path, f'word2vec.wordvectors_{vector_size}_{window}_{train_rate}')  # 词向量保存路径

# embedding 向量的长度，与词向量长度一样。RNN 网络中需要添加一层 Embedding 层
embed_size = vector_size

# 需要使用的词的最少在训练集出现的次数，去掉不常见的词
min_freq = 3

# 标签的种类
num_class = 2

best_model_dir = './results/rnn-util50/'  # 保存模型时使用

# Bert 参数
pretrained_bert_model_name = 'bert-base-uncased'  # 预训练 Bert 版本
sentence_max_length = 512  # Sentence 中最长字符个数，Bert 对于输入的最大长度有限制

# 训练参数
batch_size = 16  # 微调 Bert 时，batch_size 不能设置太大，否则会显存不够

class RawData:
    """原始数据静态类"""
    __type = None
    __data = []

    __vocab = None

    @staticmethod
    def read_imdb(type_):  # 读取原始数据, 可以读取 训练集或者测试集
        if RawData.__type != type_:
            RawData.__type = type_

            if not os.path.exists(os.path.join(root_data_path, "aclImdb")):  # 解压数据
                print(">>> start decompression")
                with tarfile.open(compress_data_file, 'r') as f:
                    f.extractall(root_data_path)
                print(">>> end decompression")

            RawData.__data = []
            print(">>> start read data")
            for label in ['pos', 'neg']:
                folder_name = os.path.join(root_data_path, 'aclImdb', type_, label)
                file_list = os.listdir(folder_name)
                if type_ == 'train':
                    file_list = file_list[:int(len(file_list) * train_rate)]  # 训练集使用部分数据
                for file in tqdm(file_list):
                    with open(os.path.join(folder_name, file), 'rb') as f:
                        review = f.read().decode('utf-8').replace('\n', '').lower()
                        RawData.__data.append([review, 1 if label == 'pos' else 0])
            print(">>> end read data")
            random.seed(2023)
            random.shuffle(RawData.__data)  # 这里就把数据打乱，这样后面读取数据就不打乱了

        return RawData.__data

    @staticmethod
    def get_vocab():  # 利用 word2vec 的方法训练 token embedding
        # 只使用训练集去建立 vocab , 这样保证训练集和测试集 token 对应的 id 是一样的，还防止数据泄露
        if RawData.__vocab is None:
            tokenizer = get_tokenizer('basic_english')
            data = RawData.read_imdb(type_="train")
            counter = Counter()
            sen_tokens = []
            for line, label in data:  # 把 sentence 变成 work list
                tokens = tokenizer(line)
                counter.update(tokens)
                sen_tokens.append(tokens)

            # 读取词向量，如果没有则训练得到
            if os.path.exists(embedding_file_path):
                word_vector = KeyedVectors.load(embedding_file_path)
            else:
                print(">>> start train word embedding")
                model = Word2Vec(sentences=sen_tokens, vector_size=vector_size, window=window, min_count=1, workers=4)
                print(">>> end train word embedding")
                word_vector = model.wv
                # 增加 <unk> 和 <pad> 词向量，用于填充
                zeros = np.zeros(vector_size)
                word_vector.add_vector('<unk>', zeros)
                word_vector.add_vector('<pad>', zeros)
                word_vector.save(embedding_file_path)

            # 创建词到下标的转换
            sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
            # min_freq，去掉出现次数小于 min_freq 的词
            # specials, 特殊字符 '<unk>', '<pad>'
            _vocab = vocab(ordered_dict=ordered_dict, min_freq=min_freq, specials=['<unk>', '<pad>'], special_first=True)  
            _vocab.set_default_index(_vocab['<unk>'])  # 这个使得对于没有出现的词，会输出 '<unk>' 的下标
            keys = _vocab.get_itos()  # all the word sorted by index
            embeddings = word_vector[keys]
            _vocab.embeddings = embeddings
            RawData.__vocab = _vocab
        return RawData.__vocab


class DatasetClassRNN(Dataset):
    def __init__(self, type_):
        if (type_ == "train") or (type_ == "val"):
            all_text = RawData.read_imdb(type_="train")
            train_len = int(len(all_text) * (1 - split_rate))
            if type_ == "train":
                self.raw_text = all_text[:train_len]
            else:
                self.raw_text = all_text[train_len:]
        else:
            self.raw_text = RawData.read_imdb(type_="test")
        self.vocab = RawData.get_vocab()  # 保证使用的是同一个 embedding matrix
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab_size = len(self.vocab)

    def text_pipeline(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]

    def __getitem__(self, item):
        text, label = self.raw_text[item]
        text_processed = self.text_pipeline(text)
        return text_processed, label

    def __len__(self):
        return len(self.raw_text)


class MySampler(Sampler):
    """RNN训练时，相似长度的句子放在一个 Batch 中
    """
    def __init__(self, dataset, batch_size):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # 这里按照长度降序排序，这样符合后面 rnn 的 pack_padded_sequence 输入
        self.indices = np.argsort([len(sentence[0]) for sentence in dataset])[::-1]
        self.count = int(len(dataset) / self.batch_size)

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count


def collate_fn(batch_data, pad=0):
    # label_list, text_list, offsets = [], [], []
    # 把每个句子填补成一样长度
    offsets = [len(sentence[0]) for sentence in batch_data]
    max_len = max(offsets)
    label_list = [sentence[1] for sentence in batch_data]
    text_list = [sentence[0] + [pad] * (max_len - len(sentence[0])) for sentence in batch_data]
    text_tensor = torch.LongTensor(text_list)
    label_tensor = torch.LongTensor(label_list)
    offsets_tensor = torch.LongTensor(offsets)
    return text_tensor, label_tensor, offsets_tensor


def data_loader_RNN(dataset, sort_length=True):
    if sort_length:
        my_sampler = MySampler(dataset=dataset, batch_size=batch_size)
        loader = DataLoader(dataset=dataset, batch_sampler=my_sampler, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    return loader


import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from datetime import datetime
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertForSequenceClassification


class RNN(nn.Module):
    def __init__(self, model_type, vocab_size, embed_dim, num_class, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.model_type = model_type

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)
        vocab = RawData.get_vocab()  # 得到训练好的词向量

        self.embedding.weight.data.copy_(torch.tensor(vocab.embeddings, dtype=torch.float32))  # 利用训练好的词向量初始化 embedding 层

        if model_type == "RNN":
            self.rnn = nn.RNN(input_size=embed_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              nonlinearity='tanh',
                              batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        elif model_type == 'Transform':
            pass  # transform 模型比较复杂，感兴趣的同学可以自己实现

        self.fc = nn.Linear(hidden_size, num_class)  # 线性模型分类

    def forward(self, inputs, offsets):
        embeds = self.embedding(inputs)
        offsets = offsets.to("cpu")
        packed_x = pack_padded_sequence(input=embeds, lengths=offsets, batch_first=True)
        x = self.rnn(packed_x)
        x = x[1]
        if self.model_type == "LSTM":
            x = x[0]
        x = x.view(-1, x.shape[1], x.shape[2])
        x = x[-1]
        x = self.fc(x)

        return x

def criterion(output, target):
    cri = nn.CrossEntropyLoss()
    loss = cri(output, target)
    return loss


def calc_acc(output, labels):
    output = output.cpu().numpy()
    output = np.argmax(output, axis=1)
    labels = labels.cpu().numpy()
    labels = labels.reshape(len(labels))

    acc = np.sum(output == labels) / len(labels)
    return acc


class Lab3_RNN(object):

    def __init__(self, lr=0.001, epochs=20, device='cuda',
                 hidden_size=100, sort_length=True, patience=5,
                 fix_embedding=False,
                 num_layers=1,
                 model_type="RNN"):
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.sort_length = sort_length
        self.patience = patience
        self.fix_embedding = fix_embedding
        self.num_layers = num_layers
        self.model_type = model_type

        self.net = None
        self.optimizer = None
        self.vocab_size = None
        self.embed_size = None
        self.num_class = None

    def train(self):
        save_path = os.path.join(best_model_dir, f"{self.model_type}_best_model.pth")
        # 加载数据
        train_dataset = DatasetClassRNN(type_="train")
        train_loader = data_loader_RNN(dataset=train_dataset, sort_length=self.sort_length)

        val_dataset = DatasetClassRNN(type_="val")
        val_loader = data_loader_RNN(dataset=val_dataset, sort_length=self.sort_length)

        self.vocab_size = train_dataset.vocab_size
        self.embed_size = embed_size
        self.num_class = num_class
        # 定义网络
        self.net = RNN(self.model_type,
                       self.vocab_size,
                       self.embed_size,
                       self.num_class,
                       self.hidden_size,
                       self.num_layers).to(self.device)

        if self.fix_embedding:
            for k, v in self.net.named_parameters():
                if k == "embedding.weight":
                    v.requires_grad = False

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                    lr=self.lr)

        total_params = sum([param.nelement() for param in self.net.parameters() if param.requires_grad])
        print(f">>> model type: {self.model_type}, train rate: {train_rate}")
        print(f">>> total parameters: {total_params}")
        patience = 0
        best_val_acc = 0.
        best_train_acc = 0.
        for epoch in range(self.epochs):
            t1 = datetime.now()
            for data in tqdm(train_loader):
                text_tensor, label_tensor, offsets_tensor = data
                text_tensor = text_tensor.to(self.device)
                label_tensor = label_tensor.to(self.device)
                offsets_tensor = offsets_tensor.to(self.device)
                predict_label = self.net(text_tensor, offsets_tensor)
                self.optimizer.zero_grad()
                loss = criterion(predict_label, label_tensor)
                loss.backward()
                self.optimizer.step()

            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            with torch.no_grad():
                for data in tqdm(train_loader):
                # for data in train_loader:
                    text_tensor, label_tensor, offsets_tensor = data
                    text_tensor = text_tensor.to(self.device)
                    label_tensor = label_tensor.to(self.device)
                    offsets_tensor = offsets_tensor.to(self.device)
                    predict_label = self.net(text_tensor, offsets_tensor)
                    loss = criterion(predict_label, label_tensor)
                    acc_ = calc_acc(predict_label, label_tensor)

                    train_acc += acc_
                    train_loss += loss.item()

                train_loss = train_loss / len(train_loader)
                train_acc = train_acc / len(train_loader)

                for data in tqdm(val_loader):
                # for data in val_loader:
                    text_tensor, label_tensor, offsets_tensor = data
                    text_tensor = text_tensor.to(self.device)
                    label_tensor = label_tensor.to(self.device)
                    offsets_tensor = offsets_tensor.to(self.device)
                    predict_label = self.net(text_tensor, offsets_tensor)
                    loss = criterion(predict_label, label_tensor)
                    acc_ = calc_acc(predict_label, label_tensor)

                    val_acc += acc_
                    val_loss += loss.item()

                val_loss = val_loss / len(val_loader)
                val_acc = val_acc / len(val_loader)
            print(f"epoch: {epoch}, train loss: {train_loss:.6f}, train acc: {train_acc:.6f}, "
                  f"val loss: {val_loss:.6f}, val acc: {val_acc:.6f},  time: {datetime.now() - t1}")
            t1 = datetime.now()
            if val_acc < best_val_acc:
                patience = patience + 1
                if patience > self.patience:
                    break
            else:
                patience = 0
                best_val_acc = val_acc
                best_train_acc = train_acc
                if not os.path.exists(best_model_dir):
                    os.mkdir(best_model_dir)
                print(f"save best weights to {save_path}")
                torch.save(self.net.state_dict(), save_path)
        return best_train_acc, best_val_acc

    def test(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(best_model_dir, f"{self.model_type}_best_model.pth")
        test_dataset = DatasetClassRNN(type_="test")
        test_loader = data_loader_RNN(dataset=test_dataset, sort_length=self.sort_length)

        test_loss = 0.0
        test_acc = 0.0
        # init model
        # 定义网络
        if self.net is None:
            self.vocab_size = test_dataset.vocab_size
            self.embed_size = embed_size
            self.num_class = num_class
            self.net = RNN(self.model_type,
                           self.vocab_size,
                           self.embed_size,
                           self.num_class,
                           self.hidden_size,
                           self.num_layers).to(self.device)
        # load best weights
        self.net.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            for data in tqdm(test_loader):
                text_tensor, label_tensor, offsets_tensor = data
                text_tensor = text_tensor.to(self.device)
                label_tensor = label_tensor.to(self.device)
                offsets_tensor = offsets_tensor.to(self.device)
                predict_label = self.net(text_tensor, offsets_tensor)
                loss = criterion(predict_label, label_tensor)
                acc_ = calc_acc(predict_label, label_tensor)

                test_loss += loss.item()
                test_acc += acc_

            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / (len(test_loader))
            print(f"test loss: {test_loss}, test acc: {test_acc}")
        return test_acc


# RNN 训练
model_rnn = Lab3_RNN(lr=1e-3, epochs=10, device="cuda:1", hidden_size=100, sort_length=True,
                     patience=3, fix_embedding=True, num_layers=3, model_type='RNN')
model_rnn.train()



