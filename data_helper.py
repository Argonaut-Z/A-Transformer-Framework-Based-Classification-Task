from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import torch
import re
from tqdm import tqdm

def my_tokenizer(s):
    tokenizer = get_tokenizer('basic_english')
    return tokenizer(s)

def clean_str(string):
    string = re.sub("[^A-Za-z0-9\-\?\!\.\,]", " ", string).lower()
    return string


def build_vocab(tokenizer, file_path, min_freq, specials=None):
    """
    根据给定的tokenizer和对应参数返回一个vocab类
    Args:
        tokenizer: 分词器
        file_path: 文本的路径
        min_freq: 最小词频，去掉小于min_freq的词
        specials: 特殊的字符，如<pad>，<unk>等
    """
    if specials is None:
        specials = ['<unk>', '<pad>']
    counter = Counter()
    with open(file_path, encoding='utf8') as f:
        for string_ in tqdm(f):
            string_ = string_.strip().split('","')[-1][:-1]  # 新闻描述
            counter.update(tokenizer(clean_str(string_)))
    return vocab(counter, min_freq=min_freq, specials=specials)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个列表中的序列进行填充（padding），使得它们具有相同的长度
    Args:
        sequences(list[Tensor]): 输入的序列列表，每个元素是一个形状为(seq_len, ...)的张量
        batch_first(bool):  是否将 batch_size 作为第一个维度输出(batch_size, max_len, ...)
        max_len: 指定填充后的最大长度
            - 如果为None，则以输入 batch 中最长序列为基准进行填充
            - 如果指定，max_len 必须 >= batch 中最长序列的长度，否则仍以最长序列为基准
        padding_value: 填充值，默认为0
    Returns:
        Tensor: 填充后的张量，形状为(max_len, batch_size, ...) or (batch_size, max_len, ...)
    """
    max_size = sequences[0].size()  # (seq_len, ...)
    trailing_dims = max_size[1:]    # 提取除长度之后的其余维度
    
    # 动态计算 batch 中最长序列的长度
    max_len_in_batch = max([s.size(0) for s in sequences])  
    if max_len is None:
        max_len = max_len_in_batch
    else:
        max_len = max(max_len, max_len_in_batch)

    # 根据 batch_first 确定输出张量的维度
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims    # 形状为 (batch_size, max_len, ...)
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims    # 形状为 (max_len, batch_size, ...)
    
    # 创建填充后的张量，并初始化为填充值
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    
    # 遍历每个输入序列，将其拷贝到填充张量中
    for i, tensor in enumerate(sequences):
        seq_len = tensor.size(0)    # 获取当前序列的长度
        # 根据 batch_first 确定填充的方式
        if batch_first:
            out_tensor[i, :seq_len, ...] = tensor
        else:
            out_tensor[:seq_len, i, ...] = tensor
    
    # 返回填充后的张量
    return out_tensor


class LoadSentenceClassificationDataset():
    def __init__(self, train_file_path=None,    # 训练集路径
                 tokenizer=None,
                 batch_size=20,
                 min_freq=1,    # 最小词频，去掉小于min_freq的词
                 max_sen_len='same'):   # 最大句子长度，默认设置其长度为整个数据集中最长样本的长度
        # max_sen_len = None时，表示按每个batch中最长的样本长度进行padding
        # 根据训练语料建立字典
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.specials = ['<unk>', '<pad>']
        self.vocab = build_vocab(self.tokenizer,
                                file_path=train_file_path,
                                min_freq=self.min_freq,
                                specials=self.specials)
        self.PAD_IDX = self.vocab['<pad>']
        self.UNK_IDX = self.vocab['<unk>']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        
    def data_process(self, filepath):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        Args:
            filepath (_type_): 数据集路径
        """
        
        raw_iter = open(filepath, encoding='utf8').readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip().split('","')
            s, l = line[-1][:-1], line[0][1:]
            s = clean_str(s)
            tensor_ = torch.tensor([self.vocab[token] if token in self.vocab else self.UNK_IDX for token in
                                    self.tokenizer(s)], dtype=torch.long)
            l = torch.tensor(int(l) - 1, dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path, test_file_path):
        train_data, max_sen_len = self.data_process(train_file_path)    # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        test_data, _ = self.data_process(test_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,     # 构造DataLoader
                                shuffle=True, collate_fn=self.generate_batch) 
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, test_iter
    
    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len = self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


if __name__ == '__main__':
    path = './data/ag_news_csv/train.csv'
    data_loader = LoadSentenceClassificationDataset(train_file_path=path,
                                                    tokenizer=my_tokenizer,
                                                    max_sen_len=None)
    data, max_len = data_loader.data_process(path)
    train_iter, test_iter = data_loader.load_train_val_test_data(path, path)
    for sample, label in train_iter:
        print(sample.shape, label)