import numpy as np
from bert import tokenization
from tqdm import tqdm
from config import Config


def load_data(data_file_name):
    with open(data_file_name) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            # if len(contends) == 0 and words[-1] == '。':
            if len(contends) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
    return lines


def create_example(lines):
    examples = []
    for (index, line) in enumerate(lines):
        guid = "%s" % index
        text = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


def get_labels():
    return ["O", "B-crop", "I-crop", "B-disease", "I-disease", "B-medicine", 'I-medicine', "X",
            "[CLS]", "[SEP]"]


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class DataIterator:
    """
    数据迭代器
    """
    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False,):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        self.label_map = {}
        for index, label in enumerate(get_labels(), 1):
            self.label_map[label] = index
        print(len(get_labels()))
        print(self.num_records)

    def convert_single_example(self, example_idx):
        text_list = self.data[example_idx].text.split(' ')
        label_list = self.data[example_idx].label.split(' ')
        tokens = text_list  # 区分大小写
        labels = label_list

        if len(tokens) >= self.seq_length - 1:
            tokens = tokens[0:(self.seq_length - 2)]
            labels = labels[0:(self.seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[CLS]"])

        for index, token in enumerate(tokens):
            ntokens.append(self.tokenizer.tokenize(token.lower())[0])  # 全部转换成小写, 方便BERT词典
            segment_ids.append(0)
            label_ids.append(self.label_map[labels[index]])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])

        # print(tokens)
        # print(ntokens)
        # print(label_ids)
        # print()

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")
            tokens.append("**NULL**")

        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        assert len(label_ids) == self.seq_length
        assert len(tokens) == self.seq_length
        return input_ids, input_mask, segment_ids, label_ids, tokens

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_ids_list = []
        tokens_list = []

        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, label_ids, tokens = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            tokens_list.append(tokens)

            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break
        while len(input_ids_list) < self.batch_size:
            input_ids_list.append(input_ids_list[0])
            input_mask_list.append(input_mask_list[0])
            segment_ids_list.append(segment_ids_list[0])
            label_ids_list.append(label_ids_list[0])
            tokens_list.append(tokens_list[0])

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, self.seq_length, tokens_list


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    #vocab_file = '/home/xxd/BERT-model_parameter/chinese_L-12_H-768_A-12/vocab.txt'
    do_lower_case = True
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    # data_iter = DataIterator(config.batch_size, data_file= config.dir_with_mission + 'train.txt', use_bert=True,
    #                         seq_length=config.sequence_length, tokenizer=tokenizer)
    data_dir = '/home/xxd/my_NER/datas/example1'
    dev_iter = DataIterator(config.batch_size, data_file=data_dir + 'dev.txt', use_bert=True,
                            seq_length=config.sequence_length, tokenizer=bert_tokenizer, is_test=True)

    i = 0
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm(dev_iter):
        i += 1



