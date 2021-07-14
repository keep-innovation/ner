import pandas as pd
import re
import codecs
from config import Config
import pickle

"""
按照标点符号切割的预处理数据
"""
config = Config()
data_dir = config.source_data_dir
print(data_dir)

# 原始数据集
train_df = pd.read_csv(config.source_data_dir + 'train.csv', encoding='utf-8')
test_df = pd.read_csv(config.source_data_dir + 'test.csv', encoding='utf-8')


def stop_words(x):
    x = re.sub(r'/\w* ', "", x)
    return x


train_df['text'] = train_df['text'].apply(stop_words)


def count_len(text):
    row_len = len(text)
    if row_len < 64:
        return 64
    elif row_len < 128:
        return 128
    elif row_len < 254:
        return 254
    elif row_len < 384:
        return 384
    elif row_len < 512:
        return 512
    else:
        return 1024


train_df['text_len'] = train_df['text'].apply(count_len)
test_df['text_len'] = test_df['text'].apply(count_len)
print(train_df['text_len'].append(test_df['text_len']).value_counts())


# 切分训练集，分成训练集和验证集，在这可以尝试五折切割
print('Train Set Size:', train_df.shape)
new_dev_df = train_df[2300:2800]
frames = [train_df[:500], train_df[500:2800]]

# new_dev_df = train_df[1800:2300]
# frames = [train_df[:1800], train_df[2300:2800]]
#
# new_dev_df = train_df[1300:1800]
# frames = [train_df[:1300], train_df[1800:2800]]
#
# new_dev_df = train_df[800:1300]
# frames = [train_df[:800], train_df[1300:2800]]
#
# new_dev_df = train_df[300:800]
# frames = [train_df[:300], train_df[800:2800]]


new_train_df = pd.concat(frames)  # 训练集
new_test_df = test_df[:]  # 测试集


print('训练集:', new_train_df.shape)
print('验证集:', new_dev_df.shape)
print('测试集:', new_test_df.shape)


def make_label(f_up, text, crop_entities, disease_entities, medicine_entities):
    text_changed = text
    for en in crop_entities:
        if en:
            text_changed = text.replace(en, 'Ё' + (len(en) - 1) * 'Ж')

    for en in disease_entities:
        if en:
            text_changed = text_changed.replace(en, 'ё' + (len(en) - 1) * 'з')

    for en in medicine_entities:
        if en:
            text_changed = text_changed.replace(en, 'б' + (len(en) - 1) * 'ш')

    for t1, t2 in zip(text, text_changed):
        if t2 == 'Ё':
            f_up.write('{0} {1}\n'.format(t1, 'B-crop'))
        elif t2 == 'Ж':
            f_up.write('{0} {1}\n'.format(t1, 'I-crop'))

        elif t2 == 'ё':
            f_up.write('{0} {1}\n'.format(t1, 'B-disease'))
        elif t2 == 'з':
            f_up.write('{0} {1}\n'.format(t1, 'I-disease'))

        elif t2 == 'б':
            f_up.write('{0} {1}\n'.format(t1, 'B-medicine'))
        elif t2 == 'ш':
            f_up.write('{0} {1}\n'.format(t1, 'I-medicine'))
        else:
            f_up.write('{0} {1}\n'.format(t1, 'O'))

    f_up.write('\n')


# 构造训练集、验证集与测试集
with codecs.open(data_dir + 'train.txt', 'w', encoding='utf-8') as up:
    for row in new_train_df.iloc[:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        n_crop = row.n_crop
        n_disease = row.n_disease
        n_medicine = row.n_medicine

        n_crop_list = n_crop.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")
        n_disease_list = n_disease.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")
        n_medicine_list = n_medicine.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")

        make_label(up, text_lbl, n_crop_list, n_disease_list, n_medicine_list)

with codecs.open(data_dir + 'dev.txt', 'w', encoding='utf-8') as up:
    for row in new_dev_df.iloc[:].itertuples():
        # print(row.unknownEntities)
        text_lbl = row.text
        n_crop = row.n_crop
        n_disease = row.n_disease
        n_medicine = row.n_medicine

        n_crop_list = n_crop.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")
        n_disease_list = n_disease.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")
        n_medicine_list = n_medicine.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")

        make_label(up, text_lbl, n_crop_list, n_disease_list, n_medicine_list)

with codecs.open(data_dir + 'test.txt', 'w', encoding='utf-8') as up:
    for row in new_test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')


def make_label(label):
    label = label.replace('[', '').replace(']', '').replace("'", '').replace(" ", '').split(",")
    return ";".join(label)


new_dev_df['n_crop'] = new_dev_df['n_crop'].apply(make_label)
new_dev_df['n_disease'] = new_dev_df['n_disease'].apply(make_label)
new_dev_df['n_medicine'] = new_dev_df['n_medicine'].apply(make_label)

n_crop_list = new_dev_df['n_crop'].tolist()
n_disease_list = new_dev_df['n_disease'].tolist()
n_medicine_list = new_dev_df['n_medicine'].tolist()

with open(data_dir + 'true_label.pickle', 'wb') as f:
    pickle.dump([n_crop_list, n_disease_list, n_medicine_list], f)

with open(data_dir + 'true_label.pickle', 'rb') as f:
    y_true_crop_label_list, y_true_disease_label_list, y_true_medicine_label_list = pickle.load(f)





