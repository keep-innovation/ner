import os
import time
import json
import tqdm
from config import Config
from model import Model
from utils import DataIterator
from optimization import create_optimizer
import numpy as np

if Config().import_name == 'electra':
    from tf_utils.electra.model import tokenization  # electra
elif Config().import_name == 'nazhe':
    from tf_utils.nezha import tokenization  # nezha
else:
    from bert import tokenization  # roberta

from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf
import pickle
# import tensorflow.compat.v1 as tf

""" 
2020年7月8日 decay_step=2500, lr=5e-5 dym roberta 第五折 lstm_dim = 256 max-len = 258
gru: /home/wangzhili/data/agri/model/runs_1/1594137453/model_0.9875_0.9923-1152 0.919
idcnn:  /home/wangzhili/data/agri/model/runs_2/1594137511/model_0.9855_0.9915-1008 0.92396	
bilstm: /home/wangzhili/data/agri/model/runs_3/1594137774/model_0.9846_0.9890-576  0.91482

2020年7月8日 decay_step=144, lr=5e-5 ori roberta 第五折 lstm_dim = 256 max-len = 258
bilstm: /home/wangzhili/data/agri/model/runs_1/1594143059/model_0.9899_0.9919-1296 0.92731
gru: /home/wangzhili/data/agri/model/runs_2/1594143193/model_0.9891_0.9927-1440 0.9155
idcnn: /home/wangzhili/data/agri/model/runs_3/1594143273/model_0.9899_0.9931-2880 0.9263


2020年7月8日 decay_step=288, lr=5e-5 ori roberta 第五折 lstm_dim = 256 max-len=512
bilstm: /home/wangzhili/data/agri/model/runs_2/1594176003/model_0.9895_0.9923-4896 0.921
idcnn: /home/wangzhili/data/agri/model/runs_1/1594176017/model_0.9879_0.9919-4896
gru: /home/wangzhili/data/agri/model/runs_3/1594175967/model_0.9903_0.9947-4896

2020年7月8日 decay_step=144, decay_rate=0.25 lr=1e-4 *2 dym nazha 第五折 lstm_dim = 256 max-len=256  
bilstm: /home/wangzhili/data/agri/model/runs_1/1594192335
2020年7月8日 decay_step=144, decay_rate=0.25 lr=5e-5 dym nazha 第五折 lstm_dim = 128 max-len=256
bilstm: /home/wangzhili/data/agri/model/runs_2/1594196661/model_0.9688_0.9695-1728

2020年7月8日 decay_step=144, decay_rate=0.25 lr=1e-4 *2 ori nazha_wwm 第五折 lstm_dim = 256 max-len=256
bilstm: /home/wangzhili/data/agri/model/runs_2/1594217372/model_0.9916_0.9940-1584  0.92766

2020年7月8日 decay_step=144, decay_rate=0.25 lr=1e-4 * 2 dym nazha 全量 lstm_dim = 256 max-len=256   
bilstm: /home/wangzhili/data/agri/model/runs_2/1594269299/model_1.0000_1.0000-3500   

bilstm: /home/wangzhili/data/agri/model/runs_0/1594286066
"""
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

result_data_dir = Config().source_data_dir
print('GPU ID: ', str(gpu_id))
print('Model Type: ', Config().model_type)
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('bilstm embedding ', Config().lstm_dim)
print('use original bert ', Config().use_origin_bert)
print('use electra model', Config().import_name)


def train(train_iter, test_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # 读取模型结构图

            # 超参数设置
            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)  # 下接结构的学习率

            all_variables = graph.get_collection('trainable_variables')

            if config.import_name == 'electra':
                word2vec_var_list = [x for x in all_variables if 'electra' in x.name]  # BERT的参数
                normal_var_list = [x for x in all_variables if 'electra' not in x.name]  # 下接结构的参数
            else:
                word2vec_var_list = [x for x in all_variables if 'bert' in x.name]  # BERT的参数
                normal_var_list = [x for x in all_variables if 'bert' not in x.name]  # 下接结构的参数

            print('bert train variable num: {}'.format(len(word2vec_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if word2vec_var_list:  # 对BERT微调
                print('word2vec trainable!!')
                word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                    model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05), use_tpu=False, variable_list=word2vec_var_list
                )

                train_op = tf.group(normal_op, word2vec_op)  # 组装BERT与下接结构参数
            else:
                train_op = normal_op

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(config.model_dir, "runs_" + str(gpu_id), timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(config.__dict__, file)
            print("Writing to {}\n".format(out_dir))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            """
            笔者在config.py设置了200个epoch，当然不能全部跑完，一般我们跑了3~4个epoch的时候，便可以停止了。
            这么设置的目的是多保存几个模型，再通过check_F1.py来查看每次训练得到的最高F1模型，取最优模型进行预测。
            """
            for i in range(config.train_epoch):  # 训练
                for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
                        train_iter):

                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.input_relation: label_ids_list,
                        model.input_x_len: seq_length,
                        model.segment_ids: segment_ids_list,

                        model.keep_prob: config.keep_prob,
                        model.is_training: True,
                    }

                    _, step, _, loss, lr = session.run(
                        fetches=[train_op,
                                 global_step,
                                 embed_step,
                                 model.loss,
                                 learning_rate
                                 ],
                        feed_dict=feed_dict)

                    if cum_step % 100 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                P, R = set_test(model, test_iter, session)
                F = 2 * P * R / (P + R)
                print('dev set :  epoch_{}, step_{},precision_{},recall_{}'.format(i, cum_step, P, R))
                if F > 0.988:
                    saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(P, R)),
                               global_step=step)
                with open(os.path.join(out_dir, 'model_record.txt'), 'a') as fw:
                    fw.write('loss {:.4f}, cum_step_{},p_{},r_{},f1_{}\n'.format(loss, cum_step, P, R, F))


def get_label(input_tokens_list, y_list, begin, end):
    y_label_list = []  # 标签
    for i, input_tokens in enumerate(input_tokens_list):
        ys = y_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []
        for index, num in enumerate(ys):

            if num == begin and len(temp) == 0:
                temp.append(input_tokens[index])
            elif num == end and len(temp) > 0:
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                label_list.append("".join(temp))
                temp = []

        y_label_list.append(";".join(label_list))
    return y_label_list


def get_text_and_label(input_tokens_list, y_list):
    """
    还原每一条数据的文本的标签
    :return:
    """
    y_list = np.concatenate(y_list)

    y_crop_label_list = get_label(input_tokens_list, y_list, 2, 3)
    y_disease_label_list = get_label(input_tokens_list, y_list, 4, 5)
    y_medicine_label_list = get_label(input_tokens_list, y_list, 6, 7)

    return y_crop_label_list, y_disease_label_list, y_medicine_label_list


def decode(logits, lengths, matrix):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels use viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * Config().relation_num + [0]])
    # print('length:', lengths)
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])

    return paths


def set_operation(row):
    content_list = row.split(';')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def get_TP_FP_FN(y_pred_label_list, y_true_label_list):
    TP = 0
    FP = 0
    FN = 0
    for i, y_true_label in enumerate(y_true_label_list):
        # y_pred_label = set(y_pred_label_list[i].split(';'))
        # y_true_label = set(y_true_label.split(';'))

        y_pred_label = y_pred_label_list[i].split(';')
        y_true_label = y_true_label.split(';')
        y_true_label_change = copy.deepcopy(y_true_label)
        current_TP = 0
        for y_pred in y_pred_label:
            if y_pred in y_true_label_change:
                y_true_label_change.remove(y_pred)
                current_TP += 1
            else:
                FP += 1
        TP += current_TP
        FN += (len(y_true_label) - current_TP)

    return TP, FP, FN


def set_test(model, test_iter, session):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    ldct_list_tokens = []
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
            test_iter):
        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.input_relation: label_ids_list,
            model.input_mask: input_mask_list,
            model.segment_ids: segment_ids_list,

            model.keep_prob: 1,
            model.is_training: False,
        }

        lengths, logits, trans = session.run(
            fetches=[model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )

        predict = decode(logits, lengths, trans)
        y_pred_list.append(predict)
        y_true_list.append(label_ids_list)
        ldct_list_tokens.append(tokens_list)

    ldct_list_tokens = np.concatenate(ldct_list_tokens)
    ldct_list_text = []
    for tokens in ldct_list_tokens:
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 获取验证集文本及其标签
    y_pred_crop_label_list, y_pred_disease_label_list, y_pred_medicine_label_list = get_text_and_label(
        ldct_list_tokens, y_pred_list)
    with open(result_data_dir + 'true_label.pickle', 'rb') as f:
        y_true_crop_label_list, y_true_disease_label_list, y_true_medicine_label_list = pickle.load(f)

    a_TP, a_FP, a_FN = get_TP_FP_FN(y_pred_crop_label_list, y_true_crop_label_list)
    b_TP, b_FP, b_FN = get_TP_FP_FN(y_pred_disease_label_list, y_true_disease_label_list)
    c_TP, c_FP, c_FN = get_TP_FP_FN(y_pred_medicine_label_list, y_true_medicine_label_list)
    TP = a_TP + b_TP + c_TP + 1e-10
    FP = a_FP + b_FP + c_FP + 1e-10
    FN = a_FN + b_FN + c_FN + 1e-10

    P = TP / (TP + FP)
    R = TP / (TP + FN)

    f1 = 2 * P * R / (P + R)

    print('precision: {}, recall {}, f1 {}'.format(P, R, f1))

    return P, R


if __name__ == '__main__':
    code_config = Config()
    vocab_file = code_config.vocab_file  # 通用词典
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_it = DataIterator(code_config.batch_size, data_file=result_data_dir + 'train.txt',
                            use_bert=code_config.use_bert, tokenizer=tokenizer, seq_length=code_config.sequence_length)

    dev_it = DataIterator(code_config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=code_config.use_bert,
                          tokenizer=tokenizer, seq_length=code_config.sequence_length, is_test=True)

    train(train_it, dev_it, code_config)
