from train_fine_tune import decode, get_text_and_label
from config import Config
import tensorflow as tf
import os
import json
import numpy as np
if Config().import_name == 'electra':
    from tf_utils.electra.model import tokenization  # electra
elif Config().import_name == 'nazhe':
    from tf_utils.nezha import tokenization  # nezha
else:
    from bert import tokenization  # roberta
import tqdm
from utils import DataIterator
import pandas as pd

result_data_dir = Config().source_data_dir
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _input_relation = graph.get_operation_by_name("input_relation").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]
            _token_ids_type = graph.get_operation_by_name('token_ids_type').outputs[0]

            used = tf.sign(tf.abs(_input_x))
            length = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(length, tf.int32)
            logits = graph.get_operation_by_name('project/pred_logits').outputs[0]

            trans = graph.get_operation_by_name('transitions').outputs[0]

            def run_predict(feed_dict):
                return session.run([logits, lengths, trans], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _token_ids_type, _input_x_len, _input_mask, _input_relation, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    ldct_list = []
    logits_list = []
    lengths_list = []
    trans_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(test_iter):
        # 对每一个batch的数据进行预测
        logits, lengths, trans = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, segment_ids_list, seq_length, input_mask_list, label_ids_list, 1, False))
                 )
        )

        logits_list.append(logits)
        lengths_list.append(lengths)
        trans_list.append(trans)
        pred = decode(logits, lengths, trans)
        y_pred_list.append(pred)
        ldct_list.append(tokens_list)

    ldct_list_tokens = np.concatenate(ldct_list)

    y_pred_crop_label_list, y_pred_disease_label_list, y_pred_medicine_label_list = get_text_and_label(
        ldct_list_tokens, y_pred_list)

    # y_pred_crop_label_list = [list(set(item.split(';'))) for item in y_pred_crop_label_list]
    # y_pred_disease_label_list = [list(set(item.split(';'))) for item in y_pred_disease_label_list]
    # y_pred_medicine_label_list = [list(set(item.split(';'))) for item in y_pred_medicine_label_list]

    y_pred_crop_label_list = [item.split(';') for item in y_pred_crop_label_list]
    y_pred_disease_label_list = [item.split(';') for item in y_pred_disease_label_list]
    y_pred_medicine_label_list = [item.split(';') for item in y_pred_medicine_label_list]

    test_df = pd.read_csv(result_data_dir + 'test.csv', encoding='utf8')
    id_list = test_df['id'].tolist()
    print(len(id_list))
    print(len(y_pred_medicine_label_list))
    result_pd = pd.DataFrame({
        'id': id_list,
        'n_crop': y_pred_crop_label_list[:len(id_list)],
        'n_disease': y_pred_disease_label_list[:len(id_list)],
        'n_medicine': y_pred_medicine_label_list[:len(id_list)]
    })
    model_name = config.checkpoint_path.split('/')[-1]
    print(result_data_dir + model_name.replace('.', '_').replace('-', '_') + "_result.csv")

    def process_na_list(row):
        if not row[0]:
            return '[]'
        else:
            return row
    result_pd['n_crop'] = result_pd['n_crop'].apply(process_na_list)
    result_pd['n_disease'] = result_pd['n_disease'].apply(process_na_list)
    result_pd['n_medicine'] = result_pd['n_medicine'].apply(process_na_list)

    result_pd.to_csv(result_data_dir + model_name.replace('.', '_').replace('-', '_') + "_result.csv", encoding='utf8', index=False)


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'test.txt', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
