class Config:
    
    def __init__(self):

        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 10 + 1  # 实体的种类

        self.batch_size = 16
        self.decay_rate = 0.25
        self.decay_step = int(2800 / self.batch_size)
        self.num_checkpoints = 20 * 3

        self.train_epoch = 8
        self.sequence_length = 256  # BERT的输入MAX_LEN

        self.learning_rate = 1e-4 * 2  # 下接结构的学习率
        self.embed_learning_rate = 5e-5  # BERT的微调学习率

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file = '/data/wangzhili/Finance_entity_recog/ensemble/source_file/'
        self.ensemble_result_file = '/data/wangzhili/Finance_entity_recog/ensemble/result_file/'

        # 存放的模型名称，用以预测
        self.checkpoint_path = "/home/wangzhili/data/agri/model/runs_1/1594137453/model_0.9875_0.9923-1152"
        self.checkpoint_path = '/home/wangzhili/data/agri/model/runs_3/1594137774/model_0.9846_0.9890-576'
        self.checkpoint_path = '/home/wangzhili/data/agri/model/runs_2/1594269299/model_1.0000_1.0000-3500'
        self.checkpoint_path = "/home/wangzhili/data/agri/model/runs_2/1594269299/model_0.9984_0.9992-1050"
        self.checkpoint_path = '/home/wangzhili/data/agri/model/runs_2/1594269299/model_1.0000_1.0000-3500'
        self.checkpoint_path = "/home/wangzhili/data/agri/model/runs_0/1594286066/model_0.9980_0.9984-875"
        self.checkpoint_path = "/home/wangzhili/data/agri/model/runs_0/1594286066/model_0.9992_0.9996-1050"
        self.checkpoint_path = "/home/wangzhili/data/agri/model/runs_0/1594286066/model_0.9944_0.9956-700"

        self.model_dir = '/home/wangzhili/data/agri/model/'  # 模型存放地址
        self.source_data_dir = '/home/wangzhili/data/agri/'  # 原始数据集

        # self.model_type = 'idcnn'  # 使用idcnn
        self.model_type = 'bilstm'  # 使用bilstm
        # self.model_type = 'gru'  # 使用gru
        self.lstm_dim = 256
        self.gru_num = 256
        self.dropout = 0.5
        self.use_origin_bert = False  # True:使用原生bert, False:使用动态融合bert

        self.import_name = 'nazhe'
        # self.import_name = 'roberta'
        # self.import_name = 'electra'

        # BERT预训练模型的存放地址
        if self.import_name == 'electra':
            self.bert_file = "/home/wangzhili/pretrained_model/chinese_electra_base_L-12_H-768_A-12/electra_base"
            self.vocab_file = "/home/wangzhili/pretrained_model/chinese_electra_base_L-12_H-768_A-12/vocab.txt"
            self.bert_config_file = "/home/wangzhili/pretrained_model/NEZHA-Base-WWM/bert_config.json"
        elif self.import_name == 'nazhe':

            self.bert_file = "/home/wangzhili/pretrained_model/NEZHA-Base/model.ckpt-900000"
            self.bert_config_file = "/home/wangzhili/pretrained_model/NEZHA-Base/bert_config.json"
            self.vocab_file = "/home/wangzhili/pretrained_model/NEZHA-Base/vocab.txt"

            self.bert_file = "/home/wangzhili/pretrained_model/NEZHA-Base-WWM/model.ckpt-691689"
            self.bert_config_file = "/home/wangzhili/pretrained_model/NEZHA-Base-WWM/bert_config.json"
            self.vocab_file = "/home/wangzhili/pretrained_model/NEZHA-Base-WWM/vocab.txt"

        else:
            self.bert_file = '/home/wangzhili/pretrained_model/roberta_zh_l12/bert_model.ckpt'
            self.bert_config_file = '/home/wangzhili/pretrained_model/roberta_zh_l12/bert_config.json'
            self.vocab_file = '/home/wangzhili/pretrained_model/roberta_zh_l12/vocab.txt'



