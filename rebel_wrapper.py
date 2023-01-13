import os

from models.wrapper import ModelWrapper

import os
import json
from models.rebel.src.rebel_main import call_rebel
import shutil

TRAIN_ARGS_LIST = ["seed","device_id","epochs","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","lr","save_model","train_batch_size","eval_batch_size","lr_warmup","weight_decay","max_grad_norm","neg_relation_count","neg_entity_count","size_embedding","prop_drop","save_optimizer"]
EVAL_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]
PREDICT_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]


class RebelWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path):
        # data conf
        self.exp_cfgs.model_args.add("types_file",self.exp_cfgs.model_args.types_path,overwrite=True)
        self.exp_cfgs.model_args.add("test_file",self.exp_cfgs.model_args.test_path,overwrite=True)
        self.exp_cfgs.model_args.add("train_file",train_path,overwrite=True)
        self.exp_cfgs.model_args.add("validation_file",valid_path,overwrite=True)
        self.exp_cfgs.model_args.add("log_path",output_path,overwrite=True)
        # with open(os.path.join(self.exp_cfgs.sim_args.cwd,'models','rebel','conf','data',self.exp_cfgs.sim_args.dataset+'_data.yaml'),'w') as outfile:
        #     json.dump(self.exp_cfgs.model_args.data.configs, outfile, ensure_ascii=False)
        
        # model conf
        self.exp_cfgs.model_args.add("model_name_or_path",model_path,overwrite=True)
        self.exp_cfgs.model_args.add("config_name",model_path,overwrite=True)
        self.exp_cfgs.model_args.add("tokenizer_name",model_path,overwrite=True)
        # with open(os.path.join(self.exp_cfgs.sim_args.cwd,'models','rebel','conf','model','rebel_model.yaml'),'w') as outfile:
        #     json.dump(self.exp_cfgs.model_args.model.configs, outfile, ensure_ascii=False)
        
        # self.exp_cfgs.model_args.train.add("seed",self.exp_cfgs.model_args.seed,overwrite=True) 
        # # train conf
        # with open(os.path.join(self.exp_cfgs.sim_args.cwd,'models','rebel','conf','train',self.exp_cfgs.sim_args.dataset+'_train.yaml'),'w') as outfile:
        #     json.dump(self.exp_cfgs.model_args.train.configs, outfile, ensure_ascii=False)

        for data_path in ['train','valid','test']:
            data = json.load(open(self.exp_cfgs.model_args.configs[f'{data_path}_path']))
            for i,sentence in enumerate(data):
                data[i]['orig_id'] = i
            with open(self.exp_cfgs.model_args.configs[f'{data_path}_path'],'w') as out_file:
               json.dump(data,out_file)
        call_rebel(self.exp_cfgs.model_args)

        return True

    def eval(self, model_path, dataset_path, output_path, data_label='test', save_embeddings = False,  Temp_rel = 1.0, Temp_ent = 1.0):

        # data conf
        self.exp_cfgs.model_args.model.add("test_file",dataset_path)
        self.exp_cfgs.model_args.model.add("log_path",output_path)
        with open(os.path.join(self.cwd,'models','rebel','conf','data',self.exp_cfgs.dataset+'_data.yaml')) as outfile:
            json.dump(self.exp_cfgs.model_args.data, outfile, ensure_ascii=False)

        # model conf
        self.exp_cfgs.model_args.model.add("model_name_or_path",model_path)
        self.exp_cfgs.model_args.model.add("config_name",model_path)
        self.exp_cfgs.model_args.model.add("tokenizer_name",model_path)
        with open(os.path.join(self.cwd,'models','rebel','conf','model','rebel_model.yaml')) as outfile:
            json.dump(self.exp_cfgs.model_args.model, outfile, ensure_ascii=False)

        self.exp_cfgs.model_args.train.add("seed",self.exp_cfgs.model_args.seed) 
        # train conf
        with open(os.path.join(self.cwd,'models','rebel','conf','train',self.exp_cfgs.dataset+'_train.yaml')) as outfile:
            json.dump(self.exp_cfgs.model_args.train, outfile, ensure_ascii=False)

        call_rebel()

    def calibrate():
        pass

    def predict():
        pass