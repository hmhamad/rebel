import os

from models.wrapper import ModelWrapper

import os
import json
from models.rebel.src.rebel_main import call_rebel
import optuna

TRAIN_ARGS_LIST = ["seed","device_id","epochs","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","lr","save_model","train_batch_size","eval_batch_size","lr_warmup","weight_decay","max_grad_norm","neg_relation_count","neg_entity_count","size_embedding","prop_drop","save_optimizer"]
EVAL_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]
PREDICT_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]


class RebelWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path, trial : optuna.trial.Trial = None, curriculum_learning = False, train_ner = True, ner_train_path = None):
        
        if trial and not curriculum_learning:
            self.exp_cfgs.model_args.edit('num_train_epochs',trial.suggest_int('num_train_epochs', 15, 40, step=5))
            self.exp_cfgs.model_args.edit('learning_rate',trial.suggest_categorical('lr', [5e-6, 7e-6, 1e-5, 3e-5, 5e-5, 7e-5]))
            self.exp_cfgs.model_args.edit('weight_decay',trial.suggest_float('re_weight_decay', 0.0, 0.1))
            self.exp_cfgs.model_args.edit('droput',trial.suggest_float('droput', 0.0, 0.5))
            self.exp_cfgs.model_args.edit('lr_scheduler',trial.suggest_categorical('lr_scheduler', ['linear', 'constant_w_warmup', 'inverse_square_root']))
            self.exp_cfgs.model_args.edit('warmup_steps',trial.suggest_int('warmup_steps', 0, 1000, step=100))
            self.exp_cfgs.model_args.edit('train_batch_size',trial.suggest_int('batch_size', 2, 4, step=2))
            
        self.exp_cfgs.model_args.edit('do_train',True) 
        self.exp_cfgs.model_args.edit('do_eval',True) 
        self.exp_cfgs.model_args.edit('do_predict',False) 

        self.exp_cfgs.model_args.edit("model_path",model_path)
        self.exp_cfgs.model_args.add("types_file",self.exp_cfgs.model_args.types_path)
        self.exp_cfgs.model_args.add("train_file",train_path)
        self.exp_cfgs.model_args.add("validation_file",valid_path)
        self.exp_cfgs.model_args.add("log_path",output_path)

        self.exp_cfgs.model_args.add("model_name_or_path",self.exp_cfgs.model_args.tokenizer_name)


        for dataset_path in [train_path,valid_path]:
            data = json.load(open(dataset_path))
            for i,sentence in enumerate(data):
                data[i]['orig_id'] = i
            with open(dataset_path,'w') as out_file:
               json.dump(data,out_file)
        
        eval_micro_f1  = call_rebel(self.exp_cfgs.model_args, trial=trial)

        return eval_micro_f1

    def eval(self, model_path, dataset_path, output_path, data_label='test', save_embeddings = False,  Temp_rel = 1.0, Temp_ent = 1.0):

        self.exp_cfgs.model_args.edit('do_train',False) 
        self.exp_cfgs.model_args.edit('do_eval',True) 
        self.exp_cfgs.model_args.edit('do_predict',False) 
        self.exp_cfgs.model_args.edit("model_path",model_path)

        # data conf
        self.exp_cfgs.model_args.add("types_file",self.exp_cfgs.model_args.types_path)
        self.exp_cfgs.model_args.add("test_file",dataset_path)
        self.exp_cfgs.model_args.add("log_path",output_path)
        self.exp_cfgs.model_args.add("data_label",data_label)

        # model conf
        self.exp_cfgs.model_args.add("model_name_or_path",self.exp_cfgs.model_args.tokenizer_name)

        data = json.load(open(dataset_path))
        for i,sentence in enumerate(data):
            data[i]['orig_id'] = i
        with open(dataset_path,'w') as out_file:
            json.dump(data,out_file)
        
        call_rebel(self.exp_cfgs.model_args)

        return True

    def predict(self, model_path, dataset_path, output_path, data_label='predict', save_embeddings = False,  Temp_rel = 1.0, Temp_ent = 1.0):

        self.exp_cfgs.model_args.edit('do_train',False) 
        self.exp_cfgs.model_args.edit('do_eval',False) 
        self.exp_cfgs.model_args.edit('do_predict',True) 
        self.exp_cfgs.model_args.edit("model_path",model_path)

        # data conf
        self.exp_cfgs.model_args.add("types_file",self.exp_cfgs.model_args.types_path)
        self.exp_cfgs.model_args.add("test_file",dataset_path)
        self.exp_cfgs.model_args.add("log_path",output_path)
        self.exp_cfgs.model_args.add("data_label",data_label)

        # model conf
        self.exp_cfgs.model_args.add("model_name_or_path",self.exp_cfgs.model_args.tokenizer_name)

        data = json.load(open(dataset_path))
        for i,sentence in enumerate(data):
            data[i]['orig_id'] = i
        with open(dataset_path,'w') as out_file:
            json.dump(data,out_file)
        
        call_rebel(self.exp_cfgs.model_args)

        return True

    def calibrate():
        pass