import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.rebel.src.pl_data_modules import BasePLDataModule
from models.rebel.src.pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from models.rebel.src.generate_samples import GenerateTextSamplesCallback
import json

def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)
    
    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>'], # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
#         "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    data_types = json.load(open(conf.types_file,'r'))
    special_tokens_list = []
    for ent_type in data_types['entities']:
        special_tokens_list.append('<' + ent_type.lower() + '>')
    tokenizer.add_tokens(special_tokens_list, special_tokens = True)


    # if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
    #     tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)

    wandb_logger = WandbLogger(project = conf.dataset_name.split('/')[-1].replace('.py', ''), name = conf.model_name_or_path.split('/')[-1])

    callbacks_store = []

    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience
            )
        )

    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            # monitor=None,
            dirpath=conf.log_path + '/best_model',
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode
        )
    )
    callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
    callbacks_store.append(LearningRateMonitor(logging_interval='step'))
    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        #max_epochs=conf.num_train_epochs,
        max_steps=conf.max_steps,
        precision=conf.precision,
        amp_level=conf.amp_level,
        #logger=TensorBoardLogger(save_dir=conf.log_path + '/best_model/'),
        logger = wandb_logger,
        resume_from_checkpoint=conf.checkpoint_path,
        limit_val_batches=conf.val_percent_check
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


def call_rebel(conf):
    conf = omegaconf.OmegaConf.create(conf.configs)
    train(conf)


# @hydra.main(config_path='../conf', config_name='root')
# def main(conf: omegaconf.DictConfig,mode="", model="",dataset="",exp=""):
#     train(conf)
