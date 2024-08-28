from models import SimpleCNNv2Lightning
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import H5Dataset
from lightning.pytorch import loggers as pl_loggers
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation import full_evaluation_df, track_full_evaluation,cmc_evaluation_df

batch_size = 256

def train_model():
  #  short_term_train = pd.read_csv("data/short_term_train.csv")
  	
    # train_dataset = H5Dataset(img_file="notebooks/short_term_train.h5")
    # val_dataset = H5Dataset(img_file="notebooks/short_term_val.h5")
    # train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size, num_workers=6,persistent_workers=True)
    # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6,persistent_workers=True)
    # model_lg = SimpleCNNv2Lightning(input_shape=(3,56,56),conv_blocks=3,latent_dim=128)
    # early_stopping = EarlyStopping('val_loss', patience=100)
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    # checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    # #model_lg.compile()
    # trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True)
    # trainer.fit(model=model_lg, train_dataloaders=train_loader,val_dataloaders=val_loader)
    # print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
    # print(checkpoint_callback.best_model_score) # and prints it score
    best_model = SimpleCNNv2Lightning.load_from_checkpoint("D:\ss24_seminarstatistic\logs\lightning_logs\\version_110\checkpoints\epoch=900-step=306340.ckpt")
    model=best_model
    n_distractors=10
    
    # tsh_df = pd.read_csv("data/test_same_hour.csv")
    # tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    # tsh_ranks = cmc_evaluation_df(model, tsh_df)
    # print(tsh_ranks)
        
    # tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    # tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    # tddsh_ranks = cmc_evaluation_df(model, tddsh_df)
 
        
    # tdd_df = pd.read_csv("data/test_different_day.csv")
    # tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    # tdd_ranks = cmc_evaluation_df(model, tdd_df)
  
        
    # result_dict = {
    #     "test_same_hour": tsh_ranks,
    #     "test_different_day_same_hour": tddsh_ranks,
    #     "test_different_day": tdd_ranks
    # }

    # result_df = pd.DataFrame(result_dict)
    # print(result_df)
    # result_df.to_csv("results_short_term.csv")
    
    best_model
    train_dataset = H5Dataset(img_file="train_data/long_term_train.h5")
    val_dataset = H5Dataset(img_file="train_data/long_term_val.h5")
    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=256, num_workers=6,persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=6,persistent_workers=True)
    early_stopping = EarlyStopping('val_loss', patience=100)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True,)
    trainer.fit(model=best_model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    best_model_finetuned = SimpleCNNv2Lightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    model=best_model_finetuned
    n_distractors=10
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df)
    print(tsh_ranks)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df)
    print(tddsh_ranks)

        
    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df)
    print(tdd_ranks)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
    print(result_df)
    result_df.to_csv("results_finetuned.csv")
    
def finetune_model(model_path):
    
    model = SimpleCNNv2Lightning.load_from_checkpoint(model_path)

    train_dataset = H5Dataset(img_file="train_data/short_term_train.h5")
    val_dataset = H5Dataset(img_file="train_data/short_term_train.h5")
    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=256, num_workers=6,persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=6,persistent_workers=True)
    early_stopping = EarlyStopping('val_loss', patience=100)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True,)
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    
    
    best_model_finetuned = SimpleCNNv2Lightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    model=best_model_finetuned
    n_distractors=10
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df)
    print(tsh_ranks)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df)
    print(tddsh_ranks)

        
    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df)
    print(tdd_ranks)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
    print(result_df)
    result_df.to_csv("results_finetuned.csv")


if __name__ == "__main__":
    print("Starting training: ")
    #train_model()
    finetune_model("D:\ss24_seminarstatistic\logs\lightning_logs\\version_106\checkpoints\epoch=146-step=2205.ckpt")
