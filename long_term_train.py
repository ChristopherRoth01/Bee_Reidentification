from models import SimpleCNNv2Lightning,TrackModel,SimpleCNNv2LightningLarge
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
from data_provider import load_torch_track_dataset,get_track_dataset,get_dataset
from evaluation import full_evaluation_df, track_full_evaluation,cmc_evaluation_df

def train_large_model():
    train_dataset = H5Dataset(img_file="notebooks/long_term_train_large.h5")
    val_dataset = H5Dataset(img_file="notebooks/long_term_val_large.h5")
    train_loader = DataLoader(train_dataset, shuffle=False,batch_size=256, num_workers=6,persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=6,persistent_workers=True)
    model_lg = SimpleCNNv2Lightning(input_shape=(3,56,56),conv_blocks=3,latent_dim=128)
    early_stopping = EarlyStopping('val_loss', patience=100)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    #model_lg.compile()
    trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True,)
    trainer.fit(model=model_lg, train_dataloaders=train_loader,val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
    print(checkpoint_callback.best_model_score) # and prints it score
    return checkpoint_callback.best_model_path
    
    
def train_model():
    train_dataset = H5Dataset(img_file="notebooks/long_term_train_unshuffled.h5")
    val_dataset = H5Dataset(img_file="notebooks/long_term_val_unshuffled.h5")
    train_loader = DataLoader(train_dataset, shuffle=False,batch_size=256, num_workers=6,persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=6,persistent_workers=True)
    model_lg = SimpleCNNv2Lightning(input_shape=(3,56,56),conv_blocks=3,latent_dim=128)
    model = TrackModel(name="TrackModel", backbone=model_lg)
    early_stopping = EarlyStopping('val_loss', patience=100)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    #model_lg.compile()
    trainer = L.Trainer(callbacks=[early_stopping,checkpoint_callback],max_epochs=1000,logger = tb_logger,default_root_dir="models",enable_checkpointing=True,)
    trainer.fit(model=model_lg, train_dataloaders=train_loader,val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
    print(checkpoint_callback.best_model_score) # and prints it score
    return checkpoint_callback.best_model_path

    
    

def eval_model(modelpath,image_size=(128,128),white=False,small_scale=False,results_csv="results.csv"):
    best_model = SimpleCNNv2Lightning.load_from_checkpoint(modelpath)
    model=best_model
    n_distractors=50
    
    tsh_df = pd.read_csv("data/test_same_hour.csv")
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df,image_size=image_size,white=white,small_scale = small_scale)
    print(tsh_ranks)
        
    tddsh_df = pd.read_csv("data/test_different_day_same_hour.csv")
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df,image_size=image_size,white=white,small_scale = small_scale)
    print(tddsh_ranks)

        
    tdd_df = pd.read_csv("data/test_different_day.csv")
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df,image_size=image_size,white=white,small_scale = small_scale)
    print(tdd_ranks)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
    print(result_df)
    result_df.to_csv(results_csv,index=False)
    
if __name__ == "__main__":
    print("Starting training: ")
    #best_normal_model = train_model() #versuon 129
    #best_cropped_model = train_large_model() #version 130
    #best_model = "D:\ss24_seminarstatistic\logs\lightning_logs\\version_125\checkpoints\epoch=99-step=1500.ckpt"
    best_long_term_model = "D:\ss24_seminarstatistic\logs\lightning_logs\\version_106\checkpoints\epoch=146-step=2205.ckpt"
    #best_finetuned_model ="D:\ss24_seminarstatistic\logs\lightning_logs\\version_122\checkpoints\epoch=683-step=232560.ckpt"
    #best_normal_model = "D:\ss24_seminarstatistic\logs\lightning_logs\\version_129\checkpoints\epoch=159-step=2400.ckpt"
    #best_cropped_model = "D:\ss24_seminarstatistic\logs\lightning_logs\\version_130\checkpoints\epoch=95-step=1440.ckpt"
    
    #unshuffled_model = "D:\ss24_seminarstatistic\logs\lightning_logs\\version_133\checkpoints\epoch=181-step=2730.ckpt"
    eval_model(best_long_term_model,image_size=(128,128), white= False,small_scale=True,results_csv="results_longterm_n_distractors_50")
    #eval_model(best_cropped_model,image_size=(256,256), white= False,small_scale=False)

    #train_model()