import pytorch_lightning as pl
from models import SimpleCNNv2Lightning
import pandas as pd
from evaluation import full_evaluation_df, track_full_evaluation
import torch
from functools import partial

# Define a random sampling function for PyTorch
def random_sampling(track, track_len):
    indices = torch.randint(0, track.size(1), (track_len,))
    return track[:, indices, :, :, :]

# Convert the model to use PyTorch
class Image2TrackModel(pl.LightningModule):
    def __init__(self, model, track_len=4):
        super(Image2TrackModel, self).__init__()
        self.track_len = track_len
        self.model = model
        self.random_sampling_func = partial(random_sampling, track_len=track_len)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.model(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.random_sampling_func(x)
        return x

def track_eval():
    #model = SimpleCNNv2Lightning.load_from_checkpoint("logs\lightning_logs\\version_65\checkpoints\epoch=419-step=70980.ckpt")
    model_long_term = SimpleCNNv2Lightning.load_from_checkpoint("logs\lightning_logs\\version_99\checkpoints\epoch=177-step=2670.ckpt")
    #track_model = Image2TrackModel(model_long_term, track_len=4)
    
    print(full_evaluation_df( model=model_long_term,n_distractors=10))
    
    
if __name__ == "__main__":
    print("Starting eval: ")
    track_eval()
