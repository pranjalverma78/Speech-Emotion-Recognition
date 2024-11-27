import warnings
from tqdm import tqdm
import os
from pathlib import Path
import math
import pandas as pd

import wandb

import librosa

import numpy as np

import torchmetrics
import torchmetrics.classification

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(self, file_path, seq_len, d_model, augment=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.file_path = file_path  # Single file path
        self.augment = augment
        self.num_augments = 1  # Default number of augmentations (original only)

        if self.augment:
            self.num_augments = 4  # Include augmented versions

        self.class_map = {
            "female_neutral": 0, "female_surprise": 1, "female_disgust": 2, 
            "female_fear": 3, "female_sad": 4, "female_happy": 5, "female_angry": 6, 
            "male_neutral": 7, "male_sad": 8, "male_fear": 9, "male_happy": 10, 
            "male_disgust": 11, "male_angry": 12, "male_surprise": 13
        }
        self.label_map = {v: k for k, v in self.class_map.items()}

    @staticmethod
    def extract_audio_features(signal, sample_rate=44100):
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sample_rate)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfe = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
        rms_energy = librosa.feature.rms(y=signal)
        all_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs, mfe, chroma, rms_energy), axis=0)
        return all_features

    def __len__(self):
        return self.num_augments  # Length corresponds to the number of augmentations

    def __getitem__(self, idx):
        # augment_idx = idx  # Single file, so index directly maps to augmentation

        # Load the audio file
        # if not os.path.exists(self.file_path):
        #     raise FileNotFoundError(f"File not found: {self.file_path}")
        signal, sr = librosa.load(self.file_path)

        # Generate augmentations
        # signals = [signal]  # Original signal
        # if self.augment:
        #     signals.extend([
        #         self.add_noise(signal),
        #         self.stretch_process(signal),
        #         self.pitch_process(signal, sr)
        #     ])

        # Select the corresponding augmentation
        # sig = signals[augment_idx]
        # sr = 44100
        inp = AudioDataset.extract_audio_features(signal, sr)

        # Padding
        padding = self.seq_len - inp.shape[1]
        if padding < 0:
            inp = inp[:, :self.seq_len]

        inp = torch.cat([
            torch.tensor(inp, dtype=torch.float32),
            torch.zeros((inp.shape[0], max(0, padding)), dtype=torch.float32)
        ], 1)

        # Define the label (example: "happy" as a hardcoded emotion for now)
        # emotion = "happy"
        # label = torch.zeros(len(self.class_map))
        # label[self.class_map[emotion]] = 1

        sample = {
            "input": inp.T
        }
        return sample

    @staticmethod
    def add_noise(signal, noise_level=0.005):
        noise = np.random.randn(len(signal))
        return signal + noise_level * noise

    @staticmethod
    def stretch_process(signal, rate=1.2):
        return librosa.effects.time_stretch(signal, rate=rate)

    @staticmethod
    def pitch_process(signal, sr, n_steps=4):
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

    
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    #print("Done")
    return mask == 0


def get_label(vector):
    # class_map = {"anger" : 0, "sadness": 1, "fear": 2, "happy": 3, "neutral": 4, "surprise": 5, "sarcastic": 6, "disgust": 7}
    label_map = {0: "anger", 1: "sadness", 2: "fear", 3: "happy", 4: "neutral", 5: "surprise", 6: "sarcastic", 7: "disgust"}
    #print("Done")
    return label_map[np.argmax(vector)]



#----------------------------------------------------------------

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        
        # linear model
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.linear_2 = nn.Linear(d_ff, d_ff)
        self.linear_3 = nn.Linear(d_ff, d_ff)
        self.linear_4 = nn.Linear(d_ff, d_ff)        
        self.linear_5 = nn.Linear(d_ff, d_model) 
        
        # cnn model
        # self.conv_1 = nn.Conv2d(1, 8, kernel_size=(2, 2), padding=1, stride=1)
        # self.conv_2 = nn.Conv2d(8, 64, kernel_size=(2, 2), padding=1, stride=1)
        # self.conv_3 = nn.Conv2d(64, 128, kernel_size=(2, 2), padding=1, stride=1)
        # self.conv_4 = nn.Conv2d(128, 64, kernel_size=(2, 2), padding=1, stride=1)
        # self.conv_5 = nn.Conv2d(64, 8, kernel_size=(2, 2), padding=1, stride=1)
        # self.conv_6 = nn.Conv2d(8, 1, kernel_size=(2, 2), padding=1, stride=1)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        x = self.linear_1(x)
        x = self.linear_2(self.dropout_1(torch.relu(x)))
        x = self.linear_3(self.dropout_2(torch.relu(x)))
        x = self.linear_4(self.dropout_3(torch.relu(x)))
        x = self.linear_5(self.dropout_4(torch.relu(x)))
        
        # convo forward 
        # (batch, seq_len, d_model) --> (batch, 1, seq_len, d_model)
        # x = x.unsqeeze(1)
        
        return x

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        sin_div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        cos_div_term = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * sin_div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * cos_div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask = None , dropout: nn.Dropout = None):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask = None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask = None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, seq_len: int, d_model: int, d_ff: int, num_of_labels: int, dropout: float) -> None:
        super().__init__()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        
        # self.proj = nn.Linear(d_ff, num_of_labels)
        
        self.proj = nn.Linear(4864, num_of_labels)
        # dense network
#         self.linear_1 = nn.Linear(d_model, d_ff)
#         self.linear_2 = nn.Linear(d_ff, d_ff)
#         self.linear_3 = nn.Linear(d_ff, d_model)
#         self.linear_4 = nn.Linear(d_model*seq_len, d_ff)
#         self.linear_5 = nn.Linear(d_ff, d_ff)
#         self.linear_6 = nn.Linear(d_ff, d_ff)
        
        
        # cnn model
        # self.conv_1 = nn.Conv2d(1, 8, padding_mode='replicate')
        # self.conv_2 = nn.Conv2d(8, 64, padding_mode='replicate')
        # self.conv_3 = nn.Conv2d(64, 8, padding_mode='replicate')
        # self.conv_4 = nn.Conv2d(8, 1, padding_mode='replicate')
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(8)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(64)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batchnorm_3 = nn.BatchNorm2d(128)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=d_ff, kernel_size=(75, 4), stride=1, padding=0)
        self.batchnorm_4 = nn.BatchNorm2d(d_ff)


    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, num_of_labels)
#         x = self.linear_1(x)
#         x = self.linear_2(self.dropout_1(torch.relu(x)))
#         x = self.linear_3(self.dropout_2(torch.relu(x)))
#         x = torch.flatten(x,1)
#         x = self.linear_4(self.dropout_3(torch.relu(x)))
#         x = self.linear_5(self.dropout_4(torch.relu(x)))
#         x = self.linear_6(self.dropout_5(torch.relu(x)))
#         x = self.proj(self.dropout_6(torch.relu(x)))
        
        # convo forward 
        # (batch, seq_len, d_model) --> (batch, 1, seq_len, d_model)
        x = torch.unsqueeze(x, 1)
        x = self.dropout_1(self.pool_1(torch.relu(self.batchnorm_1(self.conv1(x)))))
        x = self.dropout_2(self.pool_2(torch.relu(self.batchnorm_2(self.conv2(x)))))
        x = self.dropout_3(self.pool_3(torch.relu(self.batchnorm_3(self.conv3(x)))))
        x = self.dropout_4(torch.relu(self.batchnorm_4(self.conv4(x))))
        
        x = torch.flatten(x, 1)
        x = self.proj(x)
        
        return x



    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, inp_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.inp_pos = inp_pos
        self.projection_layer = projection_layer

    def encode(self, inp, inp_mask = None):
        # (batch, seq_len, d_model)
        inp = self.inp_pos(inp)
        return self.encoder(inp, inp_mask)
    
    def project(self, x):
        # (batch, num_of_label)
        return self.projection_layer(x)
    


def build_transformer(seq_len: int, num_of_labels: int, d_model: int=180, N: int=4, h: int=3, dropout: float=0.1, d_ff: int=256) -> Transformer:
    # Create the embedding layers

    # Create the positional encoding layers
    inp_pos = PositionalEncoding(d_model, seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(seq_len, d_model, d_ff, num_of_labels, dropout)
    
    # Create the transformer
    transformer = Transformer(encoder, inp_pos, projection_layer)
    transformer.to(device)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
    print("Done")


def get_config():
    return {
        "dataset_root_location": "/kaggle/working/",
        "df_location": "Data_path.csv",
        "num_of_labels": 14,
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 0.1,
        "seq_len": 600,
        "d_model": 180,
        "datasource": "Global_dataset_ravdess_tess",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel"
    }

##----------------------------------------------------------------

config = get_config()
# Path to the saved model
saved_model_path = "./tmodel_best_model_0.pt"

# Initialize the model architecture
# Ensure get_model(config) initializes the same architecture as used during training
model = build_transformer(config["seq_len"], config['num_of_labels'], d_model=config['d_model']).to(device)

# ds_raw = get_audio_location_list(config['df_location'])
    
# Keep 90% for training, 10% for validation
# train_ds_raw_size = int(0.9 * len(ds_raw))
# val_ds_raw_size = len(ds_raw) - train_ds_raw_size

# train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_raw_size, val_ds_raw_size])

# val_ds = AudioDataset(val_ds_raw, config["seq_len"], config["d_model"], augment=False)

# val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# Load the saved weights
checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Set the model to evaluation mode
model.eval()

import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'uploads'

# Ensure the directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Iterate over all files in the uploads folder
# for audio_file in os.listdir(UPLOAD_FOLDER):
#     file_path = os.path.join(UPLOAD_FOLDER, audio_file)
    
#     # Ensure it's a valid file (optional: check extensions)
#     if os.path.isfile(file_path) and audio_file.endswith(('.wav', '.mp3')):
#         print(f"Processing file: {audio_file}")

#         # Initialize AudioDataset with the file
#         val_ds = AudioDataset(file_path, config["seq_len"], config["d_model"], augment=False)

# val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# Example for predicting on a single batch from validation dataset
# with torch.no_grad():
#     for batch in val_dataloader:
#         inp = batch["input"].to(device)  # Input tensor
#         # label = batch["label"].to(device)  # Ground truth labels

#         # Forward pass through the model
#         encoder_output = model.encode(inp)  # Encoding
#         proj_output = model.project(encoder_output)  # Projecting to label space

#         # Get predicted class
#         predictions = torch.argmax(proj_output, dim=1)
#         print("Predictions:", predictions.cpu().numpy())
#         # print("True Labels:", torch.argmax(label, dim=1).cpu().numpy())
#         break  # Remove this to process the entire validation dataset
emotion_map = {
    "female_neutral": 0, "female_surprise": 1, "female_disgust": 2,
    "female_fear": 3, "female_sad": 4, "female_happy": 5, "female_angry": 6,
    "male_neutral": 7, "male_sad": 8, "male_fear": 9, "male_happy": 10,
    "male_disgust": 11, "male_angry": 12, "male_surprise": 13
}

reverse_emotion_map = {v: k for k, v in emotion_map.items()}

def emotion_predict():
    # val_ds = AudioDataset("./", config["seq_len"], config["d_model"], augment=False)

    for audio_file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, audio_file)
        
        # Ensure it's a valid file (optional: check extensions)
        if os.path.isfile(file_path) and audio_file.endswith(('.wav', '.mp3')):
            print(f"Processing file: {audio_file}")

            # Initialize AudioDataset with the file
            val_ds = AudioDataset(file_path, config["seq_len"], config["d_model"], augment=False)

    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # Example for predicting on a single batch from validation dataset
    with torch.no_grad():
        for batch in val_dataloader:
            inp = batch["input"].to(device)  # Input tensor
            # label = batch["label"].to(device)  # Ground truth labels

            # Forward pass through the model
            encoder_output = model.encode(inp)  # Encoding
            proj_output = model.project(encoder_output)  # Projecting to label space

            # Get predicted class
            predictions = torch.argmax(proj_output, dim=1)
            # print("Predictions:", predictions.cpu().numpy())
            emotion = reverse_emotion_map.get(predictions.cpu().numpy()[0], "Unknown")

            for file in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, file))
            # print("True Labels:", torch.argmax(label, dim=1).cpu().numpy())
            break  # Remove this to process the entire validation dataset

    
    return emotion

print(emotion_predict())