# Sketch drawing analogies in the latent space
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_load import get_data
from eval_skrnn import load_pretrained_congen, draw_image
from model import skrnn_sample

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.path import Path

def analogy(data_type, idx1, idx2, idx3, bi_mode):
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
    data_enc, _ , _ = get_data(data_type=data_type, part="test")
    # Compute the images at indexes idx1, idx2 and idx3
    data_points_indices = np.array([idx1,idx2,idx3])
    data_enc=np.array(data_enc)
    data_points = data_enc[data_points_indices]
    sketches = data_points[:,:,[0,1,3]]
    z = torch.zeros((3,1,latent_dim), device=device)
    hidden_dec = [(torch.zeros((1,latent_dim),device=device),torch.zeros((1,latent_dim),device=device)) for i in range(3)]
    for i in range(3):
      hidden_enc = (torch.zeros(bi_mode, 1, hid_dim, device=device), torch.zeros(bi_mode, 1, hid_dim, device=device))
      z[i], hidden_dec[i], _ , _ = encoder(torch.tensor(data_points[i],device=device,dtype=torch.float).unsqueeze(0),hidden_enc)
    z_res = z[0] + z[1] - z[2]
    hidden_dec_res = hidden_dec[0] + hidden_dec[1] - hidden_dec[2]
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, cond_gen=False, device=device, bi_mode= mode, temperature=0.3, initial_hidden_dec = hidden_dec_res, initial_z=z_res)
    draw_image(strokes)


analogy("cat",45,567,12,2)