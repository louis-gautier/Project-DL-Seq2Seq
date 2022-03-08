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

def analogy(data_type, data_type1, idx1, data_type2, idx2, data_type3, idx3, bi_mode):
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
    data_enc1, _ , _ = get_data(data_type=data_type1, part="train")
    data_enc2, _ , _ = get_data(data_type=data_type2, part="train")
    data_enc3, _ , _ = get_data(data_type=data_type3, part="train")
    # Compute the images at indexes idx1, idx2 and idx3
    img1 = data_enc1[idx1]
    img2 = data_enc2[idx2]
    img3 = data_enc3[idx3]
    data_encs = [img1,img2,img3]
    z = torch.zeros((3,1,latent_dim), device=device)
    hidden_dec = [(torch.zeros((1,latent_dim),device=device),torch.zeros((1,latent_dim),device=device)) for i in range(3)]
    for i in range(3):
      hidden_enc = (torch.zeros(bi_mode, 1, hid_dim, device=device), torch.zeros(bi_mode, 1, hid_dim, device=device))
      z[i], hidden_dec[i], _ , _ = encoder(torch.tensor(data_encs[i],device=device,dtype=torch.float).unsqueeze(0),hidden_enc)
    z_res = z[0] + z[1] - z[2]
    hidden_dec_res_0 = hidden_dec[0][0] + hidden_dec[1][0] - hidden_dec[2][0]
    hidden_dec_res_1 = hidden_dec[0][1] + hidden_dec[1][1] - hidden_dec[2][1]
    hidden_dec_res = (hidden_dec_res_0,hidden_dec_res_1)
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, cond_gen=False, device=device, bi_mode= mode, temperature=0.3, initial_hidden_dec = hidden_dec_res, initial_z=z_res)
    draw_image(strokes)


analogy("cat_owl","cat",60341,"owl",47704,"cat",51727,2)