import numpy as np
import torch
import matplotlib.pyplot as plt

from data_load import get_data
from eval_skrnn import load_pretrained_congen, draw_image
from model import skrnn_sample

def interpolation(x1,x2,n_points,data_type,bi_mode):
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
    # Compute the latent representation for the two sketches
    hidden_enc = (torch.zeros(bi_mode, 1, hid_dim, device=device), torch.zeros(bi_mode, 1, hid_dim, device=device))
    z_1, hidden_dec_1, mu_1, sigma_1 = encoder(x1, hidden_enc)
    z_2, hidden_dec_2, mu_2, sigma_2 = encoder(x2, hidden_enc)
    
    alphas = np.linspace(0,1,n_points)
    z = torch.zeros((len(alphas),1,z_1.shape[1]),device=device)
    hidden_dec = [(torch.zeros((1,hidden_dec_1[0].shape[1]),device=device),torch.zeros((1,hidden_dec_1[1].shape[1]),device=device)) for i in range(len(alphas))]
    for i,alpha in enumerate(alphas):
      z[i] = alpha * z_1 + (1.0 - alpha) * z_2
      hidden_dec[i] = (alpha*hidden_dec_1[0] + (1.0-alpha)*hidden_dec_2[0], alpha*hidden_dec_1[1] + (1.0-alpha)*hidden_dec_2[1])
    
    
    # Decode all the sketches on the interpolation line
    for i in range(len(hidden_dec)):
      strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, random_state= 98, cond_gen=False, device=device, bi_mode= mode, initial_hidden_dec = hidden_dec[i], initial_z=z[i])
      draw_image(strokes,save=True,save_dir='drawings/')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_type1="cat"
data_type2="cat"
data_enc1, _ , _ = get_data(data_type=data_type1, part="train")
data_enc2, _ , _ = get_data(data_type=data_type2, part="train")
x1 = data_enc1[69741]
draw_image(x1[:,[0,1,3]])
x1 = torch.tensor(x1,device=device,dtype=torch.float).unsqueeze(0)
x2 = data_enc2[51727]
draw_image(x2[:,[0,1,3]])
x2 = torch.tensor(x2,device=device,dtype=torch.float).unsqueeze(0)
interpolation(x1,x2,5,"cat",2)