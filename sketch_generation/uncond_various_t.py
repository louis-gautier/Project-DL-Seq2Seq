import os
import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond
import torch

data_type = 'bridge'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tested_temperatures = np.linspace(0.2,0.9,10)
encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type)
for temp in tested_temperatures:
  os.mkdir('drawings/'+"{:.2f}".format(temp))
  for i in range(5):
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, cond_gen=cond_gen, device=device, bi_mode= mode, temperature=temp)
    draw_image(strokes,save=True,save_dir='drawings/'+"{:.2f}".format(temp)+'/')