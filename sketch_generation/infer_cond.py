from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_congen
import torch
import numpy as np

data_type = 'cat' # can be kanji character or cat

data_enc, _ , _ = get_data(data_type=data_type) 
encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
enc_rnd = torch.tensor(data_enc[np.random.randint(0,len(data_enc))].unsqueeze(0),\
                                                                          dtype=torch.float, device =device)

strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, inp_enc=enc_rnd, 
                                               cond_gen=cond_gen, device=device, bi_mode=mode)
draw_image(strokes)