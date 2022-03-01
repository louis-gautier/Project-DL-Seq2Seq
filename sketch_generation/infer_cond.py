from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_congen
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_type = 'cat' # can be kanji character or cat

data_enc, _ , _ = get_data(data_type=data_type)
# 70000*129(max_length)*5
encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)

conditioned_on = data_enc[np.random.randint(0,len(data_enc))]
draw_image(conditioned_on)
enc_rnd = torch.tensor(conditioned_on,dtype=torch.float, device =device).unsqueeze(0) # convert to batch*seq*data_dim

strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, inp_enc=enc_rnd, cond_gen=cond_gen, device=device, bi_mode=mode)
draw_image(strokes)