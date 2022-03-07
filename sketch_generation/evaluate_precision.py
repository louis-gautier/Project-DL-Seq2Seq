import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss
from eval_skrnn import load_pretrained_congen, load_pretrained_uncond

def evaluate_precision(data_type,model_name,cond_gen):
    data_enc_train, data_dec_train , max_seq_len_train = get_data(data_type=data_type,part="train")
    data_enc_test, data_dec_test ,  = get_data(data_type=data_type,part="test")
    if cond_gen:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(model_name)
        encoder.train(False) #equivalent of encoder.eval()
        decoder.train(False)
        
        loss_lr, loss_kl = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
    else:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(model_name)
        encoder.train(False) #equivalent of encoder.eval()
        decoder.train(False)