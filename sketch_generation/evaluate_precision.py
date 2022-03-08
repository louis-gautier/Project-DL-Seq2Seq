import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss
from eval_skrnn import load_pretrained_congen, load_pretrained_uncond
import torch

def evaluate_precision(data_type,model_name,cond_gen):
    data_enc_train, data_dec_train , _ = get_data(data_type=data_type,part="train")
    data_enc_test, data_dec_test , _ = get_data(data_type=data_type,part="test")
    if cond_gen:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(model_name)
    else:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(model_name)
    # Compute loss on train set
    encoder.train(False)
    decoder.train(False)
    hidden_enc = hidden_dec = encoder.initHidden()
    inp_enc = torch.tensor(data_enc_train[:], dtype=torch.float, device=device)
    inp_dec = torch.tensor(data_dec_train[:], dtype=torch.float, device=device)
    
    if cond_gen:   
        z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)  
    else:
        z = mu = sigma = torch.zeros(data_dec_train.shape[0], latent_dim, device=device)

    gmm_params, _ = decoder(inp_dec, z, hidden_dec)
    loss_lr_train, loss_kl_train = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
    print("Lr on train set"+str(loss_lr_train))
    print("LKL on train set"+str(loss_kl_train))

    # Compute loss on test set
    hidden_enc = hidden_dec = encoder.initHidden()
    inp_enc = torch.tensor(data_enc_test[:], dtype=torch.float, device=device)
    inp_dec = torch.tensor(data_dec_test[:], dtype=torch.float, device=device)
    
    if cond_gen:   
        z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)  
    else:
        z = mu = sigma = torch.zeros(data_dec_train.shape[0], latent_dim, device=device)

    gmm_params, _ = decoder(inp_dec, z, hidden_dec)
    loss_lr_test, loss_kl_test = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
    print("Lr on test set"+str(loss_lr_test))
    print("LKL on test set"+str(loss_kl_test))