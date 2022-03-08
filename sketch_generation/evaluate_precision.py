import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss
from eval_skrnn import load_pretrained_congen, load_pretrained_uncond
import torch

def evaluate_precision(data_type,model_name,cond_gen,GRU=False):
    # Get the data
    data_enc_train, data_dec_train , _ = get_data(data_type=data_type,part="train")
    data_enc_test, data_dec_test , _ = get_data(data_type=data_type,part="test")
    
    # Compute loss on train set
    if cond_gen:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type, model_name=model_name, batch_size=1000, GRU=GRU)
    else:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type, model_name=model_name, batch_size=1000, GRU=GRU)
    
  
    hidden_enc = hidden_dec = encoder.initHidden()
    selected_ints = np.random.randint(0,len(data_enc_train),1000)
    inp_enc = torch.tensor(np.array([data_enc_train[i] for i in selected_ints]), dtype=torch.float, device=device)
    inp_dec = torch.tensor(np.array([data_dec_train[i] for i in selected_ints]), dtype=torch.float, device=device)

    if cond_gen:
        z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)
    else:
        z = mu = sigma = torch.zeros(1000, latent_dim, device=device)
    
    gmm_params, _ = decoder(inp_dec, z, hidden_dec)
    loss_lr_train, loss_kl_train = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
    print("Lr on train set"+str(np.mean(loss_lr_train.detach().cpu().numpy())))
    print("LKL on train set"+str(np.mean(loss_kl_train.detach().cpu().numpy())))

    # Compute loss on test set
    if cond_gen:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type, model_name=model_name, batch_size=1000, GRU=GRU)
    else:
        encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type, model_name=model_name, batch_size=1000, GRU=GRU)
    hidden_enc = hidden_dec = encoder.initHidden()
    selected_ints = np.random.randint(0,len(data_enc_test),1000)
    inp_enc = torch.tensor(np.array([data_enc_test[i] for i in selected_ints]), dtype=torch.float, device=device)
    inp_dec = torch.tensor(np.array([data_dec_test[i] for i in selected_ints]), dtype=torch.float, device=device)
    
    if cond_gen:   
        z, hidden_dec, mu, sigma = encoder(inp_enc, hidden_enc)  
    else:
        z = mu = sigma = torch.zeros(1000, latent_dim, device=device)

    gmm_params, _ = decoder(inp_dec, z, hidden_dec)
    loss_lr_test, loss_kl_test = skrnn_loss(gmm_params, [mu,sigma], inp_dec[:,1:,], device=device)
    print("Lr on test set"+str(np.mean(loss_lr_test.detach().cpu().numpy())))
    print("LKL on test set"+str(np.mean(loss_kl_test.detach().cpu().numpy())))

evaluate_precision("owl_cat_airplane_bridge","owl_cat_airplane_bridge",True)
evaluate_precision("owl_cat_airplane_bridge","owl_cat_airplane_bridge_GRU",True,GRU=True)
evaluate_precision("owl_cat_airplane_bridge","owl_cat_airplane_bridge",False)
evaluate_precision("owl_cat_airplane_bridge","owl_cat_airplane_bridge_GRU",False,GRU=True)