import os
import numpy as np
from data_load import get_data
from model import encoder_skrnn, decoder_skrnn, skrnn_loss, skrnn_sample
from eval_skrnn import draw_image, load_pretrained_uncond
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drawing_completion(first_line_csv, data_type):
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_uncond(data_type)
    
    # Get the first line
    if first_line_csv is not None:
      fcsv = open(first_line_csv)
      line_points = np.loadtxt(fcsv, delimiter=",")
      line_points[:,:2]/=25
    else:
      data_enc, data_dec, max_seq_len = get_data(data_type=data_type,part="test")
      np.random.shuffle(data_enc)
      line_points = []
      for point in data_enc[0]:
        line_points.append(point)
        if point[3]==1:
          break
      line_points = np.array(line_points)

    strokes_first_line = np.concatenate((line_points[:,:2],np.zeros((line_points.shape[0],1))),axis=1)
    strokes_first_line[line_points.shape[0]-1,2]=1
    # Encode it using the decoder RNN
    decoder.train(False)
    hidden_dec = (torch.zeros(1, 1, hid_dim, device=device), torch.zeros(1, 1, hid_dim, device=device))
    z = torch.zeros(1, latent_dim, device=device)
    for i in range(line_points.shape[0]):
        prev_x = torch.tensor(line_points[i,:],dtype=torch.float, device=device)
        _, hidden_dec = decoder(prev_x.unsqueeze(0).unsqueeze(0), z, hidden_dec)

    # Generate the rest of the drawing
    strokes, mix_params = skrnn_sample(encoder, decoder, hid_dim, latent_dim, time_step=t_step, cond_gen=False, device=device, bi_mode= mode, temperature=0.3, initial_hidden_dec = hidden_dec, start=line_points[-1,:])
    strokes = np.concatenate((strokes_first_line,strokes),axis=0)
    colors = ["red"] + ["darkgreen" for i in range(int(np.sum(strokes[:,2])))]
    draw_image(strokes,save=False,save_dir='drawings/completion',color=colors)

drawing_completion(first_line_csv=None, data_type="cat")