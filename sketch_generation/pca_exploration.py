# Exploring the latent space with PCA
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

class SketchPath(Path):
    def __init__(self, data, factor=.2, *args, **kwargs):
        print(data.shape)
        vertices = np.cumsum(data[::,:-1], axis=0) / factor
        print(vertices.shape)
        codes = np.roll(self.to_code(data[::,-1].astype(int)), 
                        shift=1)
        codes[0] = Path.MOVETO

        super(SketchPath, self).__init__(vertices, codes, *args, **kwargs)
        
    @staticmethod
    def to_code(cmd):
        # if cmd == 0, the code is LINETO
        # if cmd == 1, the code is MOVETO (which is LINETO - 1)
        return Path.LINETO - cmd


def PCA_plot(data_type, n_points, bi_mode):
    encoder, decoder, hid_dim, latent_dim, t_step, cond_gen, mode, device = load_pretrained_congen(data_type)
    data_enc, _ , _ = get_data(data_type=data_type, part="test")
    # Compute the representations of random images in the test set
    data_points_indices = np.random.choice(range(len(data_enc)),n_points,replace=False)
    data_enc=np.array(data_enc)
    data_points = data_enc[data_points_indices]
    sketches = data_points[:,:,[0,1,3]]
    z = torch.zeros((n_points,1,latent_dim), device=device)
    hidden_dec = [(torch.zeros((1,latent_dim),device=device),torch.zeros((1,latent_dim),device=device)) for i in range(n_points)]
    for i in range(n_points):
      hidden_enc = (torch.zeros(bi_mode, 1, hid_dim, device=device), torch.zeros(bi_mode, 1, hid_dim, device=device))
      z[i], hidden_dec[i], _ , _ = encoder(torch.tensor(data_points[i],device=device,dtype=torch.float).unsqueeze(0),hidden_enc)
    # Perform PCA and T-SNE on the resulting z vectors
    z = z.squeeze().detach().cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(z)
    z_pca = pca.transform(z)
    fig, ax = plt.subplots(figsize=(10, 10))
    ((pc1_min, pc2_min), (pc1_max, pc2_max)) = np.percentile(z_pca, q=[5, 95], axis=0)
    ax.set_xlim(pc1_min, pc1_max)
    ax.set_ylim(pc2_min, pc2_max)

    for i, sketch in enumerate(sketches):
        print(sketch.shape)
        sketch_path = SketchPath(sketch, factor=7e+1)
        sketch_path.vertices[::,1] *= -1
        sketch_path.vertices += z_pca[i]
        patch = patches.PathPatch(sketch_path, facecolor='none')
        ax.add_patch(patch)

    ax.set_xlabel('$pc_1$')
    ax.set_ylabel('$pc_2$')   
    plt.show()


PCA_plot("cat",100,2)