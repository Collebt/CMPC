import torch   
import matplotlib.pyplot as plt

path = 'results/pics/loc_err/'

files = ['loc_err_coarse_woProj_woWD_woGNN',
         'loc_err_coarse_wProj',
         'loc_err_coarse_wd_same',
         'loc_err_coarse_gnn_same',
         'loc_err_refine_gnn_same']

max_meter = 1000
for f in files:
    err = torch.load(path + f) * 100


    plt.plot(torch.linspace(0, max_meter, 20), err)
plt.xlabel('Threshold (m)')
plt.ylabel('Accuracy (%)')
plt.legend(['X-modality', 'AlignGAN', 'VIGOR', 'CMPC-coarse', 'CMPC-fine'])
plt.savefig(path + 'loc_err')
plt.close()
