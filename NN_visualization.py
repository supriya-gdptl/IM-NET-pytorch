import torch
import torch.onnx
import os
import numpy as np

from modelAE import im_network


input_size = 64 #input voxel grid size

ef_dim = 32
gf_dim = 128
z_dim = 256
point_dim = 3

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

model_dir = "all_vox256_img_ae_64"
checkpoint_dir = "checkpoint"

#build model
im_network = im_network(ef_dim, gf_dim, z_dim, point_dim)
im_network.to(device)

# checkpoint loading
checkpoint_path = os.path.join(checkpoint_dir, model_dir)
checkpoint_txt = os.path.join(checkpoint_path, "checkpoint")
fin = open(checkpoint_txt)
model_dir = fin.readline().strip()
fin.close()

im_network.load_state_dict(torch.load(model_dir))

im_network.eval()
# data_voxels[t:t+1].astype(np.float32)
dummy_voxel = torch.randn(-1, 1, 64, 64, 64, requires_grad=True)
dummy_voxel = dummy_voxel.to(device)

z_vector, _ = im_network(dummy_voxel, None, None, is_training=False)
torch.onnx.export(im_network, dummy_voxel, "checkpoint/imnet_architecture.onnx")



