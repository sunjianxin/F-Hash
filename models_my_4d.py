import torch
import torch.nn as nn
import torch.nn.functional as F
# import tinycudann32 as tcnn
import tinycudann as tcnn
import numpy as np

# implemantation from HashNeRF-pytorch
# BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device='cuda') # 1 x 8 x 3, int64
BOX_OFFSETS = torch.tensor([[[i,j,k,l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]], device='cuda') # 1 x 8 x 3, int64

def hash_my(coords, resolution_t, resolution_x, resolution_y, resolution_z):
    '''
    coords: this function can process upto 7 dim coordinates, B x 8 x 3
    '''
    # indices = coords[..., 2]*resolution_x*resolution_y + coords[..., 1]*resolution_x + coords[..., 0]
    indices = coords[..., 0]*resolution_x*resolution_y*resolution_z + coords[..., 3]*resolution_x*resolution_y + coords[..., 2]*resolution_x + coords[..., 1]
    # print("indices.shape:", indices.shape)
    # print(indices[0:100, :])
    indices = indices.to(dtype=torch.int64)
    return indices

def get_voxel_vertices(txyz, bounding_box, resolution_t, resolution_x, resolution_y, resolution_z):
    '''
    txyz: 3D coordinates of samples. B x 4
    bounding_box: min and max t,x,y,z coordinates of object bbox
    resolution_t/x/y/z: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    device = txyz.device  # Get the device of `txyz`
    box_min = box_min.to(device)
    box_max = box_max.to(device)
    
    keep_mask = txyz==torch.max(torch.min(txyz, box_max), box_min) # B x 8
    if not torch.all(txyz <= box_max) or not torch.all(txyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        txyz = torch.clamp(txyz, min=box_min, max=box_max)

    resolution = torch.tensor([resolution_t - 1, resolution_x - 1, resolution_y - 1, resolution_z - 1], dtype=torch.float32).to(device)
    grid_size = (box_max-box_min)/resolution
    # print("grid_size:", grid_size.dtype, grid_size.shape, grid_size, grid_size.device)
    
    bottom_left_idx = torch.floor((txyz-box_min)/grid_size).int() # B x 4, grid index (int32), [0, 1, ..., resolution]
    # change all t, x, y, z values to value_new = value - 1, if values == resolution - 1
    # change t
    bottom_left_idx[:, 0] = torch.where(bottom_left_idx[:, 0] == resolution_t - 1, resolution_t - 1 - 1, bottom_left_idx[:, 0])
    # change x
    bottom_left_idx[:, 1] = torch.where(bottom_left_idx[:, 1] == resolution_x - 1, resolution_x - 1 - 1, bottom_left_idx[:, 1])
    # change y
    bottom_left_idx[:, 2] = torch.where(bottom_left_idx[:, 2] == resolution_y - 1, resolution_y - 1 - 1, bottom_left_idx[:, 2])
    # change z
    bottom_left_idx[:, 3] = torch.where(bottom_left_idx[:, 3] == resolution_z - 1, resolution_z - 1 - 1, bottom_left_idx[:, 3])

    
    # bottom_left_idx[bottom_left_idx == resolution - 1] = resolution - 1 - 1
    voxel_min_vertex = bottom_left_idx*grid_size + box_min # B x 4, location of the min corner (float)
    voxel_max_vertex = voxel_min_vertex + grid_size # B x 4, location of the max corner (float)

    # print("bottom_left_idx max:", torch.max(bottom_left_idx))
    # print("bottom_left_idx min:", torch.min(bottom_left_idx))

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS # B x 1 x 4 + 1 x 16 x 4 = B x 16 x 4, grid index (int64), [0, 1, ..., resolution]
    # print("voxel_indices max:", torch.max(voxel_indices))
    # print("voxel_indices min:", torch.min(voxel_indices))
    hashed_voxel_indices = hash_my(voxel_indices, resolution_t, resolution_x, resolution_y, resolution_z) # B x 16, hashed_indices(returned from hash function) of 16 corners

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

class HashEmbedder(nn.Module):
    def __init__(self,
                 bounding_box,
                 n_levels,
                 n_features_per_level,
                 time_resolutions,
                 x_resolutions,
                 y_resolutions,
                 z_resolutions):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.t_resolutions = time_resolutions
        self.x_resolutions = x_resolutions
        self.y_resolutions = y_resolutions
        self.z_resolutions = z_resolutions
        

        # b is always 2 for dense grid
        # self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        # print("b", self.b)

        embedding_size = []
        for i in range(n_levels):
            embedding_size.append(self.t_resolutions[i]*self.x_resolutions[i]*self.y_resolutions[i]*self.z_resolutions[i])  

        self.embeddings = nn.ModuleList([nn.Embedding(embedding_size[i], \
                                        # self.n_features_per_level) for i in range(n_levels)])
                                        self.n_features_per_level) for i in range(len(embedding_size))])
        for i in range(len(embedding_size)):
            print(self.embeddings[i])
        
        # custom uniform initialization
        for i in range(len(embedding_size)):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def quadrilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 4
        voxel_min_vertex: B x 4
        voxel_max_vertex: B x 4
        voxel_embedds: B x 16 x 2
        '''
        # Compute weights for interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 4
    
        # Step 1: Interpolate along the x-axis (16 -> 8 points)
        c0000 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 8] * weights[:, 0][:, None]
        c0001 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 9] * weights[:, 0][:, None]
        c0010 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 10] * weights[:, 0][:, None]
        c0011 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 11] * weights[:, 0][:, None]
        c0100 = voxel_embedds[:, 4] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 12] * weights[:, 0][:, None]
        c0101 = voxel_embedds[:, 5] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 13] * weights[:, 0][:, None]
        c0110 = voxel_embedds[:, 6] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 14] * weights[:, 0][:, None]
        c0111 = voxel_embedds[:, 7] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 15] * weights[:, 0][:, None]
    
        # Step 2: Interpolate along the y-axis (8 -> 4 points)
        c000 = c0000 * (1 - weights[:, 1][:, None]) + c0100 * weights[:, 1][:, None]
        c001 = c0001 * (1 - weights[:, 1][:, None]) + c0101 * weights[:, 1][:, None]
        c010 = c0010 * (1 - weights[:, 1][:, None]) + c0110 * weights[:, 1][:, None]
        c011 = c0011 * (1 - weights[:, 1][:, None]) + c0111 * weights[:, 1][:, None]
    
        # Step 3: Interpolate along the z-axis (4 -> 2 points)
        c00 = c000 * (1 - weights[:, 2][:, None]) + c010 * weights[:, 2][:, None]
        c01 = c001 * (1 - weights[:, 2][:, None]) + c011 * weights[:, 2][:, None]
    
        # Step 4: Interpolate along the w-axis (2 -> 1 point)
        c0 = c00 * (1 - weights[:, 3][:, None]) + c01 * weights[:, 3][:, None]
    
        return c0

    def forward(self, x):
        # x is 3D point position: B x 4
        x_embedded_all = []

        for i in range(self.n_levels):
            resolution_t = self.t_resolutions[i]
            resolution_x = self.x_resolutions[i]
            resolution_y = self.y_resolutions[i]
            resolution_z = self.z_resolutions[i]
            # print("resolution: ", resolution_x, resolution_y, resolution_z)
            # voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(x,
            #                                                                                          self.bounding_box, 
            #                                                                                          resolution,
            #                                                                                          self.log2_hashmap_size)
            # print(hashed_voxel_indices.dtype)
            # voxel_min_vertex: B x 4, location of the min corner
            # voxel_max_vertex: B x 4, location of the max corner
            # hashed_voxel_indices: B x 16, hashed_indices(returned from hash function) of 8 corners
            # keep_mask.shape: B x 16
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(x,
                                                                                                     self.bounding_box,
                                                                                                     resolution_t,
                                                                                                     resolution_x,
                                                                                                     resolution_y,
                                                                                                     resolution_z)
            # print("hashed_voxel_indices.min: ", torch.min(hashed_voxel_indices))
            # print("hashed_voxel_indices.max: ", torch.max(hashed_voxel_indices))
            voxel_embedds = self.embeddings[i](hashed_voxel_indices) # B x 16 x 2(feature size), features of 16 corners
            

            # x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded = self.quadrilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask

class INR_denseGrid4D(nn.Module):
    def __init__(self, 
                 num_levels=6, # 2, 4, 8, 16(org), 32, 64, 128, 256 
                 level_dim=2,
                 time_resolutions=[],
                 x_resolutions=[],
                 y_resolutions=[],
                 z_resolutions=[],
                 hidden_dim=64):
        super().__init__()
        
        # Multi-resolution hash encoding
        self.encoder = HashEmbedder((torch.tensor([0, -1, -1, -1]), torch.tensor([8, 1, 1, 1])),
                                    n_levels=num_levels,
                                    n_features_per_level=level_dim,
                                    time_resolutions=time_resolutions,
                                    x_resolutions=x_resolutions,
                                    y_resolutions=y_resolutions,
                                    z_resolutions=z_resolutions)

        # MLP to predict the scalar value
        input_dim = num_levels * level_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 3), the 3D coordinates

        Returns:
            Tensor of shape (batch_size, 1), the predicted scalar value
        """
        # Encode input coordinates
        encoded, keep_mask = self.encoder(x)
        # print("encoded.shape", encoded.shape)

        # Predict scalar value
        return self.mlp(encoded)

        
        
        