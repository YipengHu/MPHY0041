import torch


## transformation functions
def get_reference_grid(grid_size):
    # grid_size: [height, width]
    return torch.stack(torch.meshgrid(
        torch.linspace(-1,1,grid_size[0]),
        torch.linspace(-1,1,grid_size[1])), axis=0)


def warp_images(images, ddfs, ref_grids=None):
    # images: [batch_size, height, width]
    # ddfs: [batch_size, 2, height, width]
    if ref_grids is None:
        ref_grids = get_reference_grid(ddfs.shape[2:4])
    warped_grids = ref_grids.repeat(ddfs.shape[0],1,1,1) + ddfs
    warped_grids = warped_grids.permute(0,2,3,1)[...,[1,0]]
    images = torch.unsqueeze(images,dim=1)
    warped_images = torch.nn.functional.grid_sample(images, warped_grids, align_corners=False)
    return torch.squeeze(warped_images)

    
## loss functions
def square_difference(i1, i2):
    return torch.mean((i1-i2)**2, dim=(1,2))


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2


def gradient_txy(txy, fn):
    return torch.stack([fn(txy[:,i,...]) for i in [0, 1]], axis=3)


def gradient_norm(displacement, flag_l1=False):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    if flag_l1:
        norms = torch.abs(dtdx) + torch.abs(dtdy)
    else:
        norms = dtdx**2 + dtdy**2
    return torch.mean(norms, dim=(1, 2, 3))