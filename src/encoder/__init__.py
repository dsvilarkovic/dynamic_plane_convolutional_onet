from src.encoder import (
   pointnet, unet
)


encoder_dict = {
    'unet': unet.UNet,
    'dynamic_pointnet_local_pool': pointnet.DynamicLocalPoolPointnet,
    'hybrid_pointnet_local_pool': pointnet.HybridLocalPoolPointnet,
    'local_pool' : pointnet.LocalPoolPointnet,
}
