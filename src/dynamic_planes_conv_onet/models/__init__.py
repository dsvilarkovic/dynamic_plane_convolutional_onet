import torch
import torch.nn as nn
from torch import distributions as dist
from src.dynamic_planes_conv_onet.models import decoder


# Decoder dictionary
decoder_dict = {
    'dynamic_simple_local': decoder.DynamicLocalDecoder,
    'hybrid_simple_local': decoder.HybridLocalDecoder,
    'simple_local': decoder.LocalDecoder,
}


class DynamicPlanesConvolutionalOccupancyNetwork(nn.Module):
    ''' Dynamic Planes Convolutional Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, p0_z=None,
                 device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z

    def forward(self, p, inputs, sample=True, semantic_map=None, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        batch_size = p.size(0)
        pl = None
        if semantic_map is not None:
            c = self.encoder(semantic_map.to(self._device))
        else:
            c = self.encode_inputs(inputs)
            if hasattr(self.encoder, 'plane_parameters'):
                pl = self.encoder.plane_parameters
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(p, z, c, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, semantic_map, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        batch_size = p.shape[0]

        if semantic_map is not None:
            c = self.encoder(semantic_map.to(self._device))
        else:
            c = self.encode_inputs(inputs)
            if hasattr(self.encoder, 'plane_parameters'):
                pl = self.encoder.plane_parameters
        q_z = self.infer_z(p, occ, c, inputs=inputs, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, c, **kwargs)

        rec_error = -p_r.log_prob(occ).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).view(batch_size, -1).sum(dim=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c, = torch.empty(inputs.size(0), 0)
        return c

    def decode(self, p, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, inputs, **kwargs):
        ''' Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''

        batch_size = p.size(0)
        mean_z = torch.empty(batch_size, 0).to(self._device)
        logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
