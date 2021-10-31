import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
c = - 0.5 * math.log(2*math.pi)

# # %%
# try:
#     from libcpab import cpab
#
#
#     class ST_CPAB(nn.Module):
#         def __init__(self, input_shape):
#             super(ST_CPAB, self).__init__()
#             self.input_shape = input_shape
#             self.cpab = cpab([2, 4], backend='pytorch', device='gpu',
#                              zero_boundary=True,
#                              volume_perservation=False)
#
#         def forward(self, x, theta, inverse=False):
#             if inverse:
#                 theta = -theta
#             out = self.cpab.transform_data(data=x,
#                                            theta=theta,
#                                            outsize=self.input_shape[1:])
#             return out
#
#         def trans_theta(self, theta):
#             return theta
#
#         def dim(self):
#             return self.cpab.get_theta_dim()
# except Exception as e:
#     print('Could not import libcpab, error was')
#     print(e)
#
#
#     class ST_CPAB(nn.Module):
#         def __init__(self, input_shape):
#             super(ST_CPAB, self).__init__()
#             self.input_shape = input_shape
#
#         def forward(self, x, theta, inverse=False):
#             raise ValueError('''libcpab was not correctly initialized, so you
#                              cannot run with --stn_type cpab''')


class ST_Affine(nn.Module):
    def __init__(self, input_shape):
        super(ST_Affine, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, theta, inverse=False):
        if inverse:
            A = theta[:, :4]
            b = theta[:, 4:]
            A = torch.inverse(A.view(-1, 2, 2)).reshape(-1, 4)
            b = -b
            theta = torch.cat((A, b), dim=1)

        theta = theta.view(-1, 2, 3)
        output_size = torch.Size([x.shape[0], self.input_shape])
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x

    def trans_theta(self, theta):
        return theta

    def dim(self):
        return 6

def expm(theta):
    n_theta = theta.shape[0]
    zero_row = torch.zeros(n_theta, 1, 3, dtype=theta.dtype, device=theta.device)
    theta = torch.cat([theta, zero_row], dim=1)
    theta = torch_expm(theta)
    theta = theta[:,:2,:]
    return theta


def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    #R, _ = torch.gesv(P, Q)  # solve P = Q*R
    R, _ = torch.solve(P, Q)  # solve P = Q*R

    # Unsquaring step
    n = n_squarings.max()
    res = [R]
    for i in range(n):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA

def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)


def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(A, torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[
        1] * ident)
    V = torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


class ST_AffineDiff(nn.Module):
    def __init__(self, input_shape):
        super(ST_AffineDiff, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, theta, inverse=False):
        if inverse:
            theta = -theta
        #print("affine diff: ", x.shape, theta.shape)
        theta = theta.view(-1, 2, 3)
        theta = expm(theta)
        output_size = x.shape #torch.Size([x.shape[0], 3, 64, 64]) #self.input_shape
        #print("2", theta.shape, output_size)
        #output_size = [32, 3, 64, 64]
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)
        return x

    def trans_theta(self, theta):
        return expm(theta)

    def dim(self):
        return 6


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class mlp_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super(mlp_decoder, self).__init__()
        outputnonlin = Identity()
        self.flat_dim = output_shape #np.prod(output_shape)
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 64), #256
            nn.LeakyReLU(),
            #nn.Linear(256, 512),
            #nn.LeakyReLU(),
            nn.Linear(64, self.flat_dim), #256
            outputnonlin
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            #nn.Linear(256, 512),
            #nn.LeakyReLU(),
            nn.Linear(64, self.flat_dim),
            nn.Softplus()
        )
    def forward(self, z):
        x_mu = self.decoder_mu(z).reshape(-1, self.output_shape)
        x_var = self.decoder_var(z).reshape(-1, self.output_shape)
        return x_mu, x_var



class encoder_vae(nn.Module):
    def __init__(self, in_dim):
        super(encoder_vae, self).__init__()
        self.dim = in_dim
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim, 64), #128
            nn.LeakyReLU(),
            #nn.Linear(128, 128),
            #nn.LeakyReLU(),
            nn.Linear(64, self.dim) #128
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.Linear(self.dim, 64),
            nn.LeakyReLU(),
            #nn.Linear(128, 128),
            #nn.LeakyReLU(),
            nn.Linear(64, self.dim),
            nn.Softplus()
        )
    def forward(self, x):
        #print('x before', x.shape)
        x = x.view(x.shape[0], -1)
        #print('x after', x.shape)
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var


class encoder_vae_azade(nn.Module):
    def __init__(self, in_dim):
        super(encoder_vae_azade, self).__init__()
        self.dim = in_dim
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(self.dim, 128, 1, 1, 0), #128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.dim, 1, 1, 0),  # 128  //4
            nn.LeakyReLU()
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(self.dim, 128, 1, 1, 0), #128
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.dim, 1, 1, 0),  # 128 # //4
            nn.LeakyReLU(),
            nn.Softplus()
        )
    def forward(self, x):
        #print('x before', x.shape)
        #x = x.view(x.shape[0], -1)

        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        #print('z after', z_mu.shape)
        return z_mu, z_var

class decoder_vae_azade(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super(decoder_vae_azade, self).__init__()
        outputnonlin = Identity()
        self.flatten = nn.Flatten()
        #self.flat_dim = output_shape #np.prod(output_shape)
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 64), #256
            nn.LeakyReLU(),
            #nn.Linear(256, 512),
            #nn.LeakyReLU(),
            nn.Linear(64, self.output_shape), #256
            outputnonlin
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            #nn.Linear(256, 512),
            #nn.LeakyReLU(),
            nn.Linear(64, self.output_shape),
            nn.Softplus()
        )
    def forward(self, z):
        z = self.flatten(z)
        x_mu = self.decoder_mu(z)
        #print(x_mu.shape, self.output_shape)
        x_mu = x_mu.reshape(-1, self.output_shape)
        x_var = self.decoder_var(z).reshape(-1, self.output_shape)
        return x_mu, x_var

class VITAE(nn.Module):
    def __init__(self, in_dim, dis_objs):
        super(VITAE, self).__init__()
        self.dim = in_dim

        if dis_objs:
            self.encoder1 = encoder_vae(in_dim)  # encoder_mlp
            self.encoder2 = encoder_vae(in_dim)  # encoder_mlp
        else:
            self.encoder1 = encoder_vae_azade(in_dim) #encoder_mlp
            self.encoder2 = encoder_vae_azade(in_dim) #encoder_mlp

        #self.stn = ST_CPAB(in_dim)
        #self.stn = ST_Affine(in_dim)
        self.stn = ST_AffineDiff(in_dim) #was in_dim
        #self.ST_type = ST_type

    def reparameterize(self, mu, var, x, eq_samples=1, iw_samples=1):
        #batch_size, latent_dim = mu.shape
        #eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        eps = torch.randn_like(mu, device=var.device)
        output = (mu + var.sqrt() * eps) #.reshape(x.shape) #[:, None, None, :]
        #print(output.shape)
        return output #.reshape(-1, latent_dim)

    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, x, eq_samples, iw_samples)

        mu2, var2 = self.encoder2(x)
        z2 = self.reparameterize(mu2, var2, x, eq_samples, iw_samples)

        return [z1, z2], [mu1, mu2], [var1, var2]


def vae_loss(x, x_mu, x_var, z, z_mus, z_vars, eq_samples, iw_samples,
             latent_dim, epoch, warmup, beta, outputdensity):
    """ Calculates the ELBO for a variational autoencoder
    Arguments:
        x: input data [batch_size, *input_dim]
        x_mu: mean reconstruction [batch_size x eq_samples x iw_samples, *input_dim]
        x_var: variance of reconstruction [batch_size x eq_samples x iw_samples, *input_dim]
        z: latent variable
        mus: mean in latent space
        logvars: log variance in latent space
        eq_samples: int, number of equality samples
        iw_samples: int, number of importance weighted samples
        latent_dim: int, size of the latent space
        epoch: int, which epoch we are at
        warmup: int, how many warmup epoch to do
        outputdensity: str, output density of generative model
    Output:
        lower_bound: lower bound that should be maximized
        recon_term: reconstruction term for the ELBO
        kl_term: kl terms (multiple if multiple latents) in the ELBO term
    """
    eps = 1e-5  # to control underflow in variance estimates
    weight = kl_scaling(epoch, warmup) * beta

    batch_size = x.shape[0]
    x = x.view(batch_size, 1, 1, -1)
    x_mu = x_mu.view(batch_size, eq_samples, iw_samples, -1)

    if z_mus[-1].shape[0] == batch_size:
        shape = (1, 1)
    else:
        shape = (eq_samples, iw_samples)

    z = [zs.view(-1, eq_samples, iw_samples, latent_dim) for zs in z]
    z_mus = [z_mus[0].view(-1, 1, 1, latent_dim)] + [m.view(-1, *shape, latent_dim) for m in z_mus[1:]]
    z_vars = [z_vars[0].view(-1, 1, 1, latent_dim)] + [l.view(-1, *shape, latent_dim) for l in z_vars[1:]]

    log_pz = [log_stdnormal(zs) for zs in z]
    log_qz = [log_normal2(zs, m, torch.log(l + eps)) for zs, m, l in zip(z, z_mus, z_vars)]

    if outputdensity == 'bernoulli':
        x_mu = x_mu.clamp(1e-5, 1 - 1e-5)
        log_px = (x * x_mu.log() + (1 - x) * (1 - x_mu).log())
    elif outputdensity == 'gaussian':
        x_var = x_var.view(batch_size, eq_samples, iw_samples, -1)
        log_px = log_normal2(x, x_mu, torch.log(x_var + eps), eps)
    else:
        ValueError('Unknown output density')
    #print(log_px.shape, log_pz[0].shape, log_qz[0].shape)
    #a_0 = log_px.sum(dim=3)
    a_0 = 0 #a_0.repeat(int(3538944/6),1,1) #.reshape(-1,1,1)
    a_1 = weight * sum([p.sum(dim=3) for p in log_pz])
    a_2 = weight * - sum([p.sum(dim=3) for p in log_qz])
    #print(a_0.shape, a_1.shape, a_2.shape)
    a = a_0 + a_1 + a_2 #expand_as
    a_max = torch.max(a, dim=2, keepdim=True)[0]  # (batch_size, nsamples, 1)
    lower_bound = torch.mean(a_max) + torch.mean(torch.log(torch.mean(torch.exp(a - a_max), dim=2)))
    recon_term = log_px.sum(dim=3).mean()
    kl_term = [(lp - lq).sum(dim=3).mean() for lp, lq in zip(log_pz, log_qz)]
    return lower_bound, recon_term, kl_term


# %%
def log_stdnormal(x):
    """ Log probability of standard normal distribution elementwise """
    return c - x ** 2 / 2


# %%
def log_normal2(x, mean, log_var, eps=0.0):
    """ Log probability of normal distribution elementwise """
    return c - log_var / 2 - (x - mean) ** 2 / (2 * torch.exp(log_var) + eps)


# %%
def kl_scaling(epoch=None, warmup=None):
    """ Annealing term for the KL-divergence """
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))
