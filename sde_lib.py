    # Copyright (C) 2025  Jean-Daniel PASCAL PRIETO

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""

import abc
import torch
import numpy as np
import warnings
import logging


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, y, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, y, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, y, t, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        drift, diffusion = self.sde(x, y, t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, y, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, y, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, y, t, *args)
                score = score_model(x, t, y, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, y, t, stepsize):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, y, t, stepsize)
                sc = score_model(x, t, y)
                rev_f = f - G[:, None, None, None].to(device=f.device) ** 2 * sc * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--theta",
            type=float,
            default=1.5,
            help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.",
        )
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.05,
            help="The minimum sigma to use. 0.05 by default.",
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=0.5,
            help="The maximum sigma to use. 0.5 by default.",
        )
        parser.add_argument(
            "--N",
            type=int,
            default=30,
            help="The number of timesteps in the SDE discretization. 30 by default",
        )
        parser.add_argument(
            "--sampler_type",
            type=str,
            default="pc",
            help="Type of sampler to use. 'pc' by default.",
        )
        return parser

    def __init__(
        self, sigma_min, sigma_max, theta=2.0, N=30, sampler_type="pc", **ignored_kwargs
    ):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        return OUVESDE(
            self.theta,
            self.sigma_min,
            self.sigma_max,
            N=self.N,
            sampler_type=self.sampler_type,
        )

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, y, t):
        theta = self.theta
        # print("t x0 et y ",t.shape, x0.shape, y.shape)
        exp_interp = torch.exp(-theta * t)[:, None, None, None].to(device=x0.device)
        return exp_interp * x0 + (1 - exp_interp) * y.to(device=x0.device)

    def alpha(self, t):
        return torch.exp(-self.theta * t)

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            / (theta + logsig)
        )

    def marginal_prob(self, batch, t):
        x0, y = batch
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            # print(y.reshape(shape))
            warnings.warn(
                f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape."
            )
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


class SBVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=50, help="The number of timesteps in the SDE discretization. 50 by default")
        parser.add_argument("--k", type=float, default=2.6, help="Parameter of the diffusion coefficient. 2.6 by default.")
        parser.add_argument("--c", type=float, default=0.4, help="Parameter of the diffusion coefficient. 0.4 by default.")
        parser.add_argument("--eps", type=float, default=1e-8, help="Small constant to avoid numerical instability. 1e-8 by default.")
        parser.add_argument("--sampler_type", type=str, default="sde")
        return parser

    def __init__(self, k, c, N=50, eps=1e-8, sampler_type="sde", **ignored_kwargs):
        """Construct a Schrodinger Bridge with Variance Exploding SDE.

        As described in Jukić et al., „Schrödinger Bridge for Generative Speech Enhancement“, 2024.

        Args:
            k: stiffness parameter.
            c: diffusion parameter.
            N: number of discretization steps
        """
        super().__init__(N)
        self.k = k
        self.c = c
        self.N = N
        self.eps = eps
        self.sampler_type = sampler_type

    def copy(self):
        return SBVESDE(self.k, self.c, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        f = 0.0                                                                 # Table 1
        g = torch.sqrt(torch.tensor(self.c)) * self.k ** (t)                      # Table 1
        return f, g

    def _sigmas_alphas(self, t):
        alpha_t = torch.ones_like(t)
        alpha_T = torch.ones_like(t)
        sigma_t = torch.sqrt((self.c*(self.k**(2*t)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1
        sigma_T = torch.sqrt((self.c*(self.k**(2*self.T)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1   
        
        alpha_bart = alpha_t / (alpha_T + self.eps)                             # below Eq. (9)
        sigma_bart = torch.sqrt(sigma_T**2 - sigma_t**2 + self.eps)             # below Eq. (9)

        return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart

    def _mean(self, x0, y, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        w_xt = alpha_t * sigma_bart**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        w_yt = alpha_bart * sigma_t**2 / (sigma_T**2 + self.eps)                # below Eq. (11)

        mu = w_xt[:, None, None, None] * x0 + w_yt[:, None, None, None] * y     # Eq. (11)
        return mu

    def _std(self, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        sigma_xt = (alpha_t * sigma_bart * sigma_t) / (sigma_T + self.eps) 
        return sigma_xt

    def marginal_prob(self, batch, t):  
        x0, y = batch
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        x_T = y
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for SBVE SDE not yet implemented!")  
    


#This is the SDE for the Brownian Bridge with Exploding Diffusion Coefficient. It uses the same parameterization as in the paper bbed vs ouve.
class BBED(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--T_sampling", type=float, default=0.999, help="The T so that t < T during sampling in the train step.")
        parser.add_argument("--k", type=float, default = 2.6, help="base factor for diffusion term") 
        parser.add_argument("--theta", type=float, default = 0.52, help="root scale factor for diffusion term.")
        return parser

    def __init__(self, T_sampling, k, theta, N=1000, **kwargs):
        """Construct an Brownian Bridge with Exploding Diffusion Coefficient SDE with parameterization as in the paper.
        dx = (y-x)/(Tc-t) dt + sqrt(theta)*k^t dw
        """
        super().__init__(N)
        self.k = k
        self.logk = np.log(self.k)
        self.theta = theta
        self.N = N
        self.Eilog = sc.expi(-2*self.logk)
        self.T = T_sampling #for sampling in train step and inference
        self.Tc = 1 #for constructing the SDE, dont change this


    def copy(self):
        return BBED(self.T, self.k, self.theta, N=self.N)


    def T(self):
        return self.T
    
    def Tc(self):
        return self.Tc


    def sde(self, x, y, t):
        drift = (y - x)/(self.Tc - t)
        sigma = (self.k) ** t
        diffusion = sigma * np.sqrt(self.theta)
        return drift, diffusion


    def _mean(self, x0, y, t):
        time = (t/self.Tc)[:, None, None, None]
        mean = x0*(1-time) + y*time
        return mean

    def _std(self, t):
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        h = 2*self.k**2*self.logk
        var = (self.k**(2*t_np)-1+t_np) + h*(1-t_np)*Eis
        var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        return torch.sqrt(var)

    def marginal_prob(self, batch, t):
        x0, y = batch
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T*torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for BBED not yet implemented!")

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min))
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1],
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G
