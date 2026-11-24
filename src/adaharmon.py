"""
AdaHarmon: A Harmonic-Mean Adaptive Optimizer for CNN-Based Classification of
Hematological Microscopic Images.

Reference implementation accompanying the manuscript (Yanes-Perez,
Hernandez-Nolasco and Wister). This is the class used in all experiments.
"""
import torch
from torch.optim import Optimizer


class AdaHarmon(Optimizer):
    r"""
    AdaHarmon update rule (all operations element-wise).

    1. Sign-consistency estimator:
           s_t = beta_s * s_{t-1} + (1 - beta_s) * sign(g_t)
    2. Adaptive momentum coefficient and cumulative product:
           beta1_t = beta1 * (1 + lambda_s * |s_t|) / (1 + lambda_s)
           mu_t    = mu_{t-1} * beta1_t
       Because beta1_t is per parameter, mu_t is a per-parameter tensor.
    3. First moment with bias correction generalized to variable momentum:
           m_t     = beta1_t * m_{t-1} + (1 - beta1_t) * g_t
           m_hat_t = m_t / (1 - mu_t)
    4. Second moment (as in Adam):
           v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
           v_hat_t = v_t / (1 - beta2^t)
    5. Softened l_inf state:
           delta_t = 1 + soft_decay * 1[|g_t| < tau * u_{t-1}]
           u_t     = max(beta2^{delta_t} * u_{t-1}, |g_t|)
    6. Hybrid denominator (harmonic mean of sqrt(v_hat_t) and u_t):
           h_t = 2 * sqrt(v_hat_t) * u_t / (sqrt(v_hat_t) + u_t + eps1)
    7. Parameter update:
           theta_t = theta_{t-1} - lr * m_hat_t / (h_t + eps2)

    Optimizer state: five per-parameter buffers (m, v, u, s, mu), i.e. 2.5x
    the state of Adam. With amsgrad=True a sixth buffer is added.

    Arguments:
        params:      iterable of parameters or parameter groups.
        lr:          learning rate (default: 1e-3).
        betas:       (beta1, beta2, beta_s) (default: (0.9, 0.999, 0.9)).
        eps:         (eps1, eps2), inner and outer epsilons
                     (default: (1e-8, 1e-6)).
        lambda_s:    scale of the adaptive momentum range (default: 1.0).
        soft_decay:  extra decay of the l_inf state on plateaus, the gamma
                     of the paper (default: 0.5).
        tau:         relative plateau threshold (default: 0.1).
        amsgrad:     optional AMSGrad-style maximum on the second moment.
                     Off by default and not used in the reported experiments.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999, 0.9),
        eps: tuple = (1e-8, 1e-6),
        lambda_s: float = 1.0,
        soft_decay: float = 0.5,
        tau: float = 0.1,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not (0.0 <= betas[2] < 1.0):
            raise ValueError(f"Invalid beta_s: {betas[2]}")
        if eps[0] <= 0 or eps[1] <= 0:
            raise ValueError(f"eps must be positive: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            lambda_s=lambda_s,
            soft_decay=soft_decay,
            tau=tau,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta_s = group["betas"]
            eps1, eps2 = group["eps"]
            lr = group["lr"]
            lambda_s = group["lambda_s"]
            soft_decay = group["soft_decay"]
            tau = group["tau"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaHarmon does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)   # first moment
                    state["v"] = torch.zeros_like(p)   # second moment (l2)
                    state["u"] = torch.zeros_like(p)   # softened l_inf state
                    state["s"] = torch.zeros_like(p)   # sign consistency
                    state["mu"] = torch.ones_like(p)   # cumulative product mu_t
                    if amsgrad:
                        state["v_max"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v, u, s, mu = (
                    state["m"], state["v"], state["u"], state["s"], state["mu"]
                )

                # 1. Sign-consistency estimator
                s.mul_(beta_s).add_(torch.sign(grad), alpha=(1.0 - beta_s))

                # 2. Adaptive momentum per parameter and cumulative product
                beta1_t = beta1 * (1.0 + lambda_s * s.abs()) / (1.0 + lambda_s)
                beta1_t.clamp_(0.0, 1.0 - 1e-9)
                mu.mul_(beta1_t)

                # 3. First moment with generalized bias correction
                m.mul_(beta1_t).addcmul_((1.0 - beta1_t), grad)
                bias_correction_m = 1.0 - mu
                bias_correction_m.clamp_(min=1e-12)
                m_hat = m / bias_correction_m

                # 4. Second moment (l2)
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                if amsgrad:
                    v_max = state["v_max"]
                    torch.maximum(v_max, v, out=v_max)
                    v_eff = v_max
                else:
                    v_eff = v
                v_hat = v_eff / (1.0 - beta2 ** t)

                # 5. Softened l_inf state
                grad_abs = grad.abs()
                plateau_mask = (grad_abs < tau * u).float()
                beta2_eff = beta2 ** (1.0 + soft_decay * plateau_mask)
                torch.maximum(beta2_eff * u, grad_abs, out=u)

                # 6. Hybrid denominator (harmonic mean)
                sqrt_v_hat = v_hat.sqrt()
                h_tilde = (2.0 * sqrt_v_hat * u) / (sqrt_v_hat + u + eps1)

                # 7. Parameter update
                p.addcdiv_(m_hat, h_tilde + eps2, value=-lr)

        return loss
