import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
import math
import random


# Utilities.

def invert_eq21_to_c(f_target: float, dt: float, kp: float, ko: float,
                     xi_norm: float, dx: float = 1.0) -> float:
    """
    note f_target note c，note（note，note）。
    note：G = sqrt((1+dt*kp)(1+dt*ko)), S = 1/(1+dt*kp) + 1/(1+dt*ko)
    tan(pi f dt) = 2 c dt ||xi|| / (G S)   =>  c = (G S / (2 dt ||xi||)) * tan(pi f dt)
    """
    G = math.sqrt((1.0 + dt * kp) * (1.0 + dt * ko))
    S = 1.0 / (1.0 + dt * kp) + 1.0 / (1.0 + dt * ko)
    # Guard against numerical singularities.
    c = (G * S / (2.0 * dt * max(xi_norm, 1e-6))) * math.tan(math.pi * f_target * dt)
    return max(c, 1e-6)


def cfl_c_max(dt: float, kp: float, ko: float, dx: float = 1.0) -> float:
    """
    Eq.20（noteCFL）：c <= (dx/dt) * G / sqrt(2)
    """
    G = math.sqrt((1.0 + dt * kp) * (1.0 + dt * ko))
    return (dx / dt) * (G / math.sqrt(2.0))


# Grid.

def build_region_masks(grid_h: int, grid_w: int, region_rows: int, region_cols: int) -> torch.Tensor:
    """
    note HxW note region_rows x region_cols note，note [R,H,W] note
    """
    R = region_rows * region_cols
    masks = torch.zeros(R, grid_h, grid_w)
    h_step = grid_h // region_rows
    w_step = grid_w // region_cols
    idx = 0
    for r in range(region_rows):
        for c in range(region_cols):
            h0, h1 = r * h_step, (r + 1) * h_step if r < region_rows - 1 else grid_h
            w0, w1 = c * w_step, (c + 1) * w_step if c < region_cols - 1 else grid_w
            masks[idx, h0:h1, w0:w1] = 1.0
            idx += 1
    return masks


def make_freq_pool(f_min: float, f_max: float, B: int) -> List[float]:
    """
    note [f_min, f_max] note B note
    """
    if B <= 1:
        return [0.5 * (f_min + f_max)]
    return [f_min + i * (f_max - f_min) / (B - 1) for i in range(B)]


def assign_region_freqs(region_rows: int, region_cols: int, freq_pool: List[float]) -> List[float]:
    """
    note“note/note”note round-robin note，note region note。
    """
    R = region_rows * region_cols
    B = len(freq_pool)
    freqs = []
    for r in range(region_rows):
        for c in range(region_cols):
            # Simple effective assignment.
            f = freq_pool[(r + c) % B]
            freqs.append(f)
    assert len(freqs) == R
    return freqs


# Amplitude.

class CausalOLAResampler:
    """
    note OLA：note a[m]（note region note R note），
    note（note L_s），note L_s note（note dt）note a[n]。
    note：PANN 4s note/1s stride -> BioOSS dt=0.01 note 100Hz note。
    """
    def __init__(self, R: int, dt: float, window_len_s: float = 4.0, stride_s: float = 1.0):
        self.R = R
        self.dt = dt
        self.win_len_steps = int(round(window_len_s / dt))
        self.stride_steps = int(round(stride_s / dt)) # one frame per second.
        # One-sided Hann window.
        n = np.arange(self.win_len_steps, dtype=np.float32)
        self.win = 0.5 * (1.0 - np.cos(np.pi * (n / max(self.win_len_steps - 1, 1))))
        self.num_buf = torch.zeros(R, self.win_len_steps) # Numerator accumulator.
        self.den_buf = torch.zeros(self.win_len_steps) # Denominator accumulator.
        self.ptr = 0 # Current write pointer.

    def push_frame(self, a_R: torch.Tensor):
        """
        note region note（R,）。note a_R * win note win_len_steps note。
        """
        assert a_R.shape == (self.R,)
        # circular.
        # Write interval.
        for k in range(self.R):
            # Implementation note.
            self.num_buf[k] = torch.roll(self.num_buf[k], -1 * self.ptr)
            self.num_buf[k] += a_R[k] * torch.from_numpy(self.win)
            self.num_buf[k] = torch.roll(self.num_buf[k], self.ptr)
        # Implementation note.
        self.den_buf = torch.roll(self.den_buf, -1 * self.ptr)
        self.den_buf += torch.from_numpy(self.win)
        self.den_buf = torch.roll(self.den_buf, self.ptr)

    def step(self) -> torch.Tensor:
        """
        note（dt），note a_R[n]，note。
        """
        # Current output.
        denom = max(self.den_buf[self.ptr].item(), 1e-8)
        a_now = self.num_buf[:, self.ptr] / denom  # (R,)
        # Clear consumed slots.
        self.num_buf[:, self.ptr] = 0.0
        self.den_buf[self.ptr] = 0.0
        # Advance the pointer.
        self.ptr = (self.ptr + 1) % self.win_len_steps
        return a_now


class PANNToRegionModulator(nn.Module):
    """
    note：
      1) 527 -> R note (note) note region note；
      2) note 1Hz note region note OLA note 100Hz note a_r[n]；
      3) note region note：u_r[n] = a_r[n] * sin(2π f_r n dt)

    note：
      - note PANN note，note push_pann_frame()；
      - note FDTD note step() note source（[1,1,H,W]）。
    """
    def __init__(self, grid_h: int, grid_w: int, region_rows: int, region_cols: int,
                 dt: float, region_freqs: List[float], region_masks: torch.Tensor,
                 pann_dim: int = 527, seed: int = 17):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.region_rows, self.region_cols = region_rows, region_cols
        self.R = region_rows * region_cols
        self.dt = dt
        self.freqs = torch.tensor(region_freqs, dtype=torch.float32)  # (R,)
        self.masks = region_masks.clone().float()                      # (R,H,W)
        # custom.
        torch.manual_seed(seed)
        W = torch.randn(self.R, pann_dim)
        # Normalize columns.
        W = W / (W.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer('W', W)
        # Causal OLA.
        self.resampler = CausalOLAResampler(R=self.R, dt=dt, window_len_s=4.0, stride_s=1.0)
        # Maintain phase continuity.
        self.register_buffer('n_step', torch.zeros(1, dtype=torch.long))

        # Global scaling.
        self.alpha = 1.0

    @torch.no_grad()
    def push_pann_frame(self, pann_probs_527: torch.Tensor):
        """
        note PANN note (527,) note (B,527)。note batch>1，notepush。
        note -> note -> note -> OLA note
        """
        if pann_probs_527.dim() == 2:
            pann_probs_527 = pann_probs_527.mean(dim=0)
        assert pann_probs_527.shape[0] == self.W.shape[1]
        # Dynamic-range compression.
        x = pann_probs_527.clamp_min(0.0)
        x = torch.log1p(3.0 * x)
        # 527 -> R
        a_R = (self.W @ x).clamp_min(0.0)  # (R,)
        # Implementation note.
        self.resampler.push_frame(a_R.cpu())

    @torch.no_grad()
    def step(self, device: torch.device) -> torch.Tensor:
        """
        note BioOSS note，note [1,1,H,W]
        """
        # Implementation note.
        a_R = self.resampler.step().to(device)  # (R,)
        # Maintain phase continuity.
        n = self.n_step.item()
        phi = 2.0 * math.pi * self.freqs.to(device) * (n * self.dt)  # (R,)
        g_R = torch.sin(phi)  # (R,)
        u_R = self.alpha * a_R * g_R  # (R,)

        # Spatial injection.
        src_hw = torch.einsum('r,rhw->hw', u_R, self.masks.to(device))  # (H,W)
        self.n_step += 1
        return src_hw.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


# Invert the target frequency.

class BioOSSFDTD2D(nn.Module):
    """
    BioOSS 2D FDTD：c note region note（note）note Eq.21 note。
    kp, ko note（note），periodic note。
    """
    def __init__(self, grid_h: int = 64, grid_w: int = 64,
                 dt: float = 0.01, freq_range: Tuple[float, float] = (40.0, 80.0),
                 region_rows: int = 8, region_cols: int = 8,  # R=64
                 kp: float = 0.02, ko: float = 0.02, xi_norm: float = math.sqrt(2.0),
                 boundary_condition: str = 'periodic'):
        super().__init__()
        self.H, self.W = grid_h, grid_w
        self.dt = dt
        self.kp_val, self.ko_val = kp, ko
        self.boundary_condition = boundary_condition

        # Frequency pool.
        masks = build_region_masks(grid_h, grid_w, region_rows, region_cols)  # [R,H,W]
        B = min(region_rows + region_cols, 32) # Implementation note.
        freq_pool = make_freq_pool(freq_range[0], freq_range[1], B)
        region_freqs = assign_region_freqs(region_rows, region_cols, freq_pool)  # len=R
        self.register_buffer('region_masks', masks)
        self.region_rows, self.region_cols = region_rows, region_cols
        self.R = region_rows * region_cols

        # Invert the target frequency.
        c_max = cfl_c_max(dt=dt, kp=kp, ko=ko, dx=1.0) * 0.99
        c_list = []
        for fr in region_freqs:
            c_val = invert_eq21_to_c(f_target=fr, dt=dt, kp=kp, ko=ko, xi_norm=xi_norm, dx=1.0)
            if not np.isfinite(c_val) or c_val <= 0:
                c_val = 1.0
            c_val = min(c_val, c_max)
            c_list.append(c_val)
        c_list = torch.tensor(c_list, dtype=torch.float32)  # (R,)

        # Grid.
        c_grid = torch.einsum('r,rhw->hw', c_list, masks)  # (H,W)
        kp_grid = torch.full((self.H, self.W), kp, dtype=torch.float32)
        ko_grid = torch.full((self.H, self.W), ko, dtype=torch.float32)

        self.register_buffer('c', c_grid.unsqueeze(0).unsqueeze(0))    # [1,1,H,W]
        self.register_buffer('kp', kp_grid.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
        self.register_buffer('ko', ko_grid.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
        self.register_buffer('region_freqs', torch.tensor(region_freqs, dtype=torch.float32))

        # State.
        self.register_buffer('p', torch.zeros(1, 1, self.H, self.W))
        self.register_buffer('vx', torch.zeros(1, 1, self.H, self.W))
        self.register_buffer('vy', torch.zeros(1, 1, self.H, self.W))

    @torch.no_grad()
    def reset_fields(self):
        self.p.zero_(); self.vx.zero_(); self.vy.zero_()

    def forward(self, source: Optional[torch.Tensor]) -> torch.Tensor:
        """
        note（note batch=1；note batch note）
        source: [1,1,H,W] or None
        """
        p, vx, vy = self.p, self.vx, self.vy
        c, kp, ko = self.c, self.kp, self.ko

        # div v
        if self.boundary_condition == 'periodic':
            dvx_dx = torch.roll(vx, -1, dims=2) - vx
            dvy_dy = torch.roll(vy, -1, dims=3) - vy
        else:
            dvx_dx = F.pad(vx[:, :, 1:, :] - vx[:, :, :-1, :], (0, 0, 0, 1))
            dvy_dy = F.pad(vy[:, :, :, 1:] - vy[:, :, :, :-1], (0, 1, 0, 0))

        new_p = (1.0 - self.dt * kp) * p - (c ** 2) * self.dt * (dvx_dx + dvy_dy)
        if source is not None:
            new_p = new_p + source

        # grad p
        if self.boundary_condition == 'periodic':
            dp_dx = new_p - torch.roll(new_p, 1, dims=2)
            dp_dy = new_p - torch.roll(new_p, 1, dims=3)
        else:
            dp_dx = F.pad(new_p[:, :, :-1, :] - new_p[:, :, 1:, :], (0, 0, 1, 0))
            dp_dy = F.pad(new_p[:, :, :, :-1] - new_p[:, :, :, 1:], (0, 1, 0, 0))

        new_vx = (1.0 - self.dt * ko) * vx - self.dt * dp_dx
        new_vy = (1.0 - self.dt * ko) * vy - self.dt * dp_dy

        self.p, self.vx, self.vy = new_p, new_vx, new_vy
        return self.p.squeeze(0).squeeze(0)  # [H,W]

    @torch.no_grad()
    def get_energy(self) -> float:
        ke = 0.5 * (self.vx.square().sum() + self.vy.square().sum())
        pe = 0.5 * self.p.square().sum()
        return float((ke + pe).item())


# Online processor.

class BatchOnlineBioOSSProcessor:
    """
    note：
      - note PANN note（4snote/1s stride），note push_pann_frame()
      - note FDTD note dt=0.01 note step() note，note

    note“note pann_features”，note：note step() note，
    note stride_s note push_pann_frame。
    """
    def __init__(self,
                 biooss_model: BioOSSFDTD2D,
                 pann_to_region_modulator: PANNToRegionModulator,
                 stride_s: float = 1.0):
        self.model = biooss_model
        self.mod = pann_to_region_modulator
        self.stride_steps = int(round(stride_s / self.model.dt))
        self._t = 0 # cache.
        self._device = next(self.model.parameters(), torch.tensor(0.)).device if len(list(self.model.parameters()))>0 else torch.device('cpu')

    @torch.no_grad()
    def push_pann_frame(self, pann_probs_527: torch.Tensor):
        self.mod.push_pann_frame(pann_probs_527.to('cpu'))

    @torch.no_grad()
    def step(self, pann_probs_527: Optional[torch.Tensor] = None) -> Dict:
        """
        note FDTD note。note pann_probs_527，note OLA；
        note。
        """
        if pann_probs_527 is not None:
            self.push_pann_frame(pann_probs_527)

        # Generate.
        source = self.mod.step(device=self._device)  # [1,1,H,W]
        field = self.model(source)                   # [H,W]
        energy = self.model.get_energy()

        self._t += 1
        return {'pressure_field': field.clone(),
                'energy': energy,
                't': self._t}

    @torch.no_grad()
    def reset(self):
        self.model.reset_fields()
        self.mod.n_step.zero_()


# Grid.

def build_biooss_pipeline(
    grid_h: int = 64, grid_w: int = 64,
    dt: float = 0.01,
    freq_range: Tuple[float, float] = (40.0, 80.0),
    region_rows: int = 8, region_cols: int = 8,
    kp: float = 0.02, ko: float = 0.02,
    pann_dim: int = 527
):
    model = BioOSSFDTD2D(
        grid_h=grid_h, grid_w=grid_w,
        dt=dt, freq_range=freq_range,
        region_rows=region_rows, region_cols=region_cols,
        kp=kp, ko=ko, xi_norm=math.sqrt(2.0),
        boundary_condition='periodic'
    )
    mod = PANNToRegionModulator(
        grid_h=grid_h, grid_w=grid_w,
        region_rows=region_rows, region_cols=region_cols,
        dt=dt,
        region_freqs=model.region_freqs.tolist(),
        region_masks=model.region_masks,
        pann_dim=pann_dim
    )
    proc = BatchOnlineBioOSSProcessor(model, mod, stride_s=1.0)
    return proc