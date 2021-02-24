from git_config import *
from model.istft import ISTFT

n_fft       = 424
hop_length  = 212
window = torch.hann_window(n_fft)
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window, center=True)
istft = ISTFT(n_fft, hop_length, window='hanning')

def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    bsum = lambda x: torch.sum(x, dim=1)
    def mSDRLoss(orig, est):
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


