"""System and training parameters used in the paper (Tables 1 and 2)."""
import numpy as np

FC = 3.5e9                  # carrier frequency (Hz)
C = 3e8
LAM = C / FC                # wavelength (m)
M = 32                      # base station antennas (half-wavelength ULA)
K = 4                       # single-antenna users
L = 6                       # paths per user (geometric model)
AS_DEG = 5.0                # Laplacian angular spread (deg RMS), geometric model
TS = 0.25e-3                # slot = SRS period (60 kHz SCS)
NH = 16                     # history window of the neural predictors (slots)
NAR = 64                    # history window of AR-64 (slots)
SNRP_DB = 15.0              # pilot SNR per antenna entry (dB)
S2 = 10 ** (-SNRP_DB / 10)  # pilot noise variance
B = 10                      # retained beams (fixed-budget BRACE)
HIDDEN = 32                 # GRU hidden units
TRAIN_ITERS = 3500
TRAIN_LR = 4e-3
TRAIN_BATCH = 512
F = np.fft.fft(np.eye(M)) / np.sqrt(M)   # unitary DFT (beamspace) matrix
