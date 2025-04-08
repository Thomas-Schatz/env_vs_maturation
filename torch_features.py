import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import compute_deltas


def kaldi_mfcc(wav_file):
	waveform, sampling_rate = torchaudio.load(wav_file)
	mfcc = torchaudio.compliance.kaldi.mfcc(waveform, use_energy=True, energy_floor=0)
	deltas = compute_deltas(mfcc.T, win_length=7)
	deltas_deltas = compute_deltas(deltas, win_length=7)
	mfcc_deltas = torch.cat([mfcc, deltas.T, deltas_deltas.T], axis=1)
	cmn_transform = T.SlidingWindowCmn(cmn_window=300, center=True, norm_vars=False)
	mfcc_final = cmn_transform(mfcc_deltas)
	return mfcc_final
