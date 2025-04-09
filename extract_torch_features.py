# -*- coding: utf-8 -*-
"""
Script to call a feature extraction function from torch_features.py
on a list of wavefiles and store the result in h5features format

Synopsis:
    python extract_pytorch_features.py feats_type wav_list.txt target_h5f_file.features

The list of allowed feats_type is in the script below and matches
the available functions in torch_features.

wav_list.txt should be in kaldi-like format with two columns looking like: 

wav_id1 wav_path1
wav_id2 wav_path2
...
"""
import h5features
import numpy as np
import torch_features

feats = {'kaldi_mfcc': torch_features.kaldi_mfcc}

def parse_wav_list(wav_list):
    with open(wav_list, 'r') as f:
        lines = f.readlines()
        tokens = [line.strip().split(" ") for line in lines]
        return {a: b for a, b in tokens}


def extract_feats(feats_type, wav_list, target_h5f_file):
    # might be slow for large number of small waveforms
    get_feats = feats[feats_type]
    wavs = parse_wav_list(wav_list)  # wav_id: wav_path dict
    with h5features.Writer(target_h5f_file) as writer:
        N = len(wavs)
        for i, wav_id in enumerate(wavs):
            print(f"Processing wavefile {i+1}/{N}")
            features = get_feats(wavs[wav_id])
            # Convert from torch to numpy for h5py
            features = features.numpy()
            utt_ids = [wav_id]
            times = [0.0125 + 0.01*np.arange(len(features))]
            features = [features]
            out_data = h5features.Data(utt_ids, times, features, check=True)
            writer.write(out_data, 'features', append=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('feats_type', help=f"Available feats_type are {feats.keys()}")
    parser.add_argument('wav_list', help="path to wav_list.txt with wav_id wav_path" +\
                                         "columns columns (kaldi-like format)")
    parser.add_argument('target_h5f_file', help='path to target h5 features file')
    args = parser.parse_args()
    extract_feats(args.feats_type, args.wav_list, args.target_h5f_file)
        
