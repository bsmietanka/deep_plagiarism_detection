from torch import Tensor

def last_timestep(unpacked_seq: Tensor, seq_lengths: Tensor):
        # Index of the last output for each sequence.
        lengths = seq_lengths.to(unpacked_seq.device)
        idx = (lengths - 1).view(-1, 1).expand(unpacked_seq.size(0),
                                               unpacked_seq.size(2)).unsqueeze(1)
        return unpacked_seq.gather(1, idx).squeeze()
