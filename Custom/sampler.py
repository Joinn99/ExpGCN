import torch
import numpy as np
from numba import njit

from recbole.sampler.sampler import AbstractSampler


@njit
def neg_samp_iter(pos_mat, can_mat, cnt_arr, val_mat, b_size, s_size):
    for i in range(b_size):
        val_mat[i, :] = neg_samp_assign(pos_mat[i, :], can_mat[i, :], cnt_arr[i], val_mat[i, :], s_size)
    return val_mat

@njit
def neg_samp_assign(pos_arr, can_arr, cnt, val_arr, s_size):
    i = 0
    for j in range(s_size * 2):
        val_arr[i] = can_arr[j]
        for k in range(s_size):
            if can_arr[j] == pos_arr[k]:
                i -= 1
                break
        i += 1
        if i == cnt:
            break
    return val_arr


class TagSampler(AbstractSampler):
    """:class:`SeqSampler` is used to sample negative item sequence.

        Args:
            datasets (Dataset or list of Dataset): All the dataset for each phase.
            distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
    """

    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset
        self.tag_num = dataset.num(dataset.config['TAG_FIELD']) - 1
        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.choice(self.tag_num, sample_num) + 1

    def get_used_ids(self):
        pass

    def sampling_neg(self, pos_sequence, num):
        value_list = []
        for _ in range(num):
            value_list.append(self.sample_neg_sequence(pos_sequence))
        return torch.tensor(np.concatenate(value_list))

    
    def sample_neg_sequence(self, pos_sequence):
        """For each moment, sampling one item from all the items except the one the user clicked on at that moment.

        Args:
            pos_sequence (torch.Tensor):  tag sequence of batch samples, with the shape of `(batch_size, max_seq_length)`.

        Returns:
            torch.tensor : all users' negative item history sequence.

        """
        pos_mat = pos_sequence.numpy()
        b_size, s_size = pos_sequence.shape

        val_mat = np.zeros(pos_mat.shape, dtype=np.int64)
        can_mat = self.sampling((b_size, s_size))
        cnt_arr = (pos_mat > 0).sum(axis=1)
        val_mat = neg_samp_iter(pos_mat, can_mat, cnt_arr, val_mat, b_size, s_size)
        
        return np.clip(val_mat, None, self.tag_num)
