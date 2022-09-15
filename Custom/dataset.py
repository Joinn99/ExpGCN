import torch
from scipy.special import expit
from recbole.data.dataset import Dataset

class CustomDataset(Dataset):
    def _init_alias(self):
        if 'alias_of_tag_id' in self.config:
            self._set_alias('tag_id', self.config['alias_of_tag_id'])
        super()._init_alias()

    def build(self):
        datasets = super().build()
        train_dataset, _, _ = datasets
        len_tag_seq = train_dataset.inter_feat[train_dataset.tag_field].shape[1]
        uind = torch.stack([train_dataset.inter_feat[train_dataset.uid_field].repeat(len_tag_seq, 1).transpose(0,1).reshape(-1),
                            train_dataset.inter_feat[train_dataset.tag_field].reshape(-1)])
        uind = uind[:, uind[1] > 0]
        iind = torch.stack([train_dataset.inter_feat[train_dataset.iid_field].repeat(len_tag_seq, 1).transpose(0,1).reshape(-1),
                            train_dataset.inter_feat[train_dataset.tag_field].reshape(-1)])
        iind = iind[:, iind[1] > 0]                   
        self._preloaded_weight[train_dataset.uid_field] = torch.sparse_coo_tensor(uind, torch.ones(uind.shape[1])).coalesce()
        self._preloaded_weight[train_dataset.iid_field] = torch.sparse_coo_tensor(iind, torch.ones(iind.shape[1])).coalesce()
        return datasets

    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.tag_field = self.config['TAG_FIELD']