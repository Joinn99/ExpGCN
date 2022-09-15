import torch
from recbole.data.interaction import Interaction
from recbole.utils import InputType
from recbole.data.dataloader.general_dataloader import TrainDataLoader

class TagTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, tag_sampler, shuffle=False, subs_sampler=None):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.tag_sampler = tag_sampler
        self.subs_sampler = subs_sampler if subs_sampler is not None else None

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        super()._set_neg_sample_args(config, dataset, dl_format, neg_sample_args)
        self.tag_field = dataset.tag_field
        if self.dl_format == InputType.POINTWISE:
            self.neg_prefix = config['NEG_PREFIX']

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args['strategy'] == 'by':
            user_ids = inter_feat[self.uid_field]
            item_ids = inter_feat[self.iid_field]
            tag_ids = inter_feat[self.tag_field]
            neg_item_ids = self.sampler.sample_by_user_ids(user_ids, item_ids, self.neg_sample_num)
            neg_tag_ids = self.tag_sampler.sampling_neg(tag_ids, self.times)
            subs_item_ids = self.subs_sampler.sample_by_user_ids(user_ids, item_ids, 1) if self.subs_sampler is not None else None
            return self.sampling_func(inter_feat, neg_item_ids, neg_tag_ids, subs_item_ids)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids, tag_ids, subs_item_ids):
        if subs_item_ids is not None:
            inter_feat.update(Interaction({'subs_item_id': subs_item_ids}))
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids, self.tag_field: tag_ids})
        neg_item_feat = self.dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat
    
    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids, tag_ids, subs_item_ids):
        if subs_item_ids is not None:
            inter_feat.update(Interaction({'subs_item_id': subs_item_ids}))
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        tag_feat = Interaction({self.tag_field: tag_ids})
        tag_feat.add_prefix(self.neg_prefix)
        new_data.update(tag_feat)
        return new_data