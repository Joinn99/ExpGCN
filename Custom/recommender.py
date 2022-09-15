from scipy.sparse import coo
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType

class TagRecommender(GeneralRecommender):
    """This is a abstract tagrecommender implemented from general recommender. All tag recommender model should implement this class.
    """
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(TagRecommender, self).__init__(config, dataset)

        # load parameters info

        # load dataset info
        self.TAG_ID = config['TAG_FIELD']
        self.NEG_TAG_ID = config['NEG_PREFIX'] + self.TAG_ID

        self.n_tags = dataset.num(self.TAG_ID)
    
    def tag_mask(self, tag_e):
        return torch.greater(tag_e, 0)

    def get_tag_embedding(self, tag):
        tag_e = self.tag_embedding(tag)
        return tag_e

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def tag_predict(self, interaction):

        raise NotImplementedError

class TagSampleRecommender(TagRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.user_score = dataset.get_preload_weight(self.USER_ID).to(self.device)
        self.item_score = dataset.get_preload_weight(self.ITEM_ID).to(self.device)


