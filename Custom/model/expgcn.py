'''
    ExpGCN
    author: Tianjun Wei (tjwei2-c@my.cityu.edu.hk)
'''

import torch
import numpy as np
import scipy.sparse as sp

from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole.model.init import xavier_uniform_initialization

from Custom.loss import MaskedBPRLoss
from Custom.recommender import TagSampleRecommender

class ExpGCN(TagSampleRecommender):
    r"""
    ExpGCN is a model for joint task of item recommendation and explanation ranking.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ExpGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.m_layers = config['m_layers']
        self.tag_weight = config['tag_weight']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_size)
        self.tag_embedding = torch.nn.Embedding(num_embeddings=self.n_tags, embedding_dim=self.embedding_size)
        self.mf_loss = BPRLoss()
        self.tag_loss = MaskedBPRLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_user_ua = None
        self.restore_tag_ua = None
        self.restore_item_ia = None
        self.restore_tag_ia = None

        # generate intermediate data
        self.ui_adj_matrix = self.get_norm_adj_mat(self.n_users, self.n_items, self.interaction_matrix).to(self.device)
        self.ua_adj_matrix = self.get_norm_adj_mat(self.n_users, self.n_tags, self.user_score).to(self.device)
        self.ia_adj_matrix = self.get_norm_adj_mat(self.n_items, self.n_tags, self.item_score).to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e',
                                     'restore_user_ua', 'restore_tag_ua',
                                     'restore_item_ia', 'restore_tag_ia']

    def get_norm_adj_mat(self, row_num, col_num, sp_inter):
        A = sp.dok_matrix((row_num + col_num, row_num + col_num), dtype=np.float32)
        inter_M = sp_inter
        if isinstance(inter_M, torch.Tensor):
            inter_M = sp.coo_matrix((inter_M.values().cpu().numpy(), inter_M.indices().cpu().numpy()), shape=(row_num, col_num))
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + row_num), inter_M.data)) #  [1] * inter_M.nnz
        data_dict.update(dict(zip(zip(inter_M_t.row + row_num, inter_M_t.col), inter_M_t.data))) #  [1] * inter_M_t.nnz
        A._update(data_dict)
        sumArr = A.sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        D = sp.diags(np.power(diag, -0.5))
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, row_emb, col_emb):
        return torch.cat([row_emb, col_emb], dim=0)

    def split_ego_embeddings(self, row_num, col_num, emb):
        return torch.split(emb, [row_num, col_num])

    def forward(self, row_emb, col_emb, adj_mat, layers=2):
        all_embeddings = self.get_ego_embeddings(row_emb, col_emb)

        lightgcn_all_embeddings = all_embeddings + 0
        for _ in range(layers):
            all_embeddings = torch.sparse.mm(adj_mat, all_embeddings)
            lightgcn_all_embeddings += all_embeddings
        lightgcn_all_embeddings = lightgcn_all_embeddings.div(1 + layers)
        ui_u_emb, ui_i_emb = self.split_ego_embeddings(row_emb.shape[0], col_emb.shape[0], lightgcn_all_embeddings)
        return ui_u_emb, ui_i_emb

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        if self.restore_user_ua is not None or self.restore_tag_ua is not None:
            self.restore_user_ua, self.restore_tag_ua = None, None
        if self.restore_item_ia is not None or self.restore_tag_ia is not None:
            self.restore_item_ia, self.restore_tag_ia = None, None

        ua_u_emb, ua_a_emb = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        user_ua = ua_u_emb[interaction[self.USER_ID]]
        tag_ua = torch.mm(user_ua, ua_a_emb.transpose(0, 1))
        pos_tag_ua = tag_ua.gather(1, interaction[self.TAG_ID])
        neg_tag_ua = tag_ua.gather(1, interaction[self.NEG_TAG_ID])

        ia_i_emb, ia_a_emb = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)
        item_ia = ia_i_emb[interaction[self.ITEM_ID]]
        tag_ia = torch.mm(item_ia, ia_a_emb.transpose(0, 1))
        pos_tag_ia = tag_ia.gather(1, interaction[self.TAG_ID])
        neg_tag_ia = tag_ia.gather(1, interaction[self.NEG_TAG_ID])

        mask = self.tag_mask(interaction[self.TAG_ID])
        
        tag_loss = self.tag_loss(pos_tag_ua + pos_tag_ia,
                                 neg_tag_ua + neg_tag_ia,
                                 mask)

        ui_u_emb, ui_i_emb = self.forward(self.user_embedding.weight + ua_u_emb,
                                          self.item_embedding.weight + ia_i_emb,
                                          self.ui_adj_matrix, self.n_layers)
        user_ui = ui_u_emb[interaction[self.USER_ID]]
        pos_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.ITEM_ID]])
        neg_item_ui = torch.mul(user_ui, ui_i_emb[interaction[self.NEG_ITEM_ID]])

        mf_loss = self.mf_loss(pos_item_ui.sum(dim=-1), neg_item_ui.sum(dim=-1))

        loss = mf_loss + self.tag_weight * tag_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_item_e[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_ua is None or self.restore_tag_ua is None:
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.user_embedding.weight + self.restore_user_ua,
                                                                    self.item_embedding.weight + self.restore_item_ia,
                                                                    self.ui_adj_matrix, self.n_layers)

        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

    def tag_predict(self, interaction):
        if self.restore_user_ua is None or self.restore_tag_ua is None:
            self.restore_user_ua, self.restore_tag_ua = self.forward(self.user_embedding.weight, self.tag_embedding.weight, self.ua_adj_matrix, self.m_layers)
        if self.restore_item_ia is None or self.restore_tag_ia is None:
            self.restore_item_ia, self.restore_tag_ia = self.forward(self.item_embedding.weight, self.tag_embedding.weight, self.ia_adj_matrix, self.m_layers)
        
        scores = torch.matmul(self.restore_user_ua[interaction[self.USER_ID]], self.restore_tag_ua.transpose(0, 1)) + \
                 torch.matmul(self.restore_item_ia[interaction[self.ITEM_ID]], self.restore_tag_ia.transpose(0, 1))

        return scores