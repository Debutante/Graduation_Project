from base import BaseExecutor
from utils.utilities import timer, type_error_msg, value_error_msg
from configparser import ConfigParser
from os import makedirs
from os.path import dirname, exists
from scipy.io import savemat, loadmat
import numpy as np


class Evaluator(BaseExecutor):
    """A general evaluator for all datasets.

    Args:
        name (str): A name defined in base.py.
        scene (str): A mode defined in base.py.

    Attributes:
        name (BaseExecutor.Name): A name enum defined in base.py.
        scene (BaseExecutor.Scene): A scene enum defined in base.py.

    """
    def __init__(self, name, scene):
        if not isinstance(name, Evaluator.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in Evaluator.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, Evaluator.NAME_LIST))
                name = Evaluator.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [Evaluator.Name, str]))

        if not isinstance(scene, Evaluator.Scene):
            if isinstance(scene, str):
                if not scene.islower():
                    scene = scene.lower()
                if scene not in Evaluator.SCENE_LIST:
                    raise ValueError(value_error_msg('scene', scene, Evaluator.SCENE_LIST))
                scene = Evaluator.Scene(scene)
            else:
                raise TypeError(type_error_msg('scene', scene, [Evaluator.SCENE_LIST, str]))

        self.name = name
        self.scene = scene

    @staticmethod
    def compute_AP(goods, indices, len_CMC, dictionary=None):
        """Computes aP(average Precision) and CMC(Cumulative Match Characteristic curve) for every query.
        For the triangle mAP calculation, refers to http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp.

        Args:
            goods (list of np.ndarray): The correct indices of #query * #correct_indices.
            indices (list of np.ndarray): The rank arrays of #query * #gallery_without_junk_indices.
            len_CMC (int): The length of CMC: #gallery
            dictionary (dict): A mutable dictionary for adding
                            {'ap': [ap1, ap2, ...],
                            'CMC': [CMC_sum1, CMC_sum2, ...]} (Implicit Returns)

        Returns:
            tuple(aP: list, CMC: np.ndarray)

        """
        ap_list = []
        CMC_sum = np.zeros(len_CMC)
        for good, index in zip(goods, indices):
            ap = 0
            cmc = np.zeros(len_CMC)
            ngood = len(good)

            if ngood > 0:
                rows_good = np.flatnonzero(np.in1d(index, good, assume_unique=True))
                cmc[rows_good[0]:] = 1

                delta_recall = 1 / ngood  # recall(i / ngood) - old_recall((i - 1) / ngood)
                for i in range(len(rows_good)):
                    precision = float(i + 1) / (rows_good[i] + 1)
                    if rows_good[i] != 0:
                        old_precision = float(i) / rows_good[i]
                    else:
                        old_precision = 1
                    ap += delta_recall * (old_precision + precision) / 2

            ap_list.append(ap)
            CMC_sum += cmc
        CMC = CMC_sum / len(ap_list)
        if dictionary is not None:
            dictionary['ap'] = ap_list
            dictionary['CMC'] = CMC
        return np.mean(np.asarray(ap_list)).item(), CMC

    @staticmethod
    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    @staticmethod
    def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
        """
        Created on Mon Jun 26 14:46:56 2017
        @author: luohao
        Modified by Houjing Huang, 2017-12-22.
        - This version accepts distance matrix instead of raw features.
        - The difference of `/` division between python 2 and 3 is handled.
        - numpy.float16 is replaced by numpy.float32 for numerical precision.

        Modified by Zhedong Zheng, 2018-1-12.
        - replace sort with topK, which save about 30s.

        CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
        url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
        Matlab version: https://github.com/zhunzhong07/person-re-ranking

        API
        q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
        q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
        g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
        k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
        Returns:
          final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]

        """
        # The following naming, e.g. gallery_num, is different from outer scope.
        # Don't care about it.
        original_dist = np.concatenate(
            [np.concatenate([q_q_dist, q_g_dist], axis=1),
             np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
            axis=0)
        original_dist = 2. - 2 * original_dist  # change the cosine similarity metric to euclidean similarity metric
        original_dist = np.power(original_dist, 2).astype(np.float32)
        original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float32)
        # initial_rank = np.argsort(original_dist).astype(np.int32)
        # top K1+1
        initial_rank = np.argpartition(original_dist, range(1, k1 + 1))

        query_num = q_g_dist.shape[0]
        all_num = original_dist.shape[0]

        for i in range(all_num):
            # k-reciprocal neighbors
            k_reciprocal_index = Evaluator.k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_k_reciprocal_index = \
                    Evaluator.k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist

    def run(self):
        raise NotImplementedError


class Market1501Evaluator(Evaluator):
    """A specific evaluator for market1501. Junk indices are labeled as '-1' in Market1501 dataset.
        And they should be removed when evaluating.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        scene (str): A scene defined in base.py
        re_rank (bool): Re-rank(boost performance) or not.

    Attributes:
        test_path (str): Path to the dictionary of gallery and query info.
        evaluation_path (str): Path to the dictionary of evaluation indicators and rank lists.

    """
    def __init__(self, config, scene, re_rank: bool = False):
        super(Market1501Evaluator, self).__init__('market1501', scene)

        self.config = config

        self.test_path = self.config[self.name.value]['test_path'] % self.scene.value

        self.evaluation_path = self.config[self.name.value]['evaluation_path'] % self.scene.value

        if re_rank:
            self.rerank = True
        else:
            self.rerank = False

    def sort_index(self, distance, query_cam, query_label, gallery_cam, gallery_label, dictionary=None):
        """Sorts the distance array to get rank lists for every query.

        Args:
            distance (np.ndarray): The distance array of features of gallery and query,
                                an 2d float array, shape(#gallery, #query).
            query_cam (np.ndarray): The camera labels of query imgs, an 1d int array, shape(#query).
            query_label (np.ndarray): The people labels of query imgs, an 1d int array, shape(#query).
            gallery_cam (np.ndarray): The camera labels of gallery imgs, an 1d int array, shape(#gallery).
            gallery_label (np.ndarray): The people labels of gallery imgs, an 1d int array, shape(#gallery).
            dictionary (dict): A mutable dictionary for adding
                            {'index': [index_array1, index_array2, ...]} (Implicit returns).
        Returns:
            tuple(good_list, index_without_junk): Both are lists made up of indices arrays.

        """
        index_without_junk = []
        good_list = []
        junk_index1 = np.flatnonzero(gallery_label == -1)
        for k in range(len(query_label)):
            match_mask = gallery_label == query_label[k]
            # [False False True ...](ndarray)
            good_mask = gallery_cam != query_cam[k]
            # [True False True ...](ndarray)
            junk_mask = gallery_cam == query_cam[k]
            # [False True False ...](ndarray)
            good_index = np.flatnonzero(match_mask & good_mask)
            # [index3 index9 ...](ndarray)
            junk_index2 = np.flatnonzero(match_mask & junk_mask)
            # [index4 index7 ...](ndarray)
            junk_index = np.concatenate((junk_index1, junk_index2))
            index = np.argsort(distance[:, k])
            # if self.similarity == Evaluator.Similarity.COSINE:
            #     index = index[::-1]
            index_without_junk.append(np.setdiff1d(index, junk_index, assume_unique=True))
            good_list.append(good_index)
        if dictionary is not None:
            dictionary['index'] = index_without_junk
        return good_list, index_without_junk

    @timer
    def run(self):
        """
        Reads: A mat file of saved gallery and query info.

        Processes: Computes mAP and CMC for saved gallery and query info.

        Writes: A mat file of evaluation indicators and rank lists.

        """

        Market1501Evaluator.run_info(self.__class__.__name__, self.scene.value)

        result_dict = loadmat(self.test_path)
        # {
        #     'gallery_feature': [[dim1 dim2 ... in index1] [dim1 dim2 ... in index2] ...](ndarray),
        #     'gallery_label': [[class1 class1 class2 ...]](ndarray),
        #     'gallery_cam': [[cam1 cam1 cam2 ...]](ndarray),
        #     'query_feature': [[dim1 dim2 ... in index1] [dim1 dim2 ... in index2] ...](ndarray),
        #     'query_label': [[class1 class1 class2...]](ndarray),
        #     'query_cam': [[cam1 cam1 cam2 ...]](ndarray),
        #     'multi_index': [[[[index1 index2 ...]] [[index3 index4 ...]] ...]](ndarray)(if MULTI_SHOT)
        # }
        similarity = result_dict['gallery_feature'] @ result_dict['query_feature'].T
        # condition: the two operands are normalized.
        evaluation_dict = None if self.rerank else dict()
        goods, indices = self.sort_index(1 - similarity, result_dict['query_cam'][0], result_dict['query_label'][0],
                                         result_dict['gallery_cam'][0], result_dict['gallery_label'][0],
                                         evaluation_dict)
        mAP, CMC = Market1501Evaluator.compute_AP(goods, indices, similarity.shape[0], evaluation_dict)
        print('Rank@1:{:.4f}% Rank@5:{:.4f}% Rank@10:{:.4f}% mAP:{:.4f}%'.
              format(CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, mAP * 100))
        if self.rerank:
            q_g_dist = similarity.T
            # condition: the two operands are normalized.
            q_q_dist = result_dict['query_feature'] @ result_dict['query_feature'].T
            g_g_dist = result_dict['gallery_feature'] @ result_dict['gallery_feature'].T
            new_dist = Market1501Evaluator.re_ranking(q_g_dist, q_q_dist, g_g_dist).T  # gallery * query
            evaluation_dict = dict()
            goods, indices = self.sort_index(new_dist, result_dict['query_cam'][0], result_dict['query_label'][0],
                                             result_dict['gallery_cam'][0], result_dict['gallery_label'][0],
                                             evaluation_dict)
            mAP, CMC = Market1501Evaluator.compute_AP(goods, indices, new_dist.shape[0], evaluation_dict)
            print('(Re-rank) ', end='')
            print('Rank@1:{:.4f}% Rank@5:{:.4f}% Rank@10:{:.4f}% mAP:{:.4f}%'.
                  format(CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, mAP * 100))
        evaluation_dir = dirname(self.evaluation_path)
        if not exists(evaluation_dir):
            makedirs(evaluation_dir)
        savemat(self.evaluation_path, evaluation_dict)
