import time

from sklearn.cluster import KMeans

from src.utils import *
from src.proposal import Proposal
from src.tolerant_kmeans import t_k_means


class I_Kmeans_minus_plus:

    def __init__(self, data, k, init='k-means++'):
        self.k = k
        self.data = data
        self.tkmeans = t_k_means(self.data, self.k)
        self.S = Proposal(data, k)
        if not init in ['k-means++', 'useful', 'random']:
            raise Exception("Initialization is not supported, supported initializations are: k-means++, useful, random")
        self.init = init

    def fit(self):
        # Instruction 1
        if self.init == 'useful':
            self.S.useful_init()
            kmeans = KMeans(n_clusters=self.k, n_init=1, init=self.S.centroids).fit(self.data)
        else:
            kmeans = KMeans(n_clusters=self.k, n_init=1, init=self.init).fit(self.data)
        start_time = time.time()
        cluster_assignment = kmeans.predict(self.data)
        distances = kmeans.transform(self.data)
        self.S.update_proposal(kmeans.cluster_centers_, cluster_assignment, distances)

        # instruction 2
        n_success = 0
        indivisible_clusters = []
        unmatchable_pairs = []  # list of pairs (s1,s2)
        irremovable_clusters = []
        while True:
            start1 = time.time()
            # instruction 3
            if len(indivisible_clusters) == self.k:
                break
            Si, gain_Si = self.S.get_max_gain(indivisible_clusters)
            # Instruction 4
            if self.S.k2_better_gain(indivisible_clusters, gain_Si):
                break
            # Instruction 5
            posible_Sjs = self.S.check_conditions(Si, gain_Si, unmatchable_pairs, irremovable_clusters)
            if len(posible_Sjs) == 0:
                break
            j, cost_Sj = self.S.get_min_cost(posible_Sjs)
            Sj = posible_Sjs[j]
            # Instruction 6
            if self.S.k2_better_cost(posible_Sjs, cost_Sj):
                indivisible_clusters.append(Si)
                continue
            # Instruction 7
            newCentroid = self.S.get_random_centroid(Si)
            end1 = time.time()
            start2 = time.time()
            # new solution
            S_ = Proposal(self.data, self.k, self.S.centroids, self.S.nearest_centers, self.S.second_nearest_centers)
            S_ = self.tkmeans.fit(S_, newCentroid, Si, Sj)
            S_.recompute_distances()
            # Instruction 8
            S_res = self.S.total_SSEDM()
            newS_res = S_.total_SSEDM()
            end2 = time.time()
            # print(Si, Sj)
            # print('prev: ' + str("{:e}".format(S_res)), 'new: ' + str("{:e}".format(newS_res)))
            # print('time1 :' + str(end1 - start1), 'time2 :' + str(end2 - start2))
            if newS_res >= S_res:
                unmatchable_pairs.append((Si, Sj))
            else:
                irremovable_clusters.append(Si)
                irremovable_clusters.append(Sj)
                prev_strong_adj = self.S.get_strong_adjacents(Sj)
                indivisible_clusters += prev_strong_adj
                self.S = S_
                curr_strong_adj = list(set(self.S.get_strong_adjacents(Si) + self.S.get_strong_adjacents(Sj)))
                indivisible_clusters += curr_strong_adj
                n_success += 1
                # print('succes!!!!!!!!')
            if n_success > self.k / 2:
                break
        end_time = time.time()
        return self.S.centroids, self.S.total_SSEDM(), self.S.max_SSEDM(), (end_time - start_time)

    def predict(self, X):
        centroid_idxs = []
        for x in X:
            dists = np.linalg.norm(x - self.S.centroids)
            centroid_idx = np.argmin(dists)
            centroid_idxs.append(centroid_idx)
        return centroid_idxs

