from src.proposal import Proposal
import numpy as np


class t_k_means:

    def __init__(self, data, k):
        self.data = data
        self.k = k

    def fit(self, S: Proposal, newCj, Ci, Cj):
        AC = set([Ci, Cj])
        Ac_adj = set()
        AP = []
        # Ac_adj += S.get_adjacent_centers([Cj])
        # AP += S.get_affected_points([Cj]).tolist()
        S.update_centroid(Cj, newCj)

        while len(AC) != 0:
            potencial_AC = set()
            Ac_adj = S.get_adjacent_centers(AC)
            AP = S.get_affected_points(AC)
            aux = list(AC | Ac_adj)
            pot = S.update_first_second_nearest_center(AP, aux)
            potencial_AC = potencial_AC | pot
            S.update_centers(aux)
            AC = potencial_AC
            _ = S.update_first_second_nearest_center(AP, aux)

        return S
