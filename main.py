import sys

from sklearn.cluster import KMeans
import time
import pandas as pd

from src.i_kmeans_minus_plus import I_Kmeans_minus_plus
from src.proposal import Proposal
from src.utils import read_dataset, plot_data

dataset_clusters = {'a1': 20, 'a2': 35, 'a3': 50, 's1': 15, 's2': 15, 's3': 15, 's4': 15, 'birch1': 100,
                    'iris': 3, 'letter-recognition': 9, 'musk': 2, 'statlog': 7, 'HAR': 6, 'isolet': 26,
                    'KDDCUP04Bio': 2000}

if __name__ == '__main__':
    # params
    datasets = 'synthetic'  # real, synthetic
    init = 'k-means++'  # k-means++, useful, random
    plot = False
    algorithms = ['ikmeans-+', 'kmeans', 'kmeans++']
    save_dir = None

    # arguments reading
    args = sys.argv
    for flag, arg in zip(args, args[1:]):
        if flag == '-d':
            datasets = arg
        elif flag == '-s':
            save_dir = arg
        elif flag == '-p':
            plot = True
        elif flag == '-i':
            init = arg

    # check exceptions
    if not init in ['k-means++', 'useful', 'random']:
        raise Exception('the -i parameters must be in [k-means++, useful, random]')
    if plot and datasets == "real":
        raise Exception("plots are only enabled for the synthetic data")
    exp = datasets
    if datasets == "synthetic":
        datasets = ['a1', 'a2', 'a3', 's1', 's2', 's3', 's4', 'birch1']
    elif datasets == "real":
        datasets = ['iris', 'letter-recognition', 'musk', 'statlog', 'HAR', 'isolet']
    else:
        raise Exception("the -d parameters must be in [synthetic, real]")

    results = {'1KM': [], '1KM++': [], '1IKM': [], '2KM': [], '2KM++': [], '2IKM': [], '3KM': [], '3KM++': [],
               '3IKM': []}

    for dataset in datasets:
        for algorithm in algorithms:
            # read dataset
            if dataset in ['KDDCUP04Bio', 'isolet', 'HAR', 'statlog', 'musk', 'letter-recognition', 'iris']:
                data = read_dataset('data/real/' + dataset + '.data')
            else:
                data = read_dataset('data/synthetic/' + dataset + '.txt')

            print("Working on: " + algorithm + ', ' + dataset + '...')
            # clustering
            if algorithm == 'ikmeans-+':
                alg = I_Kmeans_minus_plus(data, dataset_clusters[dataset], init)
                centroids, ssedm, max_ssedm, time_ = alg.fit()

            elif algorithm == 'kmeans++':
                s_time = time.time()
                alg = KMeans(dataset_clusters[dataset], n_init=1, init='k-means++').fit(data)
                centroids = alg.cluster_centers_
                cluster_assignment = alg.predict(data)
                distances = alg.transform(data)
                S = Proposal(data, dataset_clusters[dataset])
                S.update_proposal(centroids, cluster_assignment, distances)
                ssedm, max_ssedm = S.results()
                e_time = time.time()
                time_ = e_time - s_time
            else:
                s_time = time.time()
                alg = KMeans(dataset_clusters[dataset], n_init=1, init='random').fit(data)
                centroids = alg.cluster_centers_
                cluster_assignment = alg.predict(data)
                distances = alg.transform(data)
                S = Proposal(data, dataset_clusters[dataset])
                S.update_proposal(centroids, cluster_assignment, distances)
                ssedm, max_ssedm = S.results()
                e_time = time.time()
                time_ = e_time - s_time

            # print iteration methods
            print('##############################################################################')
            print(dataset, algorithm)
            print('RESULTS: ')
            print('- SSEDM: ', str("{:e}".format(ssedm)))
            print('- Maximum of partial SSEDMs: ', str("{:e}".format(max_ssedm)))
            print('- Time(s): ', str(time_))
            print('##############################################################################')

            if not save_dir is None:
                if algorithm == 'kmeans++':
                    results['1KM++'].append(max_ssedm)
                    results['2KM++'].append(ssedm)
                    results['3KM++'].append(round(time_, 2))
                elif algorithm == 'ikmeans-+':
                    results['1IKM'].append(max_ssedm)
                    results['2IKM'].append(ssedm)
                    results['3IKM'].append(round(time_, 2))
                else:
                    results['1KM'].append(max_ssedm)
                    results['2KM'].append(ssedm)
                    results['3KM'].append(round(time_, 2))
            # plots
            if plot:
                plot_data(data, centroids, 'results/' + dataset + '_' + algorithm)
    # save results
    if not save_dir is None:
        df = pd.DataFrame([],
                          columns=['KM_max', 'KM++_max', 'IKM_max', 'KM_SSEDM', 'KM++_SSEDM', 'IKM_SSEDM', 'KM_time', 'KM++_time', 'IKM_time'])
        df['KM_max'] = results['1KM']
        df['IKM_max'] = results['1IKM']
        df['KM++_max'] = results['1KM++']

        df['KM_SSEDM'] = results['2KM']
        df['IKM_SSEDM'] = results['2IKM']
        df['KM++_SSEDM'] = results['2KM++']

        df['KM_time'] = results['3KM']
        df['IKM_time'] = results['3IKM']
        df['KM++_time'] = results['3KM++']
        df.to_excel(save_dir + '/' + exp + '.xlsx')
