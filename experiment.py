from typing import List, Tuple, Callable, Optional
import networkx as nx
import numpy as np
import random
from scipy.stats import ttest_ind as ttest
import matplotlib.pyplot as plt

class Statistics:
    @staticmethod
    def get_tau(graph):
        return 1.0 / max(np.linalg.eigvals(nx.to_numpy_matrix(graph)))

    @staticmethod
    def get_freeman(graph):
        es = [i for i in nx.eigenvector_centrality(graph).values()]
        e_max = max(es)
        return np.mean([(e_max - e) for e in es])

    @staticmethod
    def get_gini(graph):
        es = np.sort([i for i in nx.eigenvector_centrality(graph).values()])
        n = es.shape[0]
        index = np.arange(1, n + 1)
        return (np.sum(2 * index - n - 1) * es) / (n * np.sum(es))

class Experiment:
    def __init__(self, num_graphs: int, num_nodes: int, male_ratio: float,
                 dd_without_sex_ed: List[List[float], List[float]], dd_with_sex_ed: List[List[float], List[float]],
                 is_invalid_graph:Callable=lambda x: False, match_mean_degrees:bool=False,
                 cd_without_sex_ed:Optional[List[List[float], List[float]]]=None,
                 cd_with_sex_ed:Optional[List[List[float], List[float]]]=None, condom_weight:Optional[float]=None):
        """
        :param num_graphs: The number of random graphs we want to generate
        :param num_nodes: The number of nodes we want in each random graph
        :param male_ratio: The ratio of male
        :param dd_without_sex_ed: A list of two lists,
                                 the first one corresponding to the degree distribution of men without sex ed,
                                 and the second to that of women without sex ed.
        :param dd_with_sex_ed: Same with dd_without_sex_ed, but data of men and women with sex ed
        :param is_invalid_graph: A function that takes a networkx graph as input,
                                 and outs true if the graph violates some rule, false otherwise
        :param match_mean_degrees: Whether or not we want to match the mean degrees between two groups
        :param cd_without_sex_ed: Same with dd_without_sex_ed, but is the distribution of ratio of condom usage
        :param cd_with_sex_ed: Same with dd_with_sex_ed, but is the distribution of ratio of condom usage
        :param condom_weight: The weight we want to set on edges encoding sexual interaction with condom usage
        """
        if condom_weight is None:
            assert cd_with_sex_ed is None and cd_without_sex_ed is None
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.male_ratio = male_ratio
        self.dd_without_sex_ed = dd_without_sex_ed
        self.dd_with_sex_ed = dd_with_sex_ed
        self.is_invalid_graph = is_invalid_graph
        self.match_mean_degrees = match_mean_degrees
        self.cd_without_sex_ed = cd_without_sex_ed
        self.cd_with_sex_ed = cd_with_sex_ed
        self.condom_weight = condom_weight

    @staticmethod
    def __connect_spokes(graph: nx.Graph, men_spokes: List[int], women_spokes: List[int],
                         is_invalid_graph: Callable, condom_weight: Optional[float]=None):
        men_spokes_copy = [i for i in men_spokes]
        women_spokes_copy = [i for i in women_spokes]
        random.shuffle(men_spokes_copy)
        while len(men_spokes_copy) > 0 and len(len(women_spokes_copy)) > 0:
            man_id = men_spokes_copy.pop(0)
            women = [i for i in women_spokes_copy]
            random.shuffle(women)
            while len(women) > 0:
                success = False
                woman_id = women.pop(0)
                if graph.has_edge(man_id, woman_id):
                    continue
                if condom_weight is None:
                    graph.add_edge(man_id, woman_id)
                else:
                    graph.add_edge(man_id, woman_id, weight=condom_weight)
                if is_invalid_graph(graph):
                    graph.remove_edge(man_id, woman_id)
                else:
                    success = True
                    break
            if success:
                women_spokes_copy.remove(woman_id)

    @staticmethod
    def __generate_random_graph(n_men: int, n_women: int,
                                men_dd: List[float], women_dd: List[float],
                                is_invalid_graph: Callable,
                                men_cd: Optional[List[float]]=None, women_cd: Optional[List[float]]=None,
                                condom_weight: Optional[float]=None) -> nx.Graph:
        """
        :param n_men: Number of men in the graph
        :param n_women: Number of women in the graph
        :param men_dd: Degree distribution of men
        :param women_dd: Degree distribution of women
        :param is_invalid_graph: A function that tests whether a graph violates a certain rule
        :param men_cd: Degree distribution of condom usage of men
        :param women_cd: Degree distribution of condom usage of women
        :return: A networkx Graph object
        """
        graph = nx.Graph()
        node_id = 0
        men_spokes, women_spokes, men_cspokes, women_cspokes = [], [], [], []
        men_cd = men_cd if men_cd is not None else [0] * len(men_dd)
        women_cd = women_cd if women_cd is not None else [0] * len(women_dd)
        ds = [i for i in range(1,len(men_dd)+1)]
        for _ in range(n_men):
            graph.add_node(node_id)
            node_id += 1
            total_degree = np.random.choice(ds, p=men_dd)
            use_condom_prob = men_cd[total_degree - 1]
            degree_non_condom, degree_condom = 0, 0
            for _ in range(total_degree):
                choice = np.random.choice([1, 0], p=[use_condom_prob, 1 - use_condom_prob])
                if choice == 0:
                    degree_non_condom += 1
                else:
                    degree_condom += 1
            men_spokes += [node_id] * degree_non_condom
            men_cspokes += [node_id] * degree_condom
        for i in range(n_women):
            graph.add_node(node_id)
            node_id += 1
            total_degree = np.random.choice(ds, p=women_dd)
            use_condom_prob = women_cd[total_degree - 1]
            degree_non_condom, degree_condom = 0, 0
            for _ in range(total_degree):
                choice = np.random.choice([1, 0], p=[use_condom_prob, 1 - use_condom_prob])
                if choice == 0:
                    degree_non_condom += 1
                else:
                    degree_condom += 1
            women_spokes += 1
        Experiment.__connect_spokes(graph, men_spokes, women_spokes, is_invalid_graph, condom_weight)
        Experiment.__connect_spokes(graph, men_cspokes, women_cspokes, is_invalid_graph, condom_weight)
        return graph

    def __run_one_iteration(self):
        men_without_sex_ed_dd, women_without_sex_ed_dd = self.dd_without_sex_ed
        men_with_sex_ed_dd, women_with_sex_ed_dd = self.dd_with_sex_ed
        men_without_sex_ed_cd, women_without_sex_ed_cd = self.cd_without_sex_ed
        men_with_sex_ed_cd, women_with_sex_ed_cd = self.cd_with_sex_ed
        num_men = round(self.num_nodes * self.male_ratio)
        num_women = self.num_nodes - num_men
        graph_without_sex_ed = self.__generate_random_graph(num_men, num_women,
                                                            men_without_sex_ed_dd, women_without_sex_ed_dd,
                                                            self.is_invalid_graph,
                                                            men_without_sex_ed_cd, women_without_sex_ed_cd, self.condom_weight)
        graph_with_sex_ed = self.__generate_random_graph(num_men, num_women,
                                                         men_with_sex_ed_dd, women_with_sex_ed_dd,
                                                         self.is_invalid_graph,
                                                         men_with_sex_ed_cd, women_with_sex_ed_cd, self.condom_weight)
        if self.match_mean_degrees:
            i, j = graph_without_sex_ed.number_of_edges(), graph_with_sex_ed.number_of_edges()
            diff = i - j
            if diff > 0:
                graph_without_sex_ed.remove_edges_from(random.sample(graph_without_sex_ed.edges, abs(diff)))
            elif diff < 0:
                graph_with_sex_ed.remove_edges_from(random.sample(graph_with_sex_ed.edges, abs(diff)))
        gcc_without_sex_ed = max(nx.connected_component_subgraphs(graph_without_sex_ed), key=len)
        gcc_with_sex_ed = max(nx.connected_component_subgraphs(graph_with_sex_ed), key=len)
        return {'without': {'tau': Statistics.get_tau(gcc_without_sex_ed),
                                   'gini': Statistics.get_gini(gcc_without_sex_ed),
                                   'freeman': Statistics.get_freeman(gcc_without_sex_ed),
                                   'mg': nx.average_shortest_path_length(gcc_without_sex_ed)},
                'with': {'tau': Statistics.get_tau(gcc_with_sex_ed),
                                'gini': Statistics.get_gini(gcc_with_sex_ed),
                                'freeman': Statistics.get_freeman(gcc_with_sex_ed),
                                'mg': nx.average_shortest_path_length(gcc_with_sex_ed)}
                }

    def run(self, num_iters: int):
        """
        :param num_iters: Number of iterations we want to run the experiment
        :return: A dictionary with t-test results on various metrics between the random graphs with and without sex-ed
        """
        without_taus, without_ginis, without_freemans, without_mgs = [], [], [], []
        with_taus, with_ginis, with_freemans, with_mgs = [], [], [], []
        for _ in range(num_iters):
            result = self.__run_one_iteration()
            without_taus.append(result['without']['tau'])
            without_ginis.append(result['without']['gini'])
            without_freemans.append(result['without']['freeman'])
            without_mgs.append(result['without']['mg'])
            with_taus.append(result['with']['tau'])
            with_ginis.append(result['with']['gini'])
            with_freemans.append(result['with']['freeman'])
            with_mgs.append(result['with']['mg'])
        ttest_results_tau = ttest(without_taus, with_taus)
        ttest_results_gini = ttest(without_ginis, with_ginis)
        ttest_results_freeman = ttest(without_freemans, with_freemans)
        ttest_results_mgs = ttest(without_mgs, with_mgs)
        return {'tau': ttest_results_tau, 'gini': ttest_results_gini, 'freeman': ttest_results_freeman, 'mgs': ttest_results_mgs}



