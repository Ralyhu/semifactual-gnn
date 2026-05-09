from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import glob
import numpy as np
import json
import pickle

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj, to_undirected
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected

from torch.utils.data import random_split
import torch
import torch.nn.functional as F

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils import negative_sampling, sort_edge_index, to_dense_adj
import random
import math


class MutagenicityNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'MutagenicityNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(MutagenicityNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes, force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class ProteinsNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'ProteinsNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(ProteinsNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class IMDBNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'IMDBNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='IMDB-BINARY', pre_transform=IMDBPreTransform())
        self.graph_count = len(self.original_graphs)

        super(IMDBNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class AIDSNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'AIDSNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')

        self.original_graphs = TUDataset(root=root, name='AIDS', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(AIDSNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def sample_negative_edges(graph, num_samples):
        random.seed(0)
        new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes,
                                      force_undirected=True)
        return new_edges

    def noise_graph(self, graph, num_samples):
        new_edges = self.sample_negative_edges(graph, num_samples)
        new_edge_index = torch.hstack([graph.edge_index, new_edges])
        new_edge_index = sort_edge_index(new_edge_index)
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @staticmethod
    def read_file(folder, prefix, name):
        file_path = os.path.join(folder, prefix + f'_{name}.txt')
        return np.genfromtxt(file_path, dtype=np.int64)

    @staticmethod
    def read_sentigraph_data(folder: str, prefix: str):
        txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
        json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
        txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
        json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
        names = txt_names + json_names

        with open(os.path.join(folder, prefix + "_node_features.pkl"), 'rb') as f:
            x: np.array = pickle.load(f)
        x: torch.FloatTensor = torch.from_numpy(x)
        edge_index: np.array = SentiGraphDataset.read_file(folder, prefix, 'edge_index')
        edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
        batch: np.array = SentiGraphDataset.read_file(folder, prefix, 'node_indicator') - 1  # from zero
        y: np.array = SentiGraphDataset.read_file(folder, prefix, 'graph_labels')
        y: torch.tensor = torch.tensor(y, dtype=torch.long)

        supplement = dict()
        if 'split_indices' in names:
            split_indices: np.array = SentiGraphDataset.read_file(folder, prefix, 'split_indices')
            split_indices = torch.tensor(split_indices, dtype=torch.long)
            supplement['split_indices'] = split_indices
        if 'sentence_tokens' in names:
            with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
                sentence_tokens: dict = json.load(f)
            supplement['sentence_tokens'] = sentence_tokens

        data = Data(x=x, edge_index=edge_index, y=y)
        data, slices = split(data, batch)

        return data, slices, supplement

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement = SentiGraphDataset.read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class MutagenicityFeatureNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'MutagenicityFeatureNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(MutagenicityFeatureNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def process(self):
        raise NotImplementedError

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class MutagFeatureNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'MutagenicityFeatureNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='MUTAG', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(MutagFeatureNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def process(self):
        raise NotImplementedError

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class ProteinsFeatureNoisy(Dataset):
    def __init__(self, root, noise, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.noise = noise
        self.name = f'ProteinsFeatureNoisy{noise}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(ProteinsFeatureNoisy, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    @staticmethod
    def perturb_features(graph, num_samples):
        np.random.seed(0)
        perturbed_x = graph.x.clone().detach().numpy()
        # sample 50% nodes randomly chosen
        chosen_idx = np.random.randint(0, graph.x.shape[0], size=int(0.5 * graph.x.shape[0]))

        for idx in chosen_idx:
            # for sampled nodes generate perturbation mark where num_samples% features are perturbed
            mask = np.random.choice([0, 1], size=graph.x.shape[1], p=[1 - num_samples / 100, num_samples / 100])
            perb_idx = np.where(mask == 1)[0]
            # perturb the features
            # feature-informed perb
            for id in perb_idx:
                mu, sigma = np.mean(perturbed_x[:, id]), np.std(perturbed_x[:, id])
                delta = np.random.uniform(-0.1, 0.1, 1)
                perturbed_x[idx][id] = perturbed_x[idx][id] + delta * sigma
                # completely random perb
            # mu, sigma = 0, 2 # mean and standard deviation
            # noise = np.random.normal(mu, sigma, perb_idx.shape[0])
            # for i, perb in enumerate(noise):
            #     perturbed_x[idx][perb_idx[i]] = perturbed_x[idx][perb_idx[i]] + perb 
        # return perturbed features
        return torch.tensor(perturbed_x)

    def noise_graph(self, graph, num_samples):
        perturbed_x = self.perturb_features(graph, num_samples)
        data = Data(edge_index=graph.edge_index.clone(),
                    x=perturbed_x.clone(),
                    y=graph.y.clone())
        return data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.noise)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class MutagenicityTopologyAdversarialAttack(Dataset):
    def __init__(self, root, flip_count, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.flip_count = flip_count
        self.name = f'MutagenicityTopologyAdversarialAttack{flip_count}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(MutagenicityTopologyAdversarialAttack, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def random_sample_flip(self, graph, flip_count):
        random.seed(0)
        np.random.seed(0)
        edges_to_flip = set()
        while len(edges_to_flip) < flip_count:
            patience = 100
            while patience > 0:
                all_nodes = range(graph.num_nodes)
                allowed_nodes = all_nodes
                u = np.random.choice(allowed_nodes, replace=False, )
                v = np.random.choice(allowed_nodes, replace=False, )
                if u == v:
                    patience -= 1
                    continue
                u, v = min(u, v), max(u, v)
                edges_to_flip.add((u, v))
                break
            if patience < 0:
                pass
        return edges_to_flip

    def flip_edges(self, graph, edges_to_flip):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze()
        for u, v in edges_to_flip:
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
            else:
                adj[u, v] = 1
                adj[v, u] = 1
        new_edge_index = dense_to_sparse(adj)[0]
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def noise_graph(self, graph, flip_count):
        edges_to_flip = self.random_sample_flip(graph, flip_count)
        new_data = self.flip_edges(graph, edges_to_flip)
        return new_data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.flip_count)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class ProteinsTopologyAdversarialAttack(Dataset):
    def __init__(self, root, flip_count, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.flip_count = flip_count
        self.name = f'ProteinsTopologyAdversarialAttack{flip_count}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(ProteinsTopologyAdversarialAttack, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def random_sample_flip(self, graph, flip_count):
        random.seed(0)
        np.random.seed(0)
        edges_to_flip = set()
        while len(edges_to_flip) < flip_count:
            patience = 100
            while patience > 0:
                all_nodes = range(graph.num_nodes)
                allowed_nodes = all_nodes
                u = np.random.choice(allowed_nodes, replace=False, )
                v = np.random.choice(allowed_nodes, replace=False, )
                if u == v:
                    patience -= 1
                    continue
                u, v = min(u, v), max(u, v)
                edges_to_flip.add((u, v))
                break
            if patience < 0:
                pass
        return edges_to_flip

    def flip_edges(self, graph, edges_to_flip):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze()
        for u, v in edges_to_flip:
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
            else:
                adj[u, v] = 1
                adj[v, u] = 1
        new_edge_index = dense_to_sparse(adj)[0]
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def noise_graph(self, graph, flip_count):
        edges_to_flip = self.random_sample_flip(graph, flip_count)
        new_data = self.flip_edges(graph, edges_to_flip)
        return new_data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.flip_count)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class IMDBTopologyAdversarialAttack(Dataset):
    def __init__(self, root, flip_count, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.flip_count = flip_count
        self.name = f'IMDBTopologyAdversarialAttack{flip_count}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='IMDB-BINARY', pre_transform=IMDBPreTransform())
        self.graph_count = len(self.original_graphs)

        super(IMDBTopologyAdversarialAttack, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def random_sample_flip(self, graph, flip_count):
        random.seed(0)
        np.random.seed(0)
        edges_to_flip = set()
        while len(edges_to_flip) < flip_count:
            patience = 100
            while patience > 0:
                all_nodes = range(graph.num_nodes)
                allowed_nodes = all_nodes
                u = np.random.choice(allowed_nodes, replace=False, )
                v = np.random.choice(allowed_nodes, replace=False, )
                if u == v:
                    patience -= 1
                    continue
                u, v = min(u, v), max(u, v)
                edges_to_flip.add((u, v))
                break
            if patience < 0:
                pass
        return edges_to_flip

    def flip_edges(self, graph, edges_to_flip):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze()
        for u, v in edges_to_flip:
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
            else:
                adj[u, v] = 1
                adj[v, u] = 1
        new_edge_index = dense_to_sparse(adj)[0]
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def noise_graph(self, graph, flip_count):
        edges_to_flip = self.random_sample_flip(graph, flip_count)
        new_data = self.flip_edges(graph, edges_to_flip)
        return new_data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.flip_count)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class AIDSTopologyAdversarialAttack(Dataset):
    def __init__(self, root, flip_count, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.flip_count = flip_count
        self.name = f'AIDSTopologyAdversarialAttack{flip_count}'
        self.cleaned = False
        self.max_graph_size = float('inf')
        self.original_graphs = TUDataset(root=root, name='AIDS', use_node_attr=True)
        self.graph_count = len(self.original_graphs)

        super(AIDSTopologyAdversarialAttack, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 2

    def random_sample_flip(self, graph, flip_count):
        random.seed(0)
        np.random.seed(0)
        edges_to_flip = set()
        pair_count = graph.num_nodes * (graph.num_nodes - 1) // 2
        while len(edges_to_flip) < flip_count and len(edges_to_flip) < pair_count:
            patience = 100
            while patience > 0:
                all_nodes = range(graph.num_nodes)
                allowed_nodes = all_nodes
                u = np.random.choice(allowed_nodes, replace=False, )
                v = np.random.choice(allowed_nodes, replace=False, )
                if u == v:
                    patience -= 1
                    continue
                u, v = min(u, v), max(u, v)
                edges_to_flip.add((u, v))
                break
            if patience < 0:
                pass
        return edges_to_flip

    def flip_edges(self, graph, edges_to_flip):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze()
        for u, v in edges_to_flip:
            if adj[u, v] == 1:
                adj[u, v] = 0
                adj[v, u] = 0
            else:
                adj[u, v] = 1
                adj[v, u] = 1
        new_edge_index = dense_to_sparse(adj)[0]
        data = Data(edge_index=new_edge_index.clone(),
                    x=graph.x.clone(),
                    y=graph.y.clone())
        return data

    def noise_graph(self, graph, flip_count):
        edges_to_flip = self.random_sample_flip(graph, flip_count)
        new_data = self.flip_edges(graph, edges_to_flip)
        return new_data

    def process(self):
        for i, graph in enumerate(self.original_graphs):
            data = self.noise_graph(graph, self.flip_count)
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    gen = torch.Generator().manual_seed(0)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size
    splits = random_split(data, lengths=[train_size, val_size, test_size], generator=gen)
    return splits, [split.indices for split in splits]


def split_data_equally(data, num_splits=5):
    gen = torch.Generator().manual_seed(0)
    number = len(data)
    base_quotient = number // num_splits
    remainder = number % num_splits
    numbers = [base_quotient for _ in range(num_splits)]
    for i in range(remainder):
        numbers[i] += 1
    splits = random_split(data, lengths=numbers, generator=gen)
    return splits, [split.indices for split in splits]


def sample_negative_edges(graph, num_samples):
    random.seed(0)
    new_edges = negative_sampling(graph.edge_index, num_neg_samples=num_samples * 2, num_nodes=graph.num_nodes, force_undirected=True)
    return new_edges


def noise_graph(graph, num_samples):
    new_edges = sample_negative_edges(graph, num_samples)
    new_edge_index = torch.hstack([graph.edge_index, new_edges])
    new_edge_index = sort_edge_index(new_edge_index)
    data = Data(edge_index=new_edge_index.clone(),
                x=graph.x.clone(),
                y=graph.y.clone())
    return data


def adj_from_edge_index(graph):
    if graph.edge_index.shape[1] == 0:
        adj = torch.zeros(graph.num_nodes, graph.num_nodes)
    else:
        adj = to_dense_adj(graph.edge_index, edge_attr=graph.edge_weight, max_num_nodes=graph.num_nodes)[0]
    return adj


def get_noisy_dataset_name(dataset_name, noise):
    if dataset_name == 'Mutagenicity':
        return f'MutagenicityNoisy{noise}'
    elif dataset_name == 'Proteins':
        return f'ProteinsNoisy{noise}'
    elif dataset_name == 'IMDB-B':
        return f'IMDBNoisy{noise}'
    elif dataset_name == 'AIDS':
        return f'AIDSNoisy{noise}'
    else:
        raise NotImplementedError


def get_noisy_feature_dataset_name(dataset_name, noise):
    if dataset_name == 'Proteins':
        return f'ProteinsFeatureNoisy{noise}'
    elif dataset_name == 'Mutagenicity':
        return f'MutagenicityFeatureNoisy{noise}'
    elif dataset_name == 'Mutag':
        return f'MutagFeatureNoisy{noise}'
    else:
        raise NotImplementedError


def get_topology_adversarial_attack_dataset_name(dataset_name, flip_count):
    if dataset_name == 'Mutagenicity':
        return f'MutagenicityTopologyAdversarialAttack{flip_count}'
    elif dataset_name == 'Proteins':
        return f'ProteinsTopologyAdversarialAttack{flip_count}'
    elif dataset_name == 'IMDB-B':
        return f'IMDBTopologyAdversarialAttack{flip_count}'
    elif dataset_name == 'AIDS':
        return f'AIDSTopologyAdversarialAttack{flip_count}'
    else:
        raise NotImplementedError


class IMDBPreTransform(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
        return data


class REDDITPreTransform(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=3063).to(torch.float)
        return data


def load_dataset(dataset_name, root='data/'):
    if dataset_name == 'Mutagenicity':
        data = TUDataset(root=root, name='Mutagenicity', use_node_attr=True)
    elif "MutagenicityNoisy" in dataset_name:
        noise = int(dataset_name[17:])
        data = MutagenicityNoisy(root=root, noise=noise)
    elif 'MutagenicityFeatureNoisy' in dataset_name:
        noise = int(dataset_name[24:])
        data = MutagenicityFeatureNoisy(root=root, noise=noise)
    elif 'MutagenicityTopologyAdversarialAttack' in dataset_name:
        flip = int(dataset_name[37:])
        data = MutagenicityTopologyAdversarialAttack(root=root, flip_count=flip)
    elif dataset_name == 'Mutag':
        data = TUDataset(root=root, name='MUTAG', use_node_attr=True)
    elif 'MutagFeatureNoisy' in dataset_name:
        noise = int(dataset_name[17:])
        data = MutagFeatureNoisy(root=root, noise=noise)
    elif dataset_name == 'Proteins':
        data = TUDataset(root=root, name='PROTEINS_full', use_node_attr=True)
    elif "ProteinsNoisy" in dataset_name:
        noise = int(dataset_name[13:])
        data = ProteinsNoisy(root=root, noise=noise)
    elif 'ProteinsFeatureNoisy' in dataset_name:
        noise = int(dataset_name[20:])
        data = ProteinsFeatureNoisy(root=root, noise=noise)
    elif 'ProteinsTopologyAdversarialAttack' in dataset_name:
        flip = int(dataset_name[33:])
        data = ProteinsTopologyAdversarialAttack(root=root, flip_count=flip)
    elif dataset_name == 'IMDB-B':
        data = TUDataset(root=root, name='IMDB-BINARY', pre_transform=IMDBPreTransform())
    elif 'IMDBNoisy' in dataset_name:
        noise = int(dataset_name[9:])
        data = IMDBNoisy(root=root, noise=noise)
    elif 'IMDBTopologyAdversarialAttack' in dataset_name:
        flip = int(dataset_name[29:])
        data = IMDBTopologyAdversarialAttack(root=root, flip_count=flip)
    elif dataset_name == 'AIDS':
        data = TUDataset(root=root, name='AIDS', use_node_attr=True)
    elif 'AIDSNoisy' in dataset_name:
        noise = int(dataset_name[9:])
        data = AIDSNoisy(root=root, noise=noise)
    elif 'AIDSTopologyAdversarialAttack' in dataset_name:
        flip = int(dataset_name[29:])
        data = AIDSTopologyAdversarialAttack(root=root, flip_count=flip)
    elif dataset_name == 'NCI1':
        data = TUDataset(root=root, name='NCI1', use_node_attr=True)
    elif dataset_name == 'Graph-SST2':
        data = SentiGraphDataset(root=root, name='Graph-SST2')
    elif dataset_name == 'DD':
        data = TUDataset(root=root, name='DD', use_node_attr=True)
    elif dataset_name == 'REDDIT-B':
        data = TUDataset(root=root, name='REDDIT-BINARY', pre_transform=REDDITPreTransform())
    elif dataset_name == 'ogbg_molhiv':
        data = PygGraphPropPredDataset(root=root, name='ogbg-molhiv')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} is not implemented!')

    return data


def load_explanations(dataset_name, explainer_name, gnn_type, device, run):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}.pt'
    return torch.load(path, map_location=device)


def load_explanations_test(dataset_name, explainer_name, gnn_type, device, run):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_test.pt'
    return torch.load(path, map_location=device)


def load_explanations_noisy(dataset_name, explainer_name, gnn_type, device, run, k):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_noise_{k}.pt'
    return torch.load(path, map_location=device)

def load_explanations_noisy_test(dataset_name, explainer_name, gnn_type, device, run, k):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_noise_{k}_test.pt'
    return torch.load(path, map_location=device)

def load_explanations_noisy_feature(dataset_name, explainer_name, gnn_type, device, run, k):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_feature_noise_{k}.pt'
    return torch.load(path, map_location=device)


def load_explanations_topology_adversarial(dataset_name, explainer_name, gnn_type, device, run, k):
    path = f'data/{dataset_name}/{explainer_name}/explanations_{gnn_type}_run_{run}_topology_adversarial_{k}.pt'
    return torch.load(path, map_location=device)


def select_top_k_explanations(dataset, top_k):
    top_k_dataset = []
    for graph in dataset:
        if graph.edge_index.shape[1] > 0 and not graph.edge_weight.sum().isnan().item():
            directed_edge_weight = graph.edge_weight[graph.edge_index[0] <= graph.edge_index[1]]
            directed_edge_index = graph.edge_index[:, graph.edge_index[0] <= graph.edge_index[1]]
            idx = directed_edge_weight >= directed_edge_weight.topk(min(top_k, directed_edge_weight.shape[0]))[0][-1]
            directed_edge_index = directed_edge_index[:, idx]
            new_data = Data(
                edge_index=directed_edge_index.clone(),
                x=graph.x.clone()
            )
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)
            new_data.y = graph.y.clone()
            top_k_dataset.append(new_data)
        else:
            top_k_dataset.append(graph)
    return top_k_dataset


def remove_top_k_explanations(dataset, top_k):
    top_k_dataset = []
    for i, graph in enumerate(dataset):
        num_edges = int(graph.edge_index.shape[1] / 2)
        if num_edges <= top_k:  # we keep at least one edge on each graph
            top_k = num_edges - 1
        if graph.edge_index.shape[1] > 0 and not graph.edge_weight.sum().isnan().item() and not top_k == 0:
            directed_edge_weight = graph.edge_weight[graph.edge_index[0] <= graph.edge_index[1]]
            directed_edge_index = graph.edge_index[:, graph.edge_index[0] <= graph.edge_index[1]]
            idx = directed_edge_weight < directed_edge_weight.topk(min(top_k, directed_edge_weight.shape[0]))[0][-1]
            directed_edge_index = directed_edge_index[:, idx]
            new_data = Data(
                edge_index=directed_edge_index.clone(),
                x=graph.x.clone()
            )
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)
            new_data.y = graph.y.clone()
            if new_data.edge_index.shape[1] == 0:  # get random edge from the original graph with the least edge weight
                idx = graph.edge_weight == graph.edge_weight.min()
                np.random.seed(i)
                random_idx = np.random.choice(idx.nonzero().squeeze().tolist())
                new_data.edge_index = graph.edge_index[:, random_idx:random_idx + 1].clone()
                new_data.x = graph.x.clone()
                new_data = ToUndirected()(new_data)  # TODO: this duplicates y variable as well, why?
                new_data.y = graph.y.clone()
                new_data = RemoveIsolatedNodes()(new_data)
                top_k_dataset.append(new_data)
            else:
                top_k_dataset.append(new_data)
        else:
            new_data = Data(
                edge_index=graph.edge_index.clone(),
                x=graph.x.clone()
            )
            new_data = ToUndirected()(new_data)
            new_data = RemoveIsolatedNodes()(new_data)
            new_data.y = graph.y.clone()
            top_k_dataset.append(new_data)
    return top_k_dataset


def sample_subsets(indices, dataset, num_samples=5):
    seeds = [1, 3, 5, 7, 9]
    subsets = []
    for seed in seeds:
        random.seed(seed)
        subsets.append(np.random.choice(indices, size=int(0.2 * len(indices)), replace=False))
    pickle.dump(subsets, open(f'data/{dataset}/test_subsets.pkl', 'wb'))
    return subsets

def istoskip(graph, perc):
    # Calculate number of unique undirected edges
    num_unique_undirected_edges = len(set(tuple(sorted((u, v))) for u, v in graph.edge_index.t().tolist()))

    # Calculate k based on percentage
    k = int(math.floor(num_unique_undirected_edges * perc / 100.0))

    # Early return if no edges will be added or removed
    if perc == 0 or graph.num_nodes <= 1 or k == 0:
        return True
    return False

def perturb_graphs_semifactuals(graph: Data, perc: int, verbose: bool = True, seed: int = None):
    """
    Perturbs a graph for semifactual explanation analysis based on unique undirected edges.

    Samples edges to be notionally 'removed' and 'added' based on a percentage
    of the *unique undirected* edges. Creates sets for actual additions/removals
    and potential future additions/removals. Uses efficient sampling for missing edges.

    Args:
        graph (Data): The input graph object (torch_geometric.data.Data).
                      Assumed to contain undirected edges (potentially duplicated).
        perc (int): The percentage of *unique undirected* edges to sample for each category.
        verbose (bool, optional): If True, print detailed execution information. Defaults to True.

    Returns:
        Data: A new graph object with modified edge_index and additional attributes:
              - edge_index: Original edges minus 'edges_removed' plus 'edges_added' (directed).
              - edges_to_remove: Unique edges sampled from original, marked for potential removal (directed).
              - edges_to_add: Potential edges sampled from missing, marked for potential addition (directed).
              - edges_removed: Unique edges sampled from original and actually removed (directed).
              - edges_added: Unique edges sampled from missing and actually added (directed).
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    device = graph.edge_index.device
    num_nodes = graph.num_nodes
    original_edges = graph.edge_index # Shape [2, num_total_edges]
    num_total_original_edges = original_edges.size(1)

    # --- Calculate Unique Undirected Edges ---
    original_unique_undirected_set = set()
    original_unique_undirected_list_directed_pairs = {} # Map unique tuple -> list of original indices
    original_directed_kept_indices = list(range(num_total_original_edges)) # Start assuming all kept

    if num_total_original_edges > 0:
        for i in range(num_total_original_edges):
            u, v = original_edges[:, i].tolist()
            edge_tuple = tuple(sorted((u, v)))
            original_unique_undirected_set.add(edge_tuple)
            if edge_tuple not in original_unique_undirected_list_directed_pairs:
                original_unique_undirected_list_directed_pairs[edge_tuple] = []
            original_unique_undirected_list_directed_pairs[edge_tuple].append(i)

    num_unique_undirected_edges = len(original_unique_undirected_set)
    list_of_unique_undirected_tuples = sorted(list(original_unique_undirected_set))

    if verbose:
        print(f"--- Perturbing graph with {num_nodes} nodes ---")
        print(f"Total original edge entries (directed): {num_total_original_edges}")
        print(f"Unique undirected original edges: {num_unique_undirected_edges}")
        print(f"Unique undirected tuples: {list_of_unique_undirected_tuples}")
        print(f"Perturbation percentage (perc): {perc}% (applied to unique undirected edges)")

    # --- Calculate k based on Unique Undirected Edges ---
    if num_unique_undirected_edges == 0:
        k = 0
        if verbose: print("Graph has no unique undirected edges. k=0.")
    else:
        k = int(math.floor(num_unique_undirected_edges * perc / 100.0))
        if verbose: print(f"Calculated k = {k} (target unique edges per category)")

    # --- Sample Unique Existing Edges for Removal Categories ---
    edges_removed_final_undirected = torch.empty((2, 0), dtype=torch.long, device=device)
    edges_to_remove_final_undirected = torch.empty((2, 0), dtype=torch.long, device=device)
    kept_original_edges_final_directed = original_edges.clone() # Start with all, will remove from this

    num_existing_to_sample = min(2 * k, num_unique_undirected_edges)
    if verbose: print(f"\n--- Sampling Existing Edges ---")
    if verbose: print(f"Target unique existing edges to sample (for removed/to_remove): {num_existing_to_sample} (min(2*k, num_unique))")

    if num_existing_to_sample > 0:
        # Sample indices from the list of unique tuples
        perm_unique_existing = torch.randperm(num_unique_undirected_edges, device=device)
        sampled_unique_indices = perm_unique_existing[:num_existing_to_sample]

        k_existing_removed = num_existing_to_sample // 2
        k_existing_to_remove = num_existing_to_sample - k_existing_removed

        removed_unique_indices = sampled_unique_indices[:k_existing_removed]
        to_remove_unique_indices = sampled_unique_indices[k_existing_removed:]

        removed_unique_tuples = {list_of_unique_undirected_tuples[i] for i in removed_unique_indices.tolist()}
        to_remove_unique_tuples = {list_of_unique_undirected_tuples[i] for i in to_remove_unique_indices.tolist()}

        if verbose:
            print(f"Sampled {len(removed_unique_tuples)} unique tuples for REMOVED.")
            print(f"  - REMOVED unique tuples: {sorted(list(removed_unique_tuples))}")
            print(f"Sampled {len(to_remove_unique_tuples)} unique tuples for TO_REMOVE.")
            print(f"  - TO_REMOVE unique tuples: {sorted(list(to_remove_unique_tuples))}")

        # Identify the actual *directed* indices to remove or mark
        final_removed_directed_indices = []
        final_to_remove_directed_indices = []
        final_kept_directed_indices = []

        for i in range(num_total_original_edges):
             u, v = original_edges[:, i].tolist()
             edge_tuple = tuple(sorted((u, v)))
             if edge_tuple in removed_unique_tuples:
                 final_removed_directed_indices.append(i)
             elif edge_tuple in to_remove_unique_tuples:
                 final_to_remove_directed_indices.append(i)
             else:
                 final_kept_directed_indices.append(i)

        # Create tensors for the sampled directed edges
        edges_removed_directed = original_edges[:, final_removed_directed_indices]
        edges_to_remove_directed = original_edges[:, final_to_remove_directed_indices]
        kept_original_edges_final_directed = original_edges[:, final_kept_directed_indices]

        # Store the final *undirected* representation for the graph attributes
        # Convert sampled unique tuples back to tensor format [2, N]
        if removed_unique_tuples:
             edges_removed_final_undirected = torch.tensor(list(removed_unique_tuples), dtype=torch.long, device=device).t()
        if to_remove_unique_tuples:
             edges_to_remove_final_undirected = torch.tensor(list(to_remove_unique_tuples), dtype=torch.long, device=device).t()

        if verbose:
             print(f"Identified {edges_removed_directed.size(1)} directed edges corresponding to REMOVED tuples.")
             print(f"Identified {edges_to_remove_directed.size(1)} directed edges corresponding to TO_REMOVE tuples.")
             print(f"Keeping {kept_original_edges_final_directed.size(1)} original directed edges.")

    else:
        if verbose: print("k_existing is 0 or no unique edges. No existing edges sampled for removal/to_remove.")
        # All original edges are kept (already initialized)
        # edges_removed_final_undirected and edges_to_remove_final_undirected remain empty


    # --- Identify and Sample Missing Edges Efficiently ---
    if verbose: print("\n--- Sampling Missing Edges ---")
    num_possible_edges = num_nodes * (num_nodes - 1) // 2 if num_nodes > 1 else 0
    num_missing = num_possible_edges - num_unique_undirected_edges
    if verbose: print(f"Possible unique undirected edges: {num_possible_edges}. Actual missing unique edges: {num_missing}")

    # Determine how many missing edges to sample (up to 2*k, limited by availability)
    target_missing_samples = 2 * k # Base target on corrected k
    num_missing_to_sample = min(target_missing_samples, num_missing)
    if verbose: print(f"Target missing unique edges to sample (for added/to_add): {num_missing_to_sample} (min(2*k, num_missing))")

    k_added = num_missing_to_sample // 2
    k_to_add = num_missing_to_sample - k_added
    if verbose: print(f"Will sample {k_added} unique missing for ADDED and {k_to_add} for TO_ADD.")

    # Tensors to store the *unique undirected* sampled missing edges
    edges_added_final_undirected = torch.empty((2, 0), dtype=torch.long, device=device)
    edges_to_add_final_undirected = torch.empty((2, 0), dtype=torch.long, device=device)

    if num_missing_to_sample > 0 and num_nodes > 1:
        sampled_missing_edges_list = [] # Store as [(u,v)] where u<v
        sampled_missing_set = set()
        max_attempts = max(num_missing_to_sample * 50, 500) # Increased attempts
        attempts = 0
        if verbose: print(f"Starting sampling loop for missing edges with max_attempts={max_attempts}")

        while len(sampled_missing_edges_list) < num_missing_to_sample and attempts < max_attempts:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u == v:
                attempts += 1
                continue

            edge_tuple = tuple(sorted((u, v)))

            if edge_tuple not in original_unique_undirected_set and edge_tuple not in sampled_missing_set:
                sampled_missing_set.add(edge_tuple)
                sampled_missing_edges_list.append(list(edge_tuple)) # Store as [u,v] pair
                # if verbose: print(f"Found missing edge: {edge_tuple}") # Can be very verbose
            attempts += 1

        if verbose: print(f"Finished sampling loop after {attempts} attempts. Found {len(sampled_missing_edges_list)} unique missing edges.")
        if len(sampled_missing_edges_list) < num_missing_to_sample:
            if verbose:
                print(f"Warning: Sampled only {len(sampled_missing_edges_list)} / {num_missing_to_sample} missing edges "
                      f"within {max_attempts} attempts. Adjusting sample counts.")
            num_missing_to_sample = len(sampled_missing_edges_list)
            k_added = num_missing_to_sample // 2
            k_to_add = num_missing_to_sample - k_added
            if verbose: print(f"Adjusted counts: k_added={k_added}, k_to_add={k_to_add}")

        if num_missing_to_sample > 0:
            # Convert list of pairs to tensor [N, 2]
            all_sampled_missing_pairs = torch.tensor(sampled_missing_edges_list, dtype=torch.long, device=device)
            # Shuffle indices
            perm_missing = torch.randperm(num_missing_to_sample, device=device)
            all_sampled_missing_shuffled_pairs = all_sampled_missing_pairs[perm_missing]

            # Split and transpose to get shape [2, N]
            edges_added_final_undirected = all_sampled_missing_shuffled_pairs[:k_added].t()
            edges_to_add_final_undirected = all_sampled_missing_shuffled_pairs[k_added:].t()

            if verbose:
                print(f"Assigned {edges_added_final_undirected.size(1)} unique edges to ADDED.")
                print(f"  - ADDED unique tuples: {edges_added_final_undirected.t().tolist()}")
                print(f"Assigned {edges_to_add_final_undirected.size(1)} unique edges to TO_ADD.")
                print(f"  - TO_ADD unique tuples: {edges_to_add_final_undirected.t().tolist()}")
    else:
        if verbose: print("Number of missing edges to sample is 0 or graph has <= 1 node. No edges added.")


    # --- Construct the New Graph ---
    if verbose: print("\n--- Constructing New Graph ---")
    # Combine the *kept original directed edges*, the *to_remove edges* (these should be kept in the graph), 
    # and the *directed versions of the newly added unique edges*
    
    # First, get edges_to_remove in directed form
    if edges_to_remove_final_undirected.numel() > 0:
        to_remove_directed_edges = to_undirected(edges_to_remove_final_undirected, num_nodes=num_nodes)
        if verbose: print(f"Created {to_remove_directed_edges.size(1)} directed edges from {edges_to_remove_final_undirected.size(1)} TO_REMOVE unique tuples.")
    else:
        to_remove_directed_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        if verbose: print("No TO_REMOVE unique edges to convert to directed.")
    
    # Then, get added edges in directed form
    if edges_added_final_undirected.numel() > 0:
        added_directed_edges = to_undirected(edges_added_final_undirected, num_nodes=num_nodes)
        if verbose: print(f"Created {added_directed_edges.size(1)} directed edges from {edges_added_final_undirected.size(1)} ADDED unique tuples.")
    else:
        added_directed_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        if verbose: print("No ADDED unique edges to convert to directed.")

    # Important: combine kept edges AND to_remove edges (NOT just kept edges)
    if verbose: print(f"Combining {kept_original_edges_final_directed.size(1)} kept edges, {to_remove_directed_edges.size(1)} to_remove edges, and {added_directed_edges.size(1)} added edges.")
    combined_directed_edges = torch.cat([kept_original_edges_final_directed, to_remove_directed_edges, added_directed_edges], dim=1)

    # Ensure the final edge_index is undirected and contains no duplicates.
    # to_undirected handles sorting, adding reverse edges, and removing duplicates.
    if combined_directed_edges.numel() > 0:
        final_edge_index = to_undirected(combined_directed_edges, num_nodes=num_nodes)
        if verbose:
            print(f"Final edge_index size after to_undirected: {final_edge_index.size()}")
            print(f"Final edge_index (sorted, unique, undirected pairs):\n{final_edge_index.t().tolist()}")
    else:
        final_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        if verbose: print("Created new empty edge_index.")


    # Create the perturbed graph object
    # Store the *undirected* versions of the sampled sets as attributes
    perturbed_graph = Data(
        x=graph.x.clone() if graph.x is not None else None,
        edge_index=final_edge_index, # Use the properly processed edge index
        y=graph.y.clone() if hasattr(graph, 'y') and graph.y is not None else None, # not true label (dummy one copied from original graph)
        num_nodes=num_nodes,
        # Store undirected unique pairs for the attributes
        edges_to_remove=to_undirected(edges_to_remove_final_undirected, num_nodes=num_nodes),
        edges_to_add=to_undirected(edges_to_add_final_undirected, num_nodes=num_nodes),
        edges_removed=to_undirected(edges_removed_final_undirected, num_nodes=num_nodes),
        edges_added=to_undirected(edges_added_final_undirected, num_nodes=num_nodes)
    )
    if verbose:
        print(f"\nCreated perturbed_graph object.")
        print(f"  - Final edge_index size (directed): {perturbed_graph.edge_index.size()}")
        print(f"  - edges_to_remove size (directed): {perturbed_graph.edges_to_remove.size()}")
        print(f"  - edges_to_add size (directed): {perturbed_graph.edges_to_add.size()}")
        print(f"  - edges_removed size (directed): {perturbed_graph.edges_removed.size()}")
        print(f"  - edges_added size (directed): {perturbed_graph.edges_added.size()}")
        # print the final attribute edge lists 
        print(f"  - edges_to_remove list: {perturbed_graph.edges_to_remove.t().tolist()}")
        print(f"  - edges_to_add list: {perturbed_graph.edges_to_add.t().tolist()}")
        print(f"  - edges_removed list: {perturbed_graph.edges_removed.t().tolist()}")
        print(f"  - edges_added list: {perturbed_graph.edges_added.t().tolist()}")
        


    # Add other attributes if they exist in the original graph
    if verbose: print("\nCopying other attributes...")
    copied_attrs = []
    for key, value in graph:
        if key not in ['x', 'edge_index', 'y', 'num_nodes', 'edges_to_remove', 'edges_to_add', 'edges_removed', 'edges_added'] and \
           not callable(value) and key not in perturbed_graph:
            setattr(perturbed_graph, key, value.clone() if torch.is_tensor(value) else value)
            copied_attrs.append(key)
    if verbose: print(f"Copied attributes: {copied_attrs if copied_attrs else 'None'}")

    if verbose: print(f"--- Finished perturbing graph. Returning perturbed_graph. ---")
    return perturbed_graph


if __name__ == '__main__':
    # generate datasets
    for dataset_name in ['Mutagenicity', 'Proteins', 'Mutag', 'AIDS', 'NCI1', 'Graph-SST2']:
        dataset = load_dataset(dataset_name)