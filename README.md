# Semifactual Explanations for Graph Neural Networks (GNNs).

## Overview 

This project is developed as part of the following research paper:

G. Alfano, S. Greco, D. Mandaglio, F. Parisi, R. Shahbazian, I. Trubitsyna. "Semifactual Explanations for GNN-based Classification: Formal Foundations, Complexity and Computation" published in the proceedings of the International Joint Conference on Artificial Intelligence, 2026.


**Note:** This code is embedded in the [GNNXBench](https://github.com/idea-iitd/gnn-x-bench) library (ICLR2024), which provides a comprehensive benchmarking framework for GNN explainability methods.

## Requirements

The easiest way to install the dependencies is via [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Once you have conda installed, run this command:

```setup
conda env create -f env.yml
```

If you want to install dependencies manually, we tested our code in Python 3.9.7 using the following main dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) v1.11.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) v2.1.0
- [NetworkX](https://networkx.org/documentation/networkx-2.5/install.html) v2.7.1
- [NumPY](https://numpy.org/install/) v1.22.3

Experiments were conducted using using a double 56-core Intel(R) Xeon(R) Gold 6258R CPU, equipped with 256GB RAM and two NVIDIA GeForce RTX3090s with 24GB memory each, OS Ubuntu Linux 22.04 LTS.

## Usage

### Data installation

Every datasets except for Graph-SST2 is ready to install from PyTorch Geometric libraries. For Graph-SST2, you can download the dataset from 
[here](https://drive.google.com/file/d/1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1/view?usp=sharing) and put it in `data/` directory.

Then, you can run the following command to preprocess the data. This will preprocess every dataset (see also [GNNXBench](https://github.com/idea-iitd/gnn-x-bench)).

```setup
python source/data_utils.py
```

### Training Base GNNs

As an example, we provide pretrained GNN models for the MUTAG dataset. However, we recommend to train the GNN base models from scratch, you can run the following command:

```setup
python source/basegnn.py --dataset <dataset_name> --gnn_type gcn --runs 1
```

### Running the SEMIX method

#### Basic Usage

From the main directory run the following command:

```bash
python source/semifactual_full.py --dataset Mutag --lam 0.5 --epochs 100
```

#### Command Line Parameters

##### Required Parameters
- `--dataset`: Dataset to use for explanation
  - **Choices**: `['Mutagenicity', 'Proteins', 'Mutag', 'AIDS', 'NCI1', 'Graph-SST2', 'ogbg_molhiv']`
  - **Default**: `Mutag`

##### Model Configuration
- `--gnn_type`: Type of GNN architecture to use
  - **Choices**: `['gcn', 'gat', 'gin', 'sage']`
  - **Default**: `gcn`
- `--gnn_run`: Run number for the pre-trained GNN model
  - **Default**: `1`
- `--explainer_run`: Run number for the explainer (affects random seed)
  - **Default**: `1`

##### Training Parameters
- `--epochs`: Number of training epochs for the explainer
  - **Default**: `100`
- `--lr`: Learning rate for the explainer
  - **Default**: `0.001`
- `--lam`: Lambda hyperparameter to balance between loss terms
  - **Default**: `0.5`
- `--gam`: Margin value for the semifactual loss
  - **Default**: `0.5`

##### Explanation Parameters
- `--perturb_perc`: Percentage of edges to perturb for creating the modified graph
  - **Default**: `25`
- `--mask_thresh`: Threshold to convert relaxed adjacency matrix to binary
  - **Default**: `0.5`
- `--use_last`: Flag to use the last explanation instead of best during training
  - **Default**: `True`

##### System Parameters
- `--device`: Device to run on (GPU number or 'cpu')
  - **Default**: `"0"`

#### Output Format

The method generates two main output files:

##### 1. Semifactual Explanations (`.pt` file)
**Location**: `data/{dataset}/semifactual_full/exps_lam{lam}_lr{lr}_epochs{epochs}_gam{gam}_maskthresh{mask_thresh}_perturbperc{perturb_perc}_gnntype{gnn_type}_useLast{use_last}.pt`

**Format**: PyTorch tensor containing a list of dictionaries, where each dictionary represents one graph's explanation:
```python
{
    "graph": original_modified_graph,
    "graph_sf": semifactual_graph_with_edge_weights,
    "label": original_label,
    "pred": original_prediction_probabilities,
    "pred_sf": semifactual_prediction_probabilities
}
```

##### 2. Metrics Log (`.json` file)
**Location**: `data/{dataset}/semifactual_full/log_lam{lam}_lr{lr}_epochs{epochs}_gam{gam}_maskthresh{mask_thresh}_perturbperc{perturb_perc}_gnntype{gnn_type}_useLast{use_last}.json`

**Format**: JSON file containing comprehensive metrics and statistics:

```json
{
    "run_parameters": {
        "dataset": "Mutag",
        "lambda": 0.5,
        "learning_rate": 0.001,
        "epochs": 100,
        "gamma": 0.5,
        "mask_threshold": 0.5,
        "perturb_percentage": 25,
        "gnn_type": "gcn",
        "use_last": true
    },
    "summary_statistics": {
        "avg_irrelevance": 0.1234,
        "std_irrelevance": 0.0567,
        "avg_size": 2.3456,
        "std_size": 1.2345,
        "avg_time": 45.6789,
        "std_time": 12.3456
    },
    "irrelevances": [...],
    "sizes": [...],
    "times": [...],
    "semi_losses": [...],
    "l1_losses": [...],
    "total_losses": [...],
    "edit_info": {
        "edges_added": [...],
        "edges_removed": [...],
        "total_edges_added": [...],
        "total_edges_removed": [...],
        "total_edges": [...],
        "size_ratios": [...]
    },
    "best_epochs": [...],
    "graph_information": {
        "total_graphs": 188,
        "num_classes": 2,
        "class_distribution": {"0": 125, "1": 63},
        "graph_statistics": {
            "num_nodes": [...],
            "num_edges": [...],
            "avg_degree": [...],
            "density": [...],
            "avg_num_nodes": 17.93,
            "min_num_nodes": 10,
            "max_num_nodes": 28,
            "avg_num_edges": 19.79,
            "min_num_edges": 10,
            "max_num_edges": 33
        }
    }
}
```

#### Key Metrics Explained

- **Irrelevance**: Measures how well the explanation preserves the original prediction (higher is better)
- **Size**: Number of edges modified in the explanation (higher is better)
- **Time**: Execution time per graph in seconds
- **Semi Loss**: Semifactual loss component
- **L1 Loss**: Sparsity regularization loss
- **Size Ratio**: Proportion of total editable edges that were modified

### Training Factual Explainers

Each explainer has their own code and training pipeline. You can find the code for each explainer in `source/` directory.

We also provide a shell script to run the factual explainers, available in the main directory. To execute all explainers, simply run the following command:

```setup
./factual_run.sh
```

Please check the script files to see which command should be run to receive which results.

## Contact

If you have any questions or need further assistance, please feel free to contact me at d.mandaglio@dimes.unical.it


