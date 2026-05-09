#!/bin/bash

# Array of all dataset options
datasets=("Mutag" "Mutagenicity" "Proteins" "AIDS" "NCI1" "Graph-SST2" "ogbg_molhiv")

# Hyperparameters
lam=0.5                       # Lambda to balance loss terms
lr=0.001                      # Learning rate
epochs=100                    # Number of training epochs
gam=0.5                       # Margin value for BPR loss
mask_thresh=0.5               # Threshold to binarize relaxed adjacency matrix
perturb_perc=0                # Percentage of edges to perturb

# Hardware configuration
device="0"

# Experiment runs
gnn_run=1
explainer_run=1

# GNN configuration (options: gcn, gat, gin, sage)
gnn_type="gcn"

# Robustness setting (options: topology_random, topology_adversarial, feature, na)
robustness="na"

for current_dataset in "${datasets[@]}"
do
  # CFF
  echo "Running CFF with dataset: $dataset"
  mkdir -p "output/${dataset}/cff"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/cff.py" \
        --dataset "${dataset}" \
        --gnn_type ${gnn_type} \
        --device ${device} \
        --alp 1.0 \
        > "output/${dataset}/cff/log.txt" 2>&1

  # GNNExplainer
  echo "Running GNNExplainer with dataset: $dataset"
  mkdir -p "output/${dataset}/gnnexplainer"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/gnnexplainer.py" \
        --dataset "${dataset}" \
        > "output/${dataset}/gnnexplainer/log.txt" 2>&1

  # GEM-GT
  echo "Running GEM-GT with dataset: $dataset"
  mkdir -p "output/${dataset}/gem_gt"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/gem_gt.py" \
        --dataset "${dataset}" \
        > "output/${dataset}/gem_gt/log.txt" 2>&1

  # GEM
  echo "Running GEM with dataset: $dataset"
  mkdir -p "output/${dataset}/gem"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/gem.py" \
        --dataset "${dataset}" \
        > "output/${dataset}/gem/log.txt" 2>&1

  # PGExplainer
  echo "Running PGExplainer with dataset: $dataset"
  mkdir -p "output/${dataset}/pgexplainer"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/pgexplainer.py" \
        --dataset "${dataset}" \
        > "output/${dataset}/pgexplainer/log.txt" 2>&1

  # SubgraphX
  echo "Running SubgraphX with dataset: $dataset"
  mkdir -p "output/${dataset}/subgraphx"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/subgraphx.py" \
        --dataset "${dataset}" \
        --explain_test_only \
        > "output/${dataset}/subgraphx/log.txt" 2>&1

  # TAGExplainer
  echo "Running TAGExplainer (stage 1) with dataset: $dataset"
  mkdir -p "output/${dataset}/tagexplainer"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/tagexplainer.py" \
        --dataset "${dataset}" \
        --stage 1 \
        > "output/${dataset}/tagexplainer/log_stage1.txt" 2>&1

  echo "Running TAGExplainer (stage 2) with dataset: $dataset"
  mkdir -p "output/${dataset}/tagexplainer"
  /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/tagexplainer.py" \
        --dataset "${dataset}" \
        > "output/${dataset}/tagexplainer/log.txt" 2>&1
done