#!/bin/bash

datasets=("Proteins" "Mutag" "AIDS" "NCI1" "Graph-SST2" "ogbg_molhiv" "Mutagenicity") # full datasets ablation

# Methods
methods=("semifactual_full" "semifactual_full_singlebranch" "semifactual_mlp" "semifactual_direct")

# Hyperparameters
lam=0.5                       # Lambda to balance loss terms
lr=0.001                      # Learning rate
epochs=100                    # Number of training epochs
gam=0.5                       # Margin value for BPR loss
mask_thresh=0.5               # Threshold to binarize relaxed adjacency matrix
perturb_perc=25               # Percentage of edges to perturb

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
  dataset="$current_dataset"
  for current_method in "${methods[@]}"
  do
    method="$current_method"
    
      echo "Running with dataset: $dataset, method: $method, use_last: True"
      mkdir -p "output/${dataset}/${method}"
      /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/${method}.py" \
        --dataset "${dataset}" \
        --lam ${lam} \
        --lr ${lr} \
        --epochs ${epochs} \
        --gam ${gam} \
        --mask_thresh ${mask_thresh} \
        --perturb_perc ${perturb_perc} \
        --device ${device} \
        --gnn_run ${gnn_run} \
        --explainer_run ${explainer_run} \
        --gnn_type ${gnn_type} \
        --robustness ${robustness} \
        --use_last \
        > "output/${dataset}/${method}/log_lam${lam}_lr${lr}_epochs${epochs}_gam${gam}_maskthresh${mask_thresh}_perturbperc${perturb_perc}_gnntype${gnn_type}_useLastTrue.txt" 2>&1
      
      echo "Running with dataset: $dataset, method: $method, use_last: False"
      /root/miniconda3/envs/gnnxbench/bin/python3 \
        "/root/workspace/semifactual-gnn/source/${method}.py" \
        --dataset "${dataset}" \
        --lam ${lam} \
        --lr ${lr} \
        --epochs ${epochs} \
        --gam ${gam} \
        --mask_thresh ${mask_thresh} \
        --perturb_perc ${perturb_perc} \
        --device ${device} \
        --gnn_run ${gnn_run} \
        --explainer_run ${explainer_run} \
        --gnn_type ${gnn_type} \
        --robustness ${robustness} \
        > "output/${dataset}/${method}/log_lam${lam}_lr${lr}_epochs${epochs}_gam${gam}_maskthresh${mask_thresh}_perturbperc${perturb_perc}_gnntype${gnn_type}_useLastFalse.txt" 2>&1
  done
done