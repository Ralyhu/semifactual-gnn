import torch
import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import math
import time
import data_utils
from gnn_trainer import GNNTrainer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json

# This class is responsible for explaining a single graph.
class ExplainModelGraph(torch.nn.Module):

    def __init__(self, graph, pred_orig, device, base_model):
        super(ExplainModelGraph, self).__init__()
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.pred_orig = pred_orig
        self.device = device

        # E^m lo costruisco dall'oggetto graph di tipo Data
        self.edges_to_remove = torch.cat([graph.edges_to_remove, graph.edges_added], dim=1)
        self.edges_to_add = torch.cat([graph.edges_to_add, graph.edges_removed], dim=1)

        # Add to the edge index the entries corresponding to edges in edges_to_add and edges_removed as they are needed for learning the mask
        self.graph = self.create_graph_with_added_edges(graph)
        
        self.uneditable_edges = self.graph.edge_index.size(1) - self.edges_to_remove.size(1) - self.edges_to_add.size(1)
        
        # Initialize mask parameters ONLY for edges we need to learn
        # Count unique edges that need masks
        masked_edges_count = 0
        if self.edges_to_remove.numel() > 0:
            masked_edges_count += self.edges_to_remove.size(1) // 2  # Account for undirected edges
        if self.edges_to_add.numel() > 0:
            masked_edges_count += self.edges_to_add.size(1) // 2  # Account for undirected edges
        
        # Create parameter vector instead of matrix
        if masked_edges_count > 0:
            self.edge_mask_params = torch.nn.Parameter(torch.FloatTensor(masked_edges_count))
            
            # Old way as in CF2
            #std = torch.nn.init.calculate_gain("relu") * math.sqrt(2.0 / (2 * masked_edges_count))
            #with torch.no_grad():
            #    self.edge_mask_params.normal_(1.0, std)

            # This way, after sigmoid, all values will be exactly 0.5
            with torch.no_grad():
                self.edge_mask_params.fill_(0.0)
    
        else:
            raise ValueError("No edge in E^m (edges_to_remove and edges_to_add are both empty)")
        
        # Create a mapping from edge to parameter index - use canonical edge representation (smaller node first)
        self.edge_to_param_idx = {}
        param_idx = 0
        
        # Map edges_to_remove to parameter indices (only unique undirected edges)
        if self.edges_to_remove.numel() > 0:
            added_edges = set()
            for i in range(self.edges_to_remove.size(1)):
                u, v = self.edges_to_remove[0, i].item(), self.edges_to_remove[1, i].item()
                edge_key = (min(u, v), max(u, v))  # Canonical form
                if edge_key not in added_edges:
                    self.edge_to_param_idx[edge_key] = param_idx
                    param_idx += 1
                    added_edges.add(edge_key)
            
        # Map edges_to_add to parameter indices (only unique undirected edges)
        if self.edges_to_add.numel() > 0:
            added_edges = set()
            for i in range(self.edges_to_add.size(1)):
                u, v = self.edges_to_add[0, i].item(), self.edges_to_add[1, i].item()
                edge_key = (min(u, v), max(u, v))  # Canonical form
                if edge_key not in added_edges:
                    self.edge_to_param_idx[edge_key] = param_idx
                    param_idx += 1
                    added_edges.add(edge_key)
        
        # Precompute which edges are "to be added" (w.r.t. E^m) for forward pass efficiency
        if self.edges_to_add.numel() > 0:
            # Create canonical representation for all edges in graph
            self.edge_index_canonical = torch.stack([
                torch.min(self.graph.edge_index[0], self.graph.edge_index[1]),
                torch.max(self.graph.edge_index[0], self.graph.edge_index[1])
            ], dim=0)
            
            # Create canonical representation for all edges_to_add
            edges_to_add_canonical = torch.stack([
                torch.min(self.edges_to_add[0], self.edges_to_add[1]),
                torch.max(self.edges_to_add[0], self.edges_to_add[1])
            ], dim=0)

            # Convert edges_to_add to a set of tuples for O(1) lookup
            edges_to_add_set = {(edges_to_add_canonical[0, i].item(), edges_to_add_canonical[1, i].item()) 
                               for i in range(edges_to_add_canonical.size(1))}
            
            # Create Boolean mask in one go using list comprehension
            edge_matches = [(self.edge_index_canonical[0, i].item(), self.edge_index_canonical[1, i].item()) in edges_to_add_set
                           for i in range(self.edge_index_canonical.size(1))]
            
            self.is_added_edge = torch.tensor(edge_matches, dtype=torch.bool, device=self.device)
        else:
            # No edges to add, all are "to be removed" edges
            self.is_added_edge = torch.zeros(self.graph.edge_index.size(1), dtype=torch.bool, device=self.device)
        

    def create_graph_with_added_edges(self, graph):
        new_graph = Data()

        # Print the edge index tensor for debugging
        #print(f"edge_index before: {graph.edge_index}, size: {graph.edge_index.size()}")
        
        # Copy all attributes from the original graph
        for key, value in graph:
            if key != 'edge_index':  # We'll handle edge_index separately
                setattr(new_graph, key, value.clone() if torch.is_tensor(value) else value)
        
        # Start with original edge_index
        combined_edges = graph.edge_index.clone()
        
        # Add edges_to_remove if they exist - these must be included for masked_adj to work properly
        if hasattr(graph, 'edges_removed') and graph.edges_removed.numel() > 0:
            combined_edges = torch.cat([combined_edges, graph.edges_removed], dim=1)
        
        # Add edges_to_add if they exist - these must be included for masked_adj to work properly
        if hasattr(graph, 'edges_to_add') and graph.edges_to_add.numel() > 0:
            combined_edges = torch.cat([combined_edges, graph.edges_to_add], dim=1)
        
        # Remove duplicate edges
        if combined_edges.numel() > 0:
            # this operation is not needed as combined_edges is already unique (no damage leaving for now)
            new_edge_index = torch.unique(combined_edges, dim=1) 
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Set the new edge_index
        new_graph.edge_index = new_edge_index

        #print(f"edge_index after: {new_graph.edge_index}, size: {new_graph.edge_index.size()}")
        
        return new_graph

    def get_masked_adj(self):
        # Create a mask tensor of ones with length equal to the number of edges in the graph
        num_edges = self.graph.edge_index.size(1)
        edge_mask = torch.zeros(num_edges, device=self.device) # TODO: check if this is correct or should be ones
        
        # If no parameters, return all ones mask
        if self.edge_mask_params.numel() == 0:
            return edge_mask
        
        # Apply sigmoid to the parameters
        mask_values = torch.sigmoid(self.edge_mask_params)
        
        # Process edges that need masking
        for edge_idx in range(num_edges):
            u = self.graph.edge_index[0, edge_idx].item()
            v = self.graph.edge_index[1, edge_idx].item()
            
            # Use canonical edge representation
            edge_key = (min(u, v), max(u, v))
            
            # Check if edge is in our mapping
            if edge_key in self.edge_to_param_idx: # TODO: probably there is not need to check the condition in the if statement (always true)
                param_idx = self.edge_to_param_idx[edge_key]
                edge_mask[edge_idx] = mask_values[param_idx]
        
        return edge_mask

    def debug_check_mask_values(self, masked_adj, expected_values=None):
        """
        Debug utility that checks if all sigmoid parameter values appear in the masked adjacency matrix.
        
        Args:
            masked_adj (Tensor): The masked adjacency matrix
            expected_values (numpy.ndarray, optional): Pre-computed sigmoid values
            
        Returns:
            None, but prints detailed debug information
        """
        if not hasattr(self, 'edge_mask_params') or self.edge_mask_params.numel() == 0:
            return
            
        if expected_values is None:
            expected_values = torch.sigmoid(self.edge_mask_params).detach().cpu().numpy()
        present_values = masked_adj.detach().cpu().numpy()

        # Use a tolerance for floating point comparison
        tolerance = 1e-6
        all_present = True
        missing_values = []
        
        # Create a set of unique rounded values present in masked_adj for efficient lookup
        # Rounding based on tolerance to handle precision issues
        present_values_rounded_set = set(np.round(present_values, decimals=int(-np.log10(tolerance))))

        for val in expected_values:
            rounded_val = np.round(val, decimals=int(-np.log10(tolerance)))
            # Check if a value close to rounded_val exists in the set
            found = any(np.isclose(rounded_val, present_rounded, atol=tolerance) for present_rounded in present_values_rounded_set)

            if not found:
                all_present = False
                missing_values.append(val)

        
        if not all_present:
            print(f"--- DEBUG CHECK WARNING ---")
            print(f"Not all sigmoid(param) values found in masked_adj!")
            print(f"Expected values (sigmoid(params)): {np.round(expected_values, 4)}")
            print(f"Unique values present in masked_adj: {np.unique(np.round(present_values, 4))}")
            print(f"Missing values (approx): {np.round(missing_values, 4)}")


            # Map missing values to their corresponding edges and categorize them
            if len(missing_values) > 0:
                missing_edges = []
                missing_categories = []
                masked_adj_values = []
                edge_found_in_graph = []
                
                # Iterate through edge_to_param_idx to find edges corresponding to missing values
                for edge_key, param_idx in self.edge_to_param_idx.items():
                    param_value = expected_values[param_idx]
                    if any(np.isclose(param_value, missing_val, atol=tolerance) for missing_val in missing_values):
                        u, v = edge_key
                        
                        # Determine if this edge is in to_add or to_remove set
                        edge_tensor = torch.tensor([[u, v], [v, u]]).t()
                        
                        # Check if edge is in edges_to_add
                        is_to_add = False
                        for i in range(self.edges_to_add.size(1)):
                            e = self.edges_to_add[:, i]
                            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                is_to_add = True
                                break
                        
                        # Check if edge is in edges_to_remove
                        is_to_remove = False
                        for i in range(self.edges_to_remove.size(1)):
                            e = self.edges_to_remove[:, i]
                            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                is_to_remove = True
                                break
                        
                        category = "TO_ADD" if is_to_add else "TO_REMOVE" if is_to_remove else "UNKNOWN"
                        missing_edges.append(f"({u},{v})")
                        missing_categories.append(category)
                        
                        # Check if the edge exists in the graph's edge_index
                        edge_found = False
                        adj_value = None
                        for edge_idx in range(self.graph.edge_index.size(1)):
                            e_u = self.graph.edge_index[0, edge_idx].item()
                            e_v = self.graph.edge_index[1, edge_idx].item()
                            if (e_u == u and e_v == v) or (e_u == v and e_v == u):
                                edge_found = True
                                adj_value = masked_adj[edge_idx].item()
                                break
                        
                        edge_found_in_graph.append(edge_found)
                        masked_adj_values.append(adj_value)
                
                # Print missing edges with their categories and masked_adj values
                if missing_edges:
                    print(f"Missing edges and their categories:")
                    for i, (edge, category, found, adj_val) in enumerate(zip(missing_edges, missing_categories, edge_found_in_graph, masked_adj_values)):
                        expected_val = np.round(expected_values[self.edge_to_param_idx[tuple(map(int, edge.strip('()').split(',')))]],4)
                        status = f"Found in graph, masked_adj value: {np.round(adj_val, 4)}" if found else "Not found in graph.edge_index!"
                        print(f"  - Edge {edge}: {category} (expected: {expected_val}, {status})")

                print(f"---------------------------")
        # Optional: Uncomment to confirm when check passes
        #else:
        #    print("--- DEBUG CHECK OK --- All sigmoid(param) values appear in masked_adj.")

    def loss(self, prob_original_class, prob_different_class, gam, lambda_param):
        sigmoid = torch.nn.Sigmoid()
        # Semifactual loss: penalize when prediction deviates from original
        semifactual_loss = sigmoid(4.0 * (gam + prob_different_class - prob_original_class))
        
        # Maximize number of removed edges (minimize number of kept edges)

        # Since we're now using a more efficient representation, we compute L1 directly from edge_mask_params
        # masked_adj = self.get_masked_adj()
        # Run debug check if needed
        # self.debug_check_mask_values(masked_adj)  # Uncomment for debugging
        
        # Efficiently compute L1 only on the editable edges
        # Since edge_mask_params already contains only parameters for editable edges
        # We can directly compute the mean from these values after sigmoid
        L1 = 1.0 - torch.mean(torch.sigmoid(self.edge_mask_params))
        
        if lambda_param > 0:
            # Total loss: balance between maintaining prediction and maximizing edits
            loss = semifactual_loss + lambda_param * L1
        else: 
            loss = semifactual_loss
        
        return semifactual_loss, L1, loss

    def forward(self, base_model, masked_adj=None):
        training = masked_adj is None

        if masked_adj is None:
            masked_adj = self.get_masked_adj()
        
        # Apply mask differently based on edge type using pre-computed is_added_edge tensor:
        # - For potential edges (is_added_edge): use mask directly
        # - For existing edges (!is_added_edge): use 1-mask

        # Run debug check if needed
        if training:
            self.debug_check_mask_values(masked_adj)  # Uncomment for debugging

        edge_weights = self.build_edge_weights(masked_adj)
        #edge_weights = torch.where(self.is_added_edge, masked_adj, 1 - masked_adj) # corresponds to "continuous" adjacency matrix of G^diamond
        
        pred_probs = torch.softmax(base_model(self.graph, edge_weights)[-1][0], dim=0)
        #print(f"pred_probs: {pred_probs}")

        orig_class = self.pred_orig
        different_class = 1 - orig_class
        #print(f"orig_class: {orig_class}, different_class: {different_class}")

        prob_original_class = pred_probs[orig_class]
        prob_different_class = pred_probs[different_class]
        # Print with different formats for training vs final solution
        # if training:
        #     # During training
        #     print(f"Training: prob_original_class: {prob_original_class}, prob_different_class: {prob_different_class}")
        # else:
        #     # rounded solution
        #     print(f"Rounded: prob_original_class: {prob_original_class:.4f}, prob_different_class: {prob_different_class:.4f}")
        return prob_original_class, prob_different_class, masked_adj
    
    def build_edge_weights(self, edge_mask):
        edge_weights = torch.where(self.is_added_edge, edge_mask, 1 - edge_mask) 
        return edge_weights


    def get_parameter_info(self):
        """Returns detailed information about each learnable parameter.

        Returns:
            list[dict]: A list where each dictionary contains:
                - 'edge': The canonical edge tuple (min_node, max_node).
                - 'raw_value': The raw value of the learned parameter.
                - 'mask_value': The sigmoid-transformed value (mask probability).
                - 'type': 'remove' if the edge was in edges_to_remove, 'add' if in edges_to_add.
        """
        param_info_list = []

        # Ensure params are on CPU for easy access
        edge_params = self.edge_mask_params.cpu().detach()
        mask_values = torch.sigmoid(edge_params)

        # Create canonical sets for quick type lookup
        removable_edges_set = set()
        if self.edges_to_remove.numel() > 0:
            edges_rem = self.edges_to_remove.cpu().numpy()
            for i in range(edges_rem.shape[1]):
                removable_edges_set.add(tuple(sorted((edges_rem[0, i], edges_rem[1, i]))))

        addable_edges_set = set()
        if self.edges_to_add.numel() > 0:
            edges_add = self.edges_to_add.cpu().numpy()
            for i in range(edges_add.shape[1]):
                addable_edges_set.add(tuple(sorted((edges_add[0, i], edges_add[1, i]))))

        # Iterate through the mapping to gather information
        for edge_key, param_idx in self.edge_to_param_idx.items():
            raw_value = edge_params[param_idx].item()
            mask_value = mask_values[param_idx].item()
            edge_type = 'unknown' # Default
            if edge_key in removable_edges_set:
                edge_type = 'remove'
            elif edge_key in addable_edges_set:
                edge_type = 'add'

            param_info_list.append({
                'edge': edge_key,
                'raw_value': raw_value,
                'mask_value': mask_value,
                'type': edge_type
            })

        return param_info_list

# This class is responsible for explaining the dataset, i.e. each graph in the dataset.
class GraphExplainerEdge(torch.nn.Module):

    def __init__(self, base_model, G_dataset, args, device, use_last):

        super(GraphExplainerEdge, self).__init__()
        self.base_model = base_model
        self.G_dataset = G_dataset
        self.args = args
        self.device = device
        self.use_last = use_last

    def explain_dataset(self, perturb_perc: int = 25):

        sf_list = []
        # Initialize lists to store individual values
        all_irrelevances_list = []
        all_sizes_list = []
        all_times_list = []
        all_semi_losses = []
        all_l1_losses = []
        all_total_losses = []
        all_edges_added = []
        all_edges_removed = []
        all_total_edges_added = []
        all_total_edges_removed = []
        all_total_edges = []
        all_size_ratios = []
        all_best_epochs = []

        index = 0
        for g in tqdm(self.G_dataset, desc='Graph'):
            # if index != 0:
            #     break
            print(f"Graph {index} - Number of nodes: {g.num_nodes}, Number of edges: {g.edge_index.size(1) // 2}")
            index += 1

            if data_utils.istoskip(g, perturb_perc):
                continue
            
            g = g.to(self.device)
            label = int(g.y)

            # create a modified graph by calling a function using the parameter
            graph_modified = data_utils.perturb_graphs_semifactuals(g, perturb_perc, False, seed=args.explainer_run)

            start_time = time.time()

            pred = torch.softmax(self.base_model(graph_modified)[-1][0], dim=0)
            print(f"pred_probs original: {pred}")


            # if pred[1] > pred[0]:
            #     print(f"Found case where second class probability ({pred[1]:.4f}) is higher than first class ({pred[0]:.4f})")
            pred_orig = pred.argmax().item()
            print(f"pred_orig before building explainer: {pred_orig}")

            masked_adj, binarized_mask, irrelevance, size, explainer, metrics = self.explain(graph_modified, pred_orig)

            end_time = time.time()
            current_time = end_time - start_time
            print(f"Time taken to explain graph {index}: {current_time:.2f} seconds")
        

            # Append individual values to lists
            all_irrelevances_list.append(irrelevance)
            all_sizes_list.append(size)
            all_times_list.append(current_time)
            
            # Append loss metrics
            all_semi_losses.append(metrics['semi_losses'])
            all_l1_losses.append(metrics['l1_losses'])
            all_total_losses.append(metrics['total_losses'])
            
            # Append edit information
            edit_info = metrics['edit_info']
            all_edges_added.append(edit_info['edges_added'])
            all_edges_removed.append(edit_info['edges_removed'])
            all_total_edges_added.append(edit_info['total_edges_added'])
            all_total_edges_removed.append(edit_info['total_edges_removed'])
            all_total_edges.append(edit_info['total_edges'])
            all_size_ratios.append(edit_info['size_ratio'])
            all_best_epochs.append(metrics['best_epoch'])
        
            weights_sf = explainer.build_edge_weights(binarized_mask)

            graph_sf = Data(
                x=explainer.graph.x.clone(),
                edge_index=explainer.graph.edge_index.clone().detach(),
                edge_attr=explainer.graph.edge_attr.clone().detach() if explainer.graph.edge_attr is not None else None,
                edge_weight=weights_sf.clone().detach()
            ).to(self.device)
            
            pred_sf = self.base_model(graph_sf, graph_sf.edge_weight)[-1][0]

            sf_list.append({
                "graph": graph_modified.cpu(), "graph_sf": graph_sf.cpu(),
                "label": label, "pred": pred.cpu(), "pred_sf": pred_sf.cpu()
            })

        # Store all metrics in a dictionary
        metrics = {
            'irrelevances': all_irrelevances_list,
            'sizes': all_sizes_list,
            'times': all_times_list,
            'semi_losses': all_semi_losses,
            'l1_losses': all_l1_losses,
            'total_losses': all_total_losses,
            'edit_info': {
                'edges_added': all_edges_added,
                'edges_removed': all_edges_removed,
                'total_edges_added': all_total_edges_added,
                'total_edges_removed': all_total_edges_removed,
                'total_edges': all_total_edges,
                'size_ratios': all_size_ratios
            },
            'best_epochs': all_best_epochs
        }

        return sf_list, metrics

    def explain(self, g, pred_orig):
        explainer = ExplainModelGraph(
            graph=g,
            pred_orig=pred_orig,
            device=self.device,
            base_model=self.base_model
        ).to(self.device)
        
        # For tracking the best explanation
        best_mask = torch.zeros_like(explainer.is_added_edge, dtype=torch.float, device=self.device)
        best_size_ratio = 0.0
        best_epoch = 0

        # For tracking solutions from last 10% of epochs (when use_last is True)
        last_epochs_start = int(self.args.epochs * 0.9)
        tracked_solutions = []  # List to store (mask, preserves_prediction, size_ratio, epoch)

        # Initialize lists to store metrics
        semi_losses = []
        l1_losses = []
        total_losses = []

        # debug
        pred_prob_orig_after_edit, _, _ = explainer(self.base_model, masked_adj=best_mask)
        print(f"pred_prob_orig before training loop: {pred_prob_orig_after_edit}")

        # Calculate total edges that could be modified
        total_edges_to_add = 0
        total_edges_to_remove = 0
        if explainer.edges_to_add.numel() > 0:
            total_edges_to_add = explainer.edges_to_add.size(1) // 2
        if explainer.edges_to_remove.numel() > 0:
            total_edges_to_remove = explainer.edges_to_remove.size(1) // 2
        total_edges = total_edges_to_add + total_edges_to_remove
        
        # Train explainer
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()
        
        for epoch in range(1, self.args.epochs + 1):
            optimizer.zero_grad()
            prob_orig, prob_diff, masked_adj = explainer(self.base_model)
            
            # Calculate loss
            semi_loss, l1, loss = explainer.loss(
                prob_orig, prob_diff, self.args.gam, self.args.lam)
            
            # Store loss values
            semi_losses.append(semi_loss.item())
            l1_losses.append(l1.item())
            total_losses.append(loss.item())
            
            # Print loss values
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.args.epochs}: Semi Loss: {semi_loss.item():.4f}, L1: {l1.item():.4f}, Total Loss: {loss.item():.4f}")
            
            # Round the mask and check if it preserves the class
            binarized_mask = (masked_adj > self.args.mask_thresh).to(torch.float32)
            pred_prob_orig_after_edit, _, _ = explainer(self.base_model, binarized_mask)
            preserves_prediction = pred_prob_orig_after_edit > 0.5
            
            # Count modified edges
            is_added_edge = explainer.is_added_edge
            above_threshold = binarized_mask > 0.5
            
            added_mask = is_added_edge & above_threshold
            removed_mask = (~is_added_edge) & above_threshold 
            edges_being_added = added_mask.sum().item() // 2
            edges_being_removed = removed_mask.sum().item() // 2
            size = edges_being_added + edges_being_removed
            size_ratio = size / total_edges if total_edges > 0 else 0
            
            if self.use_last:
                # Track solutions from last 10% of epochs
                if epoch >= last_epochs_start:
                    tracked_solutions.append((binarized_mask, preserves_prediction, size_ratio, epoch))
            else:
                # Update best explanation if this one is better
                if preserves_prediction and size_ratio >= best_size_ratio:
                    best_mask = binarized_mask
                    best_size_ratio = size_ratio
                    best_epoch = epoch
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        if self.use_last:
            # Select from tracked solutions based on criteria
            solutions_with_same_class = [(mask, ratio, epoch) for mask, preserves, ratio, epoch in tracked_solutions if preserves]
            if solutions_with_same_class:
                # If we have solutions that preserve the original class, pick the one with max distance
                best_mask, best_size_ratio, best_epoch = max(solutions_with_same_class, key=lambda x: x[1])
            else:
                # Otherwise pick the solution with max distance
                best_mask, _, best_size_ratio, best_epoch = max(tracked_solutions, key=lambda x: x[2])
            print(f"Selected solution from epoch {best_epoch} with size ratio {best_size_ratio:.4f}")

        # Final evaluation with best mask
        pred_prob_orig_after_edit, _, _ = explainer(self.base_model, best_mask.to(self.device))
        preserves_prediction_check = int(pred_prob_orig_after_edit > 0.5)
        
        # Count final modified edges
        is_added_edge = explainer.is_added_edge
        above_threshold = best_mask > 0.5
        
        added_mask = is_added_edge & above_threshold
        removed_mask = (~is_added_edge) & above_threshold 
        edges_being_added = added_mask.sum().item() // 2
        edges_being_removed = removed_mask.sum().item() // 2
        size = edges_being_added + edges_being_removed
        size_ratio = size / total_edges if total_edges > 0 else 0
        
        # Print summary
        print(f"\n===== Final Explanation Summary =====")
        print(f"Input Graph class: {g.y.item()}")
        print(f"Edges modified: {size}/{total_edges} ({size_ratio:.2%}) [Added: {edges_being_added}/{total_edges_to_add} ({edges_being_added/total_edges_to_add:.2%}), Removed: {edges_being_removed}/{total_edges_to_remove} ({edges_being_removed/total_edges_to_remove:.2%})]")
        print(f"Maintains prediction: {'Yes' if preserves_prediction_check else 'No'} (Prob: {pred_prob_orig_after_edit.item():.4f})")
        print("=====================================\n")
        
        # Return additional metrics
        return masked_adj.cpu(), best_mask.cpu(), preserves_prediction_check, size_ratio, explainer, {
            'semi_losses': semi_losses,
            'l1_losses': l1_losses,
            'total_losses': total_losses,
            'best_epoch': best_epoch,
            'edit_info': {
                'edges_added': edges_being_added,
                'edges_removed': edges_being_removed,
                'total_edges_added': total_edges_to_add,
                'total_edges_removed': total_edges_to_remove,
                'total_edges': total_edges,
                'size_ratio': size_ratio
            }
        }
    
    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutag',
                    choices=['Mutagenicity', 'Proteins', 'Mutag', 'IMDB-B', 'AIDS', 'NCI1', 'Tree-of-Life', 'Graph-SST2', 'DD', 'REDDIT-B', 'ogbg_molhiv'],
                    help="Dataset name")
parser.add_argument('--lam', type=float, default=0.5,
                    help='Lambda hyperparameter to balance between loss terms')
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of the training epochs")
parser.add_argument("--gam", dest="gam", type=float, default=0.5, help="margin value for bpr loss")
parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=.5, help="threshold to convert relaxed adj matrix to binary")
parser.add_argument("--perturb_perc", type=int, default=25, help='Percentage of edges to perturb for creating the modified graph for semifactual explanation')
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--gnn_run', type=int, default=1)
parser.add_argument('--explainer_run', type=int, default=1)
parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'])
parser.add_argument('--use_last', action='store_true', default=True, help="Boolean flag to indicate whether to use the last explanation.")

args = parser.parse_args()

# Print all arguments
print("===== Script Arguments =====")
for arg, value in sorted(vars(args).items()):
    print(f"{arg}: {value}")
print("==========================\n")

# Logging.
result_folder = f'data/{args.dataset}/semifactual_direct/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
dataset = data_utils.load_dataset(args.dataset)
splits, indices = data_utils.split_data(dataset)
dataset = dataset[indices[2]]  # test graphs only as this is not an inductive explainer
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

label_counts = {}
for idx, graph in enumerate(dataset):
    label = int(graph.y)
    if label not in label_counts:
        label_counts[label] = 0
    label_counts[label] += 1

print("\n===== Labels distribution =====")
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count} #graphs ({count/len(dataset):.2%})")

torch.manual_seed(args.explainer_run)
torch.cuda.manual_seed(args.explainer_run)
np.random.seed(args.explainer_run)
random.seed(args.explainer_run)

semifactual_path = os.path.join(result_folder, f"exps_lam{args.lam}_lr{args.lr}_epochs{args.epochs}_gam{args.gam}_maskthresh{args.mask_thresh}_perturbperc{args.perturb_perc}_gnntype{args.gnn_type}_useLast{args.use_last}.pt")
args.method = 'classification'

trainer = GNNTrainer(dataset_name=args.dataset, gnn_type=args.gnn_type, task='basegnn', device=args.device)
model = trainer.load(args.gnn_run)
model.eval()

node_embeddings, graph_embeddings, outs = trainer.load_gnn_outputs(args.gnn_run)

    
explainer = GraphExplainerEdge(
    base_model=model,
    G_dataset=dataloader,
    args=args,
    device=device,
    use_last=args.use_last
)
sfs, metrics = explainer.explain_dataset(perturb_perc=args.perturb_perc)

# Convert lists to tensors for calculations
all_irrelevances_tensor = torch.tensor(metrics['irrelevances'], dtype=torch.float)
all_sizes_tensor = torch.tensor(metrics['sizes'], dtype=torch.float)
all_times_tensor = torch.tensor(metrics['times'], dtype=torch.float)

# Calculate mean and standard deviation
avg_irrelevance = torch.mean(all_irrelevances_tensor)
std_irrelevance = torch.std(all_irrelevances_tensor)
avg_size = torch.mean(all_sizes_tensor)
std_size = torch.std(all_sizes_tensor)
avg_time = torch.mean(all_times_tensor)
std_time = torch.std(all_times_tensor)

print("FINAL RESULTS (ACROSS ALL GRAPHS)")
print(f"Irrelevance: {avg_irrelevance:.4f} +/- {std_irrelevance:.4f}")
print(f"Size: {avg_size:.4f} +/- {std_size:.4f}")
print(f"Time (seconds): {avg_time:.4f} +/- {std_time:.4f}")

# Save semifactuals as PyTorch file
torch.save(sfs, semifactual_path)

# Create metrics filename with specified format
metrics_filename = f"log_lam{args.lam}_lr{args.lr}_epochs{args.epochs}_gam{args.gam}_maskthresh{args.mask_thresh}_perturbperc{args.perturb_perc}_gnntype{args.gnn_type}_useLast{args.use_last}.json"
metrics_path = os.path.join(result_folder, metrics_filename)

# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Collect graph information
graph_info = {
    'total_graphs': len(dataset),
    'num_classes': len(label_counts),
    'class_distribution': label_counts,
    'graph_statistics': {
        'num_nodes': [],
        'num_edges': [],
        'avg_degree': [],
        'density': []
    }
}

# Calculate class distribution
for graph in dataset:
    label = int(graph.y)
    if label not in graph_info['class_distribution']:
        graph_info['class_distribution'][label] = 0
    graph_info['class_distribution'][label] += 1
    
    # Calculate graph statistics
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1) // 2  # Divide by 2 for undirected graphs
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    graph_info['graph_statistics']['num_nodes'].append(num_nodes)
    graph_info['graph_statistics']['num_edges'].append(num_edges)
    graph_info['graph_statistics']['avg_degree'].append(avg_degree)
    graph_info['graph_statistics']['density'].append(density)

# Calculate average statistics
for stat in ['num_nodes', 'num_edges', 'avg_degree', 'density']:
    values = graph_info['graph_statistics'][stat]
    graph_info['graph_statistics'][f'avg_{stat}'] = sum(values) / len(values)
    graph_info['graph_statistics'][f'min_{stat}'] = min(values)
    graph_info['graph_statistics'][f'max_{stat}'] = max(values)

# Add run parameters and summary statistics to metrics
metrics['run_parameters'] = {
    'dataset': args.dataset,
    'lambda': args.lam,
    'learning_rate': args.lr,
    'epochs': args.epochs,
    'gamma': args.gam,
    'mask_threshold': args.mask_thresh,
    'perturb_percentage': args.perturb_perc,
    'gnn_type': args.gnn_type,
    'use_last': args.use_last
}

metrics['summary_statistics'] = {
    'avg_irrelevance': avg_irrelevance.item(),
    'std_irrelevance': std_irrelevance.item(),
    'avg_size': avg_size.item(),
    'std_size': std_size.item(),
    'avg_time': avg_time.item(),
    'std_time': std_time.item()
}

# Add graph information to metrics
metrics['graph_information'] = graph_info

serializable_metrics = convert_to_serializable(metrics)

# Save metrics as JSON file
with open(metrics_path, 'w') as f:
    json.dump(serializable_metrics, f, indent=2)
