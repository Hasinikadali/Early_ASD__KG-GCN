# subject_gcn.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def nx_subjects_to_pyg_with_meta(subject_graphs, pheno_df,
                                  region_names,
                                  pheno_cols=["AGE_AT_SCAN", "SEX"],
                                  node_feature_key="feat",
                                  fc_weight_key="weight"):

    if len(subject_graphs) == 0:
        raise ValueError("subject_graphs is empty.")

    node_features = []
    node_labels = []
    edge_index = []
    edge_weight = []
    node_to_subject = []
    node_to_roi = []
    subject_node_ranges = {}
    node_counter = 0

    pheno_subs = pheno_df.copy()
    for col in pheno_cols:
        if col in pheno_subs.columns:
            pheno_subs[col] = pd.to_numeric(pheno_subs[col], errors='coerce')
            pheno_subs[col] = pheno_subs[col].fillna(pheno_subs[col].mean())

    if len(pheno_cols) > 0:
        pheno_norm = (pheno_subs[pheno_cols] - pheno_subs[pheno_cols].mean()) / (pheno_subs[pheno_cols].std() + 1e-9)
        pheno_norm = pheno_norm.fillna(0.0)
    else:
        pheno_norm = pd.DataFrame(index=pheno_subs.index)

    sample_subj = next(iter(subject_graphs))
    sample_G = subject_graphs[sample_subj]

    if node_feature_key in sample_G.nodes[0]:
        sample_feat = np.array(sample_G.nodes[0][node_feature_key], dtype=np.float32)
    else:
        sample_feat = np.array([float(sample_G.degree(0)), float(nx.clustering(sample_G, 0))], dtype=np.float32)

    base_feat_len = len(sample_feat)

    for subj in subject_graphs:
        G = subject_graphs[subj]
        n_nodes = G.number_of_nodes()
        start = node_counter

        per_node_feats = []
        for roi in range(n_nodes):

            if node_feature_key in G.nodes[roi]:
                base_feat = np.array(G.nodes[roi][node_feature_key], dtype=np.float32)
            else:
                base_feat = np.array([float(G.degree(roi)), float(nx.clustering(G, roi))], dtype=np.float32)

            if subj in pheno_norm.index and len(pheno_cols) > 0:
                pheno_vec = pheno_norm.loc[subj].values.astype(np.float32)
            else:
                pheno_vec = np.zeros(len(pheno_cols), dtype=np.float32)

            feat = np.concatenate([base_feat, pheno_vec], axis=0)

            per_node_feats.append(feat)
            node_to_subject.append(subj)
            node_to_roi.append(roi)

        x_subj = np.vstack(per_node_feats).astype(np.float32)

        for u, v, d in G.edges(data=True):
            w = float(d.get(fc_weight_key, 1.0))
            if not np.isfinite(w):
                w = 0.0

            edge_index.append([u + node_counter, v + node_counter])
            edge_index.append([v + node_counter, u + node_counter])
            edge_weight.append(w)
            edge_weight.append(w)

        if subj in pheno_df.index and "DX_GROUP" in pheno_df.columns:
            raw = int(pheno_df.loc[subj, "DX_GROUP"])
            label = 1 if raw == 1 else 0
        else:
            label = 0

        node_features.append(torch.tensor(x_subj))
        node_labels.append(torch.tensor([label] * n_nodes, dtype=torch.long))

        node_counter += n_nodes
        subject_node_ranges[subj] = (start, node_counter)

    x = torch.cat(node_features, dim=0).numpy()
    x = np.nan_to_num(x)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-9)
    x = torch.tensor(x, dtype=torch.float)

    y = torch.cat(node_labels, dim=0)

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ew = np.array(edge_weight, dtype=np.float32)
        if not np.allclose(ew, 0):
            ew = (ew - ew.min()) / (ew.max() - ew.min() + 1e-9)
        ew = np.nan_to_num(ew, nan=0.0, posinf=0.0, neginf=0.0)
        edge_weight = torch.tensor(ew, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.edge_weight = edge_weight

    num_nodes = data.num_nodes
    perm = np.random.permutation(num_nodes)
    split = int(0.8 * num_nodes)
    train_idx = perm[:split]
    test_idx = perm[split:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data, node_to_subject, node_to_roi, subject_node_ranges, pheno_cols 

class SubjectGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data_or_x, edge_index=None, edge_weight=None):

        if isinstance(data_or_x, Data):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
            edge_weight = getattr(data_or_x, "edge_weight", None)
        else:
            x = data_or_x

        if edge_index is None or edge_index.size(1) == 0:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.nn.Linear(x.size(1), self.conv2.out_channels).to(x.device)(x)
            return x

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x    
#Trainig Function 
def train_test_gcn_with_metrics(data, epochs=200, lr=1e-3, weight_decay=5e-5, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)

    model = SubjectGCN(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=2,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train = data.y[data.train_mask]
    counts = torch.bincount(y_train, minlength=2).float()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    class_weights = (counts.sum() / counts) / counts

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_loss = float("inf")
    train_losses = []

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        if not torch.isfinite(loss):
            print(f"[epoch {epoch}] Loss is not finite.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()

    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()

    test_mask_np = data.test_mask.cpu().numpy()
    y_true_test = y_true[test_mask_np]
    y_pred_test = preds[test_mask_np]

    print("\n=== CLASSIFICATION REPORT (node-level) ===")
    print(classification_report(y_true_test, y_pred_test, digits=4, zero_division=0))

    acc = accuracy_score(y_true_test, y_pred_test)
    print(f"Test Accuracy: {acc:.4f}")

    return model, data, y_pred_test 
def explain_node_full(model, data, node_global_idx, node_to_subject, node_to_roi,
                      region_names, pheno_cols, top_k_edges=10, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    data = data.to(device)
    model.eval()

    x = data.x.clone().detach().to(device).requires_grad_(True)

    temp_data = Data(
        x=x,
        edge_index=data.edge_index,
        edge_weight=getattr(data, "edge_weight", None)
    )

    out = model(temp_data)
    pred_class = int(out[node_global_idx].argmax().item())

    loss = F.cross_entropy(out[[node_global_idx]], data.y[[node_global_idx]])

    model.zero_grad()
    loss.backward(retain_graph=True)

    if x.grad is None:
        raise RuntimeError("x.grad is None even after backward.")

    grad = x.grad[node_global_idx].abs().detach().cpu().numpy()
    grad = grad / (grad.sum() + 1e-9)

    n_total_feats = x.shape[1]
    n_pheno = len(pheno_cols)
    base_feat_len = n_total_feats - n_pheno

    base_feature_names = [f"feat_{i}" for i in range(base_feat_len)]
    pheno_feature_names = [f"PHENO_{c}" for c in pheno_cols]
    feature_names = base_feature_names + pheno_feature_names

    subj = node_to_subject[node_global_idx]
    roi = node_to_roi[node_global_idx]

    region_name = (
        region_names[roi]
        if roi < len(region_names)
        else f"ROI_{roi}"
    )

    true_label = int(data.y[node_global_idx].item())

    print(f"\n=== Explanation for global node {node_global_idx} ===")
    print(f"Subject: {subj} | ROI idx: {roi} ({region_name})")
    print(f"Predicted class: {pred_class} | True label: {true_label}")
    print("Feature importance (normalized grads):")

    for n, v in zip(feature_names, grad):
        print(f"  {n}: {v:.4f}")

    edge_index = data.edge_index.to(device)
    n_edges = edge_index.size(1)

    if n_edges == 0:
        print("No edges present in the data; skipping edge mask.")
        top_edges = []
    else:
        edge_mask = torch.nn.Parameter(
            torch.randn(n_edges, device=device) * 0.01,
            requires_grad=True
        )

        opt = torch.optim.Adam([edge_mask], lr=0.8)

        for i in range(120):
            opt.zero_grad()

            mask = torch.sigmoid(edge_mask)
            temp_data.edge_weight = mask

            out_masked = model(temp_data)
            loss_mask = F.cross_entropy(
                out_masked[[node_global_idx]],
                data.y[[node_global_idx]]
            )

            reg = 0.05 * mask.mean()
            (loss_mask + reg).backward()
            opt.step()

        mask_final = torch.sigmoid(edge_mask).detach().cpu().numpy()
        edges = edge_index.t().cpu().numpy()

        edge_list = [
            (int(u), int(v), float(w))
            for (u, v), w in zip(edges, mask_final)
        ]

        top_edges = sorted(
            edge_list,
            key=lambda x: x[2],
            reverse=True
        )[:top_k_edges]

        print("\nTop edges (global indices) influencing prediction:")

        for u, v, w in top_edges:
            subj_u, roi_u = node_to_subject[u], node_to_roi[u]
            subj_v, roi_v = node_to_subject[v], node_to_roi[v]

            name_u = (
                region_names[roi_u]
                if roi_u < len(region_names)
                else f"ROI_{roi_u}"
            )

            name_v = (
                region_names[roi_v]
                if roi_v < len(region_names)
                else f"ROI_{roi_v}"
            )

            print(
                f"  ({name_u} [subj:{subj_u}], "
                f"{name_v} [subj:{subj_v}]) -> weight {w:.4f}"
            )

    plt.figure(figsize=(6, 3))
    plt.bar(range(len(feature_names)), grad)
    plt.xticks(
        range(len(feature_names)),
        feature_names,
        rotation=45,
        ha="right"
    )
    plt.title(
        f"Node {node_global_idx}: "
        f"{region_name} feature importance"
    )
    plt.tight_layout()
    plt.show()

    return {
        "feature_importance": dict(zip(feature_names, grad)),
        "top_edges": top_edges,
        "subject": subj,
        "roi": roi,
        "region_name": region_name
    }
    