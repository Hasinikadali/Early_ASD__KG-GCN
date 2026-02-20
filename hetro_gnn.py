# hetero_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from rdflib import Graph, Namespace, RDF, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------
# 1️⃣ RDF -> HeteroData
# ---------------------------------------------------

def rdf_to_hetero_graph_auto(rdf_path):

    g = Graph()
    g.parse(rdf_path, format="turtle")
    EX = Namespace("http://example.org/")

    data = HeteroData()

    subjects = list(g.subjects(RDF.type, EX.Subject))
    subj_idx_map = {}
    subj_features, subj_labels = [], []

    numeric_predicates = set()

    for s in subjects:
        for p, o in g.predicate_objects(s):
            if isinstance(o, Literal):
                try:
                    float(str(o))
                    if not str(p).endswith("hasDiagnosis"):
                        numeric_predicates.add(p)
                except:
                    pass

    if len(numeric_predicates) == 0:
        possible = [EX.hasAge, EX.hasSex]
        numeric_predicates = set(
            [p for p in possible if any(g.objects(s, p) for s in subjects)]
        )

    numeric_predicates = sorted(list(numeric_predicates))
    feature_names = [str(p).split('/')[-1] for p in numeric_predicates]

    print(f"Detected numeric features: {feature_names}")

    for i, s in enumerate(subjects):

        subj_idx_map[str(s).split('/')[-1]] = i
        feats = []

        for p in numeric_predicates:
            try:
                val = next(g.objects(s, p), 0.0)
                feats.append(float(val))
            except:
                feats.append(0.0)

        subj_features.append(feats)

        dx_obj = next(g.objects(s, EX.hasDiagnosis), None)
        if dx_obj is None:
            raise RuntimeError(f"No diagnosis found for subject {s}")

        dx = int(str(dx_obj)) - 1
        subj_labels.append(dx)

    subj_features = np.array(subj_features, dtype=float)
    subj_features = StandardScaler().fit_transform(subj_features)

    data['subject'].x = torch.tensor(subj_features, dtype=torch.float)
    data['subject'].y = torch.tensor(subj_labels, dtype=torch.long)

    rois = set()
    for s in subjects:
        for r in g.objects(s, EX.connectedToROI):
            rois.add(r)

    rois = list(rois)
    roi_idx_map = {str(r).split('/')[-1]: i for i, r in enumerate(rois)}

    data['roi'].x = torch.zeros((len(rois), 3), dtype=torch.float) \
        if len(rois) > 0 else torch.zeros((0, 3))

    # subject -> ROI edges
    edge_subj_roi_src, edge_subj_roi_tgt = [], []

    for s in subjects:
        s_idx = subj_idx_map[str(s).split('/')[-1]]
        for r in g.objects(s, EX.connectedToROI):
            r_idx = roi_idx_map[str(r).split('/')[-1]]
            edge_subj_roi_src.append(s_idx)
            edge_subj_roi_tgt.append(r_idx)

    data['subject', 'connectedToROI', 'roi'].edge_index = (
        torch.tensor([edge_subj_roi_src, edge_subj_roi_tgt], dtype=torch.long)
        if edge_subj_roi_src else torch.empty((2, 0), dtype=torch.long)
    )

    # ROI -> ROI edges
    edge_roi_roi_src, edge_roi_roi_tgt = [], []

    for r1 in rois:
        for r2 in g.objects(r1, EX.connectedToROI):
            r1_idx = roi_idx_map[str(r1).split('/')[-1]]
            r2_idx = roi_idx_map.get(str(r2).split('/')[-1])
            if r2_idx is not None:
                edge_roi_roi_src.append(r1_idx)
                edge_roi_roi_tgt.append(r2_idx)

    if edge_roi_roi_src:
        data['roi', 'connectedToROI', 'roi'].edge_index = \
            torch.tensor([edge_roi_roi_src, edge_roi_roi_tgt], dtype=torch.long)

    # subject -> subject edges
    edge_subj_subj_src, edge_subj_subj_tgt = [], []

    for s in subjects:
        s_idx = subj_idx_map[str(s).split('/')[-1]]
        for s2 in g.objects(s, EX.similarTo):
            s2_key = str(s2).split('/')[-1]
            if s2_key in subj_idx_map:
                s2_idx = subj_idx_map[s2_key]
                edge_subj_subj_src.append(s_idx)
                edge_subj_subj_tgt.append(s2_idx)

    if edge_subj_subj_src:
        data['subject', 'similarTo', 'subject'].edge_index = \
            torch.tensor([edge_subj_subj_src, edge_subj_subj_tgt], dtype=torch.long)

    return data, feature_names


# ---------------------------------------------------
# 2️⃣ HeteroGNN Model
# ---------------------------------------------------

class HeteroGNN(nn.Module):

    def __init__(self, metadata, hidden_channels=64, out_channels=2):
        super().__init__()

        self.conv1 = HeteroConv({
            ('subject', 'connectedToROI', 'roi'): SAGEConv((-1, -1), hidden_channels),
            ('roi', 'connectedToROI', 'roi'): SAGEConv((-1, -1), hidden_channels),
            ('subject', 'similarTo', 'subject'): SAGEConv((-1, -1), hidden_channels)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('subject', 'connectedToROI', 'roi'): SAGEConv((-1, -1), hidden_channels),
            ('roi', 'connectedToROI', 'roi'): SAGEConv((-1, -1), hidden_channels),
            ('subject', 'similarTo', 'subject'): SAGEConv((-1, -1), hidden_channels)
        }, aggr='mean')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return self.lin(x_dict['subject'])


# ---------------------------------------------------
# 3️⃣ Training
# ---------------------------------------------------

def train_hetero_model(data, epochs=500, lr=1e-3,
                       weight_decay=5e-5, patience=70):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = HeteroGNN(data.metadata()).to(device)

    data = data.clone()

    for k, v in data.x_dict.items():
        data.x_dict[k] = v.to(device)

    for k, v in data.edge_index_dict.items():
        data.edge_index_dict[k] = v.to(device)

    data['subject'].y = data['subject'].y.to(device)

    y = data['subject'].y
    y_cpu = y.cpu().numpy()

    train_idx, test_idx = train_test_split(
        list(range(len(y_cpu))),
        test_size=0.2,
        stratify=y_cpu,
        random_state=42
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.25,
        stratify=y_cpu[train_idx],
        random_state=42
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    train_mask = torch.zeros_like(y, dtype=torch.bool)
    val_mask = torch.zeros_like(y, dtype=torch.bool)
    test_mask = torch.zeros_like(y, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_out = out[val_mask].argmax(dim=1)
            val_f1 = f1_score(
                y[val_mask].cpu(),
                val_out.cpu(),
                average='macro'
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                k: v.cpu()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}, Val F1 {val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, test_mask, data