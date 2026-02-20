# graph_builder.py

import torch
import networkx as nx
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal


def build_subject_graph_gpu(nifti_file, atlas_path, confounds=None, threshold=0.3):
    masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)
    time_series = masker.fit_transform(nifti_file, confounds=confounds)
    time_series = signal.clean(time_series, standardize="zscore_sample")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.tensor(time_series, dtype=torch.float32, device=device)

    corr = torch.corrcoef(ts.T)
    corr[torch.abs(corr) < threshold] = 0
    corr = corr.cpu().numpy()

    G = nx.from_numpy_array(corr)
    return G, corr