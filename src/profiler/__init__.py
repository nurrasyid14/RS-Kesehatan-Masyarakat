'''
src/profiler
============
Profiling pasca-clustering: representasi konten dan ringkasan regional.

Kelas
-----
ContentRepresentation   : feature vectors & similarity matrix
RegionalProfiler        : semantic profiles, policy tags, SDG mapping

Penggunaan cepat
----------------
from src.profiler import ContentRepresentation, RegionalProfiler

cr  = ContentRepresentation(scaled_mart_df)
sim = cr.compute_similarity_matrix()

rp  = RegionalProfiler(feature_mart_df, cluster_labels)
profiles = rp.generate_cluster_profiles()
tags     = rp.generate_policy_tags()
'''

from .content_representation import ContentRepresentation
from .regional_profiler      import RegionalProfiler

__all__ = ["ContentRepresentation", "RegionalProfiler"]