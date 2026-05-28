'''
src/recommender
===============
Hybrid Healthcare Recommendation System.

Tiga engine yang bekerja berlapis:

    ContentBasedRecommender  — kemiripan profil fitur wilayah
    CaseBasedRecommender     — memori kasus kebijakan sebelumnya
    HybridRecommender        — gabungan berbobot: R = α·R_c + β·R_k

Referensi
---------
Aggarwal, C. C. (2016). Recommender Systems: The Textbook.
    Springer. Ch. 4 (Content-Based) & Ch. 9 (Hybrid).

Penggunaan cepat
----------------
from src.recommender import HybridRecommender

rec = HybridRecommender(feature_mart_df, scaled_mart_df)
recs = rec.generate_final_recommendations("Papua")
'''

from .content_based  import ContentBasedRecommender
from .case_based     import CaseBasedRecommender
from .hybrid         import HybridRecommender

__all__ = [
    "ContentBasedRecommender",
    "CaseBasedRecommender",
    "HybridRecommender",
]