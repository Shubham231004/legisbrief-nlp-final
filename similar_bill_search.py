# Semantic similar-bill search for LegisBrief-NLP.

from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel


class SimilarBillSearch:
    def __init__(
        self,
        artifact_directory,
        device="cpu"
    ):
        self.artifact_directory = Path(
            artifact_directory
        )

        self.model = SentenceTransformer(
            str(
                self.artifact_directory
                / "sentence_transformer_model"
            ),
            device=device
        )

        self.metadata = pd.read_csv(
            self.artifact_directory
            / "bill_metadata.csv",
            dtype={
                "document_id": str,
                "text_hash": str
            }
        )

        self.faiss_index = faiss.read_index(
            str(
                self.artifact_directory
                / "faiss_index.bin"
            )
        )

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    @staticmethod
    def _normalize_text(value):
        if value is None:
            return ""

        return " ".join(
            str(value).split()
        ).strip()

    @classmethod
    def build_query(
        cls,
        title,
        summary_or_text
    ):
        title = cls._normalize_text(
            title
        )

        content = cls._normalize_text(
            summary_or_text
        )

        parts = []

        if title:
            parts.append(
                f"title: {title}"
            )

        if content:
            parts.append(
                f"summary: {content}"
            )

        return "\n".join(parts)

    def semantic_search(
        self,
        title,
        summary_or_text,
        top_k=5,
        exclude_document_id=None
    ):
        query_text = self.build_query(
            title,
            summary_or_text
        )

        if not query_text:
            raise ValueError(
                "The search query is blank."
            )

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(
            np.float32,
            copy=False
        )

        search_count = min(
            int(
                self.faiss_index.ntotal
            ),
            max(
                top_k + 25,
                top_k * 5
            )
        )

        scores, indices = (
            self.faiss_index.search(
                query_embedding,
                search_count
            )
        )

        excluded_id = (
            str(
                exclude_document_id
            )
            if exclude_document_id
            is not None
            else None
        )

        rows = []

        for score, index_value in zip(
            scores[0],
            indices[0]
        ):
            if index_value < 0:
                continue

            metadata_row = (
                self.metadata.iloc[
                    int(
                        index_value
                    )
                ]
            )

            document_id = str(
                metadata_row[
                    "document_id"
                ]
            )

            if (
                excluded_id is not None
                and document_id
                == excluded_id
            ):
                continue

            row = {
                "rank": (
                    len(rows) + 1
                ),
                "document_id": (
                    document_id
                ),
                "title": str(
                    metadata_row.get(
                        "title",
                        ""
                    )
                ),
                "summary": str(
                    metadata_row.get(
                        "reference_summary",
                        ""
                    )
                ),
                "prepared_split": str(
                    metadata_row.get(
                        "prepared_split",
                        ""
                    )
                ),
                "similarity_score": float(
                    score
                )
            }

            if (
                "text_hash"
                in metadata_row.index
            ):
                row["text_hash"] = str(
                    metadata_row.get(
                        "text_hash",
                        ""
                    )
                )

            rows.append(
                row
            )

            if len(rows) >= top_k:
                break

        return pd.DataFrame(
            rows
        )

    def _ensure_tfidf_loaded(
        self
    ):
        if (
            self.tfidf_vectorizer
            is not None
            and self.tfidf_matrix
            is not None
        ):
            return

        self.tfidf_vectorizer = (
            joblib.load(
                self.artifact_directory
                / "tfidf_vectorizer.joblib"
            )
        )

        self.tfidf_matrix = (
            sparse.load_npz(
                self.artifact_directory
                / "tfidf_matrix.npz"
            )
        )

    def tfidf_search(
        self,
        title,
        summary_or_text,
        top_k=5,
        exclude_document_id=None
    ):
        self._ensure_tfidf_loaded()

        query_text = self.build_query(
            title,
            summary_or_text
        )

        if not query_text:
            raise ValueError(
                "The search query is blank."
            )

        query_vector = (
            self.tfidf_vectorizer
            .transform(
                [query_text]
            )
        )

        scores = linear_kernel(
            query_vector,
            self.tfidf_matrix
        ).ravel()

        candidate_count = min(
            len(scores),
            max(
                top_k + 25,
                top_k * 5
            )
        )

        if candidate_count <= 0:
            return pd.DataFrame()

        candidate_indices = (
            np.argpartition(
                -scores,
                candidate_count - 1
            )[
                :candidate_count
            ]
        )

        candidate_indices = (
            candidate_indices[
                np.argsort(
                    -scores[
                        candidate_indices
                    ]
                )
            ]
        )

        excluded_id = (
            str(
                exclude_document_id
            )
            if exclude_document_id
            is not None
            else None
        )

        rows = []

        for matrix_index in (
            candidate_indices
        ):
            metadata_row = (
                self.metadata.iloc[
                    int(
                        matrix_index
                    )
                ]
            )

            document_id = str(
                metadata_row[
                    "document_id"
                ]
            )

            if (
                excluded_id is not None
                and document_id
                == excluded_id
            ):
                continue

            rows.append(
                {
                    "rank": (
                        len(rows) + 1
                    ),
                    "document_id": (
                        document_id
                    ),
                    "title": str(
                        metadata_row.get(
                            "title",
                            ""
                        )
                    ),
                    "summary": str(
                        metadata_row.get(
                            "reference_summary",
                            ""
                        )
                    ),
                    "prepared_split": str(
                        metadata_row.get(
                            "prepared_split",
                            ""
                        )
                    ),
                    "similarity_score": float(
                        scores[
                            matrix_index
                        ]
                    )
                }
            )

            if len(rows) >= top_k:
                break

        return pd.DataFrame(
            rows
        )
