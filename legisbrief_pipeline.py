# Final generalized grounded pipeline for LegisBrief-NLP.

import gc
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from similar_bill_search import SimilarBillSearch


DISCLAIMER_TEXT = (
    "This output is generated from the supplied bill "
    "text for informational purposes only. It is not "
    "legal advice and should be checked against the "
    "official legislative text."
)


POLICY_FUNCTIONS = {
    "Requirement": (
        "This provision creates a legal duty or requires "
        "a person, organization, business, agency, or "
        "government body to take or avoid an action."
    ),
    "Prohibition": (
        "This provision prohibits, restricts, bars, or "
        "makes an action unlawful."
    ),
    "Authorization": (
        "This provision grants legal authority, "
        "permission, discretion, or power to an actor."
    ),
    "Right or protection": (
        "This provision gives a person or group a right, "
        "protection, benefit, remedy, or legal safeguard."
    ),
    "Eligibility or coverage": (
        "This provision changes who or what qualifies, "
        "is covered, is excluded, or falls within the "
        "scope of the law."
    ),
    "Funding or taxation": (
        "This provision appropriates money, changes "
        "government spending, creates funding, imposes "
        "a fee, or changes a tax."
    ),
    "Deadline or duration": (
        "This provision sets a deadline, waiting period, "
        "effective date, extension, expiration, or "
        "duration."
    ),
    "Enforcement or penalty": (
        "This provision creates enforcement authority, "
        "a violation, a penalty, a fine, liability, or "
        "a legal remedy."
    ),
    "Reporting or oversight": (
        "This provision requires a report, study, audit, "
        "review, notice, disclosure, record, or oversight "
        "action."
    ),
    "Program or agency change": (
        "This provision creates, changes, transfers, "
        "reorganizes, or ends a program, office, agency, "
        "regulation, or administrative responsibility."
    ),
    "Amendment or repeal": (
        "This provision amends, repeals, replaces, or "
        "makes a substantive change to an existing law "
        "or legal rule."
    ),
    "Designation or naming": (
        "This provision officially designates, names, "
        "renames, recognizes, or commemorates a place, "
        "facility, event, object, or program."
    )
}


NON_POLICY_FUNCTIONS = {
    "Purpose or background": (
        "This sentence only states a purpose, finding, "
        "background fact, reason, or general policy goal "
        "without independently creating a legal effect."
    ),
    "Definition only": (
        "This sentence only defines a term or explains "
        "what a word or phrase means or includes."
    ),
    "Example or detail only": (
        "This sentence only provides an example, list, "
        "illustration, or supporting detail and does not "
        "independently create a legal effect."
    )
}


STAKEHOLDER_ROLES = {
    "Receives a right, protection, benefit, or remedy": (
        "The identified actor receives a right, "
        "protection, benefit, eligibility, payment, "
        "service, remedy, or legal safeguard under this "
        "provision."
    ),
    "Has a legal duty or requirement": (
        "The identified actor is legally required to "
        "take an action or comply with a duty under this "
        "provision."
    ),
    "Is prohibited, restricted, or regulated": (
        "The identified actor is prohibited, restricted, "
        "regulated, limited, or made subject to a legal "
        "condition under this provision."
    ),
    "Administers, enforces, or oversees": (
        "The identified actor administers, enforces, "
        "investigates, regulates, supervises, or oversees "
        "the law or program in this provision."
    ),
    "Pays, receives, or controls funding": (
        "The identified actor pays, receives, distributes, "
        "administers, or controls money, grants, taxes, "
        "fees, or appropriations under this provision."
    ),
    "Must report, disclose, notify, or keep records": (
        "The identified actor must report, disclose, "
        "notify, study, audit, review, or keep records "
        "under this provision."
    ),
    "Is covered or eligible": (
        "The identified actor is included, covered, "
        "eligible, excluded, or otherwise placed within "
        "the scope of this provision."
    )
}


class LegisBriefPipeline:
    def __init__(
        self,
        project_root,
        device=None
    ):
        self.project_root = Path(
            project_root
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        project_config_path = (
            self.project_root
            / "project_config.json"
        )

        generation_config_path = (
            self.project_root
            / "outputs"
            / "metrics"
            / "t5_selected_generation_config.json"
        )

        with project_config_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.project_config = (
                json.load(file)
            )

        with generation_config_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.generation_config = (
                json.load(file)
            )

        self.max_source_length = int(
            self.project_config[
                "maximum_input_tokens"
            ]
        )

        self.max_target_length = int(
            self.project_config[
                "maximum_target_tokens"
            ]
        )

        self.summary_model_path = (
            self.project_root
            / "models"
            / "t5_small_legisbrief_final"
        )

        self.similarity_artifact_path = (
            self.project_root
            / "artifacts"
            / "similarity_search"
        )

        self.grounded_artifact_path = (
            self.project_root
            / "artifacts"
            / "grounded_inference"
        )

        self.nli_model_path = (
            self.grounded_artifact_path
            / "nli_model"
        )

        self.plain_model_path = (
            self.grounded_artifact_path
            / "plain_english_model"
        )

        self.spacy_model_path = (
            self.grounded_artifact_path
            / "spacy_en_core_web_sm"
        )

        self.summary_tokenizer = (
            AutoTokenizer.from_pretrained(
                str(
                    self.summary_model_path
                )
            )
        )

        self.nli_tokenizer = (
            AutoTokenizer.from_pretrained(
                str(
                    self.nli_model_path
                )
            )
        )

        self.plain_tokenizer = (
            AutoTokenizer.from_pretrained(
                str(
                    self.plain_model_path
                )
            )
        )

        self.nli_config = (
            AutoConfig.from_pretrained(
                str(
                    self.nli_model_path
                )
            )
        )

        self._set_nli_label_indices()

        self.language_processor = (
            spacy.load(
                str(
                    self.spacy_model_path
                )
            )
        )

        self.language_processor.max_length = (
            3_000_000
        )

        self.similar_bill_search = (
            SimilarBillSearch(
                artifact_directory=(
                    self.similarity_artifact_path
                ),
                device=self.device.type
            )
        )

        # Reuse the already-loaded MPNet model for
        # within-bill evidence retrieval and semantic
        # legislative-function prefiltering.
        self.evidence_model = (
            self.similar_bill_search.model
        )

        self.policy_names = list(
            POLICY_FUNCTIONS.keys()
        )

        self.policy_descriptions = [
            POLICY_FUNCTIONS[
                policy_name
            ]
            for policy_name
            in self.policy_names
        ]

        self.non_policy_names = list(
            NON_POLICY_FUNCTIONS.keys()
        )

        self.non_policy_descriptions = [
            NON_POLICY_FUNCTIONS[
                label_name
            ]
            for label_name
            in self.non_policy_names
        ]

        self.policy_label_embeddings = (
            self.evidence_model.encode(
                self.policy_descriptions,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                np.float32,
                copy=False
            )
        )

        self.non_policy_label_embeddings = (
            self.evidence_model.encode(
                self.non_policy_descriptions,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                np.float32,
                copy=False
            )
        )

        self.summary_model = None
        self.nli_model = None
        self.plain_model = None

        self.chunk_token_limit = min(
            760,
            max(
                384,
                self.max_source_length - 180
            )
        )

        self.chunk_overlap_sentences = 2

        self.maximum_policy_nli_sentences = (
            160
        )

        self.maximum_summary_claims = 5
        self.minimum_summary_claims = 3
        self.maximum_policy_changes = 7
        self.maximum_affected_groups = 8

    def _set_nli_label_indices(
        self
    ):
        raw_labels = {
            int(label_id): str(
                label_name
            ).casefold()
            for label_id, label_name
            in self.nli_config.id2label.items()
        }

        def find_index(
            keyword,
            fallback
        ):
            for label_id, label_name in (
                raw_labels.items()
            ):
                if keyword in label_name:
                    return int(
                        label_id
                    )

            return int(
                fallback
            )

        self.contradiction_index = (
            find_index(
                "contradiction",
                0
            )
        )

        self.entailment_index = (
            find_index(
                "entailment",
                1
            )
        )

        self.neutral_index = (
            find_index(
                "neutral",
                2
            )
        )

    @staticmethod
    def normalize_text(
        value
    ):
        if value is None:
            return ""

        text = str(
            value
        )

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\s*\n+\s*",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    @staticmethod
    def normalize_title(
        value
    ):
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            LegisBriefPipeline
            .normalize_text(
                value
            )
            .casefold()
        ).strip()

    @staticmethod
    def count_words(
        value
    ):
        return len(
            LegisBriefPipeline
            .normalize_text(
                value
            )
            .split()
        )

    @staticmethod
    def ensure_sentence_ending(
        sentence
    ):
        sentence = (
            LegisBriefPipeline
            .normalize_text(
                sentence
            )
        )

        if (
            sentence
            and sentence[-1]
            not in ".!?"
        ):
            sentence += "."

        return sentence

    @staticmethod
    def split_sentences(
        value
    ):
        text = (
            LegisBriefPipeline
            .normalize_text(
                value
            )
        )

        if not text:
            return []

        protected = text

        abbreviations = {
            "U.S.": "U§S§",
            "U.S.C.": "U§S§C§",
            "Sec.": "Sec§",
            "No.": "No§",
            "Mr.": "Mr§",
            "Mrs.": "Mrs§",
            "Ms.": "Ms§",
            "Dr.": "Dr§"
        }

        for original, replacement in (
            abbreviations.items()
        ):
            protected = (
                protected.replace(
                    original,
                    replacement
                )
            )

        parts = re.split(
            r"(?<=[.!?])\s+"
            r"(?=[\"'\(\[]?[A-Z0-9])|"
            r"(?<=;)\s+"
            r"(?=[A-Z0-9])",
            protected
        )

        sentences = []

        for part in parts:
            for original, replacement in (
                abbreviations.items()
            ):
                part = part.replace(
                    replacement,
                    original
                )

            part = (
                LegisBriefPipeline
                .normalize_text(
                    part
                )
                .strip(
                    " -•\t"
                )
            )

            if (
                part
                and len(
                    part.split()
                ) >= 3
            ):
                sentences.append(
                    part
                )

        return sentences

    @staticmethod
    def sentence_word_set(
        sentence
    ):
        return {
            word
            for word in re.findall(
                r"[A-Za-z0-9']+",
                sentence.casefold()
            )
            if len(word) > 2
        }

    @classmethod
    def sentence_jaccard(
        cls,
        sentence_a,
        sentence_b
    ):
        words_a = (
            cls.sentence_word_set(
                sentence_a
            )
        )

        words_b = (
            cls.sentence_word_set(
                sentence_b
            )
        )

        if (
            not words_a
            or not words_b
        ):
            return 0.0

        return (
            len(
                words_a
                & words_b
            )
            / len(
                words_a
                | words_b
            )
        )

    def _release_model(
        self,
        model_name
    ):
        if model_name == "summary":
            self.summary_model = None

        elif model_name == "nli":
            self.nli_model = None

        elif model_name == "plain":
            self.plain_model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_summary_model(
        self
    ):
        if self.summary_model is None:
            self.summary_model = (
                AutoModelForSeq2SeqLM
                .from_pretrained(
                    str(
                        self.summary_model_path
                    )
                )
            )

            self.summary_model.config.use_cache = (
                True
            )

            self.summary_model.to(
                self.device
            )

            if self.device.type == "cuda":
                self.summary_model.half()

            self.summary_model.eval()

        return self.summary_model

    def _load_nli_model(
        self
    ):
        if self.nli_model is None:
            self.nli_model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    str(
                        self.nli_model_path
                    )
                )
            )

            self.nli_model.to(
                self.device
            )

            self.nli_model.eval()

        return self.nli_model

    def _load_plain_model(
        self
    ):
        if self.plain_model is None:
            self.plain_model = (
                AutoModelForSeq2SeqLM
                .from_pretrained(
                    str(
                        self.plain_model_path
                    )
                )
            )

            self.plain_model.to(
                self.device
            )

            if self.device.type == "cuda":
                self.plain_model.half()

            self.plain_model.eval()

        return self.plain_model

    def _generation_kwargs(
        self
    ):
        config = self.generation_config

        kwargs = {
            "max_new_tokens": int(
                config.get(
                    "max_new_tokens",
                    256
                )
            ),
            "min_new_tokens": int(
                config.get(
                    "min_new_tokens",
                    16
                )
            ),
            "num_beams": int(
                config.get(
                    "num_beams",
                    1
                )
            ),
            "do_sample": False,
            "no_repeat_ngram_size": int(
                config.get(
                    "no_repeat_ngram_size",
                    3
                )
            ),
            "repetition_penalty": float(
                config.get(
                    "repetition_penalty",
                    1.0
                )
            ),
            "pad_token_id": (
                self.summary_tokenizer
                .pad_token_id
            ),
            "eos_token_id": (
                self.summary_tokenizer
                .eos_token_id
            )
        }

        if kwargs["num_beams"] > 1:
            kwargs[
                "length_penalty"
            ] = float(
                config.get(
                    "length_penalty",
                    1.0
                )
            )

            kwargs[
                "early_stopping"
            ] = True

        return kwargs

    def _summarize_texts(
        self,
        input_texts,
        batch_size=2
    ):
        if not input_texts:
            return []

        model = (
            self._load_summary_model()
        )

        outputs = []

        for batch_start in range(
            0,
            len(input_texts),
            batch_size
        ):
            batch_texts = input_texts[
                batch_start:
                batch_start + batch_size
            ]

            encoded = (
                self.summary_tokenizer(
                    batch_texts,
                    max_length=(
                        self.max_source_length
                    ),
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                generated_ids = (
                    model.generate(
                        **encoded,
                        **self._generation_kwargs()
                    )
                )

            decoded = (
                self.summary_tokenizer
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            )

            outputs.extend(
                [
                    self.normalize_text(
                        output
                    )
                    for output
                    in decoded
                ]
            )

            del encoded
            del generated_ids

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs

    def _sentence_token_count(
        self,
        sentence
    ):
        return len(
            self.summary_tokenizer(
                sentence,
                add_special_tokens=False
            )[
                "input_ids"
            ]
        )

    def _split_bill_into_chunks(
        self,
        bill_text
    ):
        sentences = self.split_sentences(
            bill_text
        )

        if not sentences:
            return [
                self.normalize_text(
                    bill_text
                )
            ]

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = (
                self._sentence_token_count(
                    sentence
                )
            )

            if (
                sentence_tokens
                > self.chunk_token_limit
            ):
                if current_sentences:
                    chunks.append(
                        " ".join(
                            current_sentences
                        )
                    )

                    current_sentences = []
                    current_tokens = 0

                token_ids = (
                    self.summary_tokenizer(
                        sentence,
                        add_special_tokens=False
                    )[
                        "input_ids"
                    ]
                )

                overlap_tokens = min(
                    96,
                    self.chunk_token_limit
                    // 6
                )

                step = max(
                    1,
                    self.chunk_token_limit
                    - overlap_tokens
                )

                for start_index in range(
                    0,
                    len(token_ids),
                    step
                ):
                    piece = token_ids[
                        start_index:
                        start_index
                        + self.chunk_token_limit
                    ]

                    chunks.append(
                        self.normalize_text(
                            self.summary_tokenizer
                            .decode(
                                piece,
                                skip_special_tokens=True
                            )
                        )
                    )

                    if (
                        start_index
                        + self.chunk_token_limit
                        >= len(token_ids)
                    ):
                        break

                continue

            if (
                current_sentences
                and current_tokens
                + sentence_tokens
                > self.chunk_token_limit
            ):
                chunks.append(
                    " ".join(
                        current_sentences
                    )
                )

                overlap = (
                    current_sentences[
                        -self.chunk_overlap_sentences:
                    ]
                )

                current_sentences = list(
                    overlap
                )

                current_tokens = sum(
                    self._sentence_token_count(
                        overlap_sentence
                    )
                    for overlap_sentence
                    in current_sentences
                )

                while (
                    current_sentences
                    and current_tokens
                    + sentence_tokens
                    > self.chunk_token_limit
                ):
                    removed = (
                        current_sentences
                        .pop(0)
                    )

                    current_tokens -= (
                        self._sentence_token_count(
                            removed
                        )
                    )

            current_sentences.append(
                sentence
            )

            current_tokens += (
                sentence_tokens
            )

        if current_sentences:
            chunks.append(
                " ".join(
                    current_sentences
                )
            )

        return [
            self.normalize_text(
                chunk
            )
            for chunk in chunks
            if self.normalize_text(
                chunk
            )
        ]

    def _hierarchical_t5_draft(
        self,
        title,
        bill_text
    ):
        chunks = (
            self._split_bill_into_chunks(
                bill_text
            )
        )

        chunk_inputs = []

        for chunk in chunks:
            parts = [
                "summarize:"
            ]

            if title:
                parts.append(
                    f"title: {title}"
                )

            parts.append(
                f"bill text: {chunk}"
            )

            chunk_inputs.append(
                "\n".join(
                    parts
                )
            )

        summaries = (
            self._summarize_texts(
                chunk_inputs,
                batch_size=2
            )
        )

        reduction_levels = 0

        while len(summaries) > 1:
            reduction_levels += 1

            groups = []
            current_group = []
            current_tokens = 0

            token_budget = max(
                360,
                self.max_source_length
                - 180
            )

            for summary in summaries:
                token_count = (
                    self._sentence_token_count(
                        summary
                    )
                )

                if (
                    current_group
                    and current_tokens
                    + token_count
                    > token_budget
                ):
                    groups.append(
                        " ".join(
                            current_group
                        )
                    )

                    current_group = []
                    current_tokens = 0

                current_group.append(
                    summary
                )

                current_tokens += (
                    token_count
                )

            if current_group:
                groups.append(
                    " ".join(
                        current_group
                    )
                )

            reduction_inputs = []

            for group in groups:
                parts = [
                    "summarize:"
                ]

                if title:
                    parts.append(
                        f"title: {title}"
                    )

                parts.append(
                    (
                        "bill section summaries: "
                        f"{group}"
                    )
                )

                reduction_inputs.append(
                    "\n".join(
                        parts
                    )
                )

            summaries = (
                self._summarize_texts(
                    reduction_inputs,
                    batch_size=2
                )
            )

            if reduction_levels >= 8:
                break

        draft_summary = (
            summaries[0]
            if summaries
            else ""
        )

        self._release_model(
            "summary"
        )

        return {
            "draft_summary": (
                self.normalize_text(
                    draft_summary
                )
            ),
            "chunks": chunks,
            "chunk_count": len(
                chunks
            ),
            "reduction_levels": (
                reduction_levels
            )
        }

    def _build_evidence_store(
        self,
        bill_text
    ):
        all_sentences = (
            self.split_sentences(
                bill_text
            )
        )

        evidence_sentences = [
            self.ensure_sentence_ending(
                sentence
            )
            for sentence in all_sentences
            if (
                5
                <= self.count_words(
                    sentence
                )
                <= 150
            )
        ]

        if not evidence_sentences:
            evidence_sentences = [
                self.ensure_sentence_ending(
                    bill_text
                )
            ]

        embeddings = (
            self.evidence_model.encode(
                evidence_sentences,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                np.float32,
                copy=False
            )
        )

        index = faiss.IndexFlatIP(
            int(
                embeddings.shape[1]
            )
        )

        index.add(
            embeddings
        )

        return {
            "sentences": (
                evidence_sentences
            ),
            "embeddings": (
                embeddings
            ),
            "index": index
        }

    def _retrieve_evidence(
        self,
        claim,
        evidence_store,
        top_k=4
    ):
        query_embedding = (
            self.evidence_model.encode(
                [claim],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                np.float32,
                copy=False
            )
        )

        search_count = min(
            top_k,
            len(
                evidence_store[
                    "sentences"
                ]
            )
        )

        scores, indices = (
            evidence_store[
                "index"
            ].search(
                query_embedding,
                search_count
            )
        )

        rows = []

        for score, index_value in zip(
            scores[0],
            indices[0]
        ):
            if index_value < 0:
                continue

            rows.append(
                {
                    "sentence": (
                        evidence_store[
                            "sentences"
                        ][
                            int(
                                index_value
                            )
                        ]
                    ),
                    "similarity": float(
                        score
                    )
                }
            )

        return rows

    def _score_nli_pairs(
        self,
        premises,
        hypotheses,
        batch_size=16
    ):
        if len(premises) != len(
            hypotheses
        ):
            raise ValueError(
                "Premise and hypothesis counts "
                "must match."
            )

        if not premises:
            return np.empty(
                (
                    0,
                    int(
                        self.nli_config
                        .num_labels
                    )
                ),
                dtype=np.float32
            )

        model = (
            self._load_nli_model()
        )

        probabilities = []

        for batch_start in range(
            0,
            len(premises),
            batch_size
        ):
            batch_premises = premises[
                batch_start:
                batch_start + batch_size
            ]

            batch_hypotheses = (
                hypotheses[
                    batch_start:
                    batch_start
                    + batch_size
                ]
            )

            encoded = (
                self.nli_tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                logits = (
                    model(
                        **encoded
                    ).logits
                )

                batch_probabilities = (
                    F.softmax(
                        logits,
                        dim=-1
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            probabilities.append(
                batch_probabilities
            )

            del encoded
            del logits

        return np.vstack(
            probabilities
        )

    def _verify_rewrite(
        self,
        source_sentence,
        rewritten_sentence
    ):
        probabilities = (
            self._score_nli_pairs(
                [source_sentence],
                [rewritten_sentence]
            )[0]
        )

        entailment = float(
            probabilities[
                self.entailment_index
            ]
        )

        contradiction = float(
            probabilities[
                self.contradiction_index
            ]
        )

        return {
            "entailment": entailment,
            "contradiction": (
                contradiction
            ),
            "accepted": (
                entailment >= 0.50
                and contradiction <= 0.30
            )
        }

    def _has_complete_clause(
        self,
        sentence
    ):
        document = (
            self.language_processor(
                sentence
            )
        )

        has_verb = any(
            token.pos_
            in {
                "VERB",
                "AUX"
            }
            for token in document
        )

        has_actor_or_object = any(
            token.dep_
            in {
                "nsubj",
                "nsubjpass",
                "dobj",
                "obj",
                "pobj",
                "agent",
                "attr",
                "oprd"
            }
            and token.pos_
            in {
                "NOUN",
                "PROPN",
                "PRON"
            }
            for token in document
        )

        return bool(
            has_verb
            and has_actor_or_object
        )

    def _choose_policy_candidates(
        self,
        semantic_policy_scores,
        semantic_non_policy_scores
    ):
        sentence_count = len(
            semantic_policy_scores
        )

        maximum_candidates = min(
            self.maximum_policy_nli_sentences,
            sentence_count
        )

        if (
            sentence_count
            <= maximum_candidates
        ):
            return np.arange(
                sentence_count,
                dtype=int
            )

        ranking_score = (
            semantic_policy_scores
            - 0.30
            * semantic_non_policy_scores
        )

        global_count = max(
            1,
            maximum_candidates - 60
        )

        selected = set(
            int(index_value)
            for index_value in (
                np.argsort(
                    -ranking_score
                )[
                    :global_count
                ]
            )
        )

        bin_count = min(
            20,
            sentence_count
        )

        for bin_indices in (
            np.array_split(
                np.arange(
                    sentence_count
                ),
                bin_count
            )
        ):
            if len(
                bin_indices
            ) == 0:
                continue

            local_count = min(
                3,
                len(
                    bin_indices
                )
            )

            local_ranked = (
                bin_indices[
                    np.argsort(
                        -ranking_score[
                            bin_indices
                        ]
                    )[
                        :local_count
                    ]
                ]
            )

            selected.update(
                int(
                    index_value
                )
                for index_value
                in local_ranked
            )

        selected = sorted(
            selected,
            key=lambda index_value: (
                -ranking_score[
                    index_value
                ],
                index_value
            )
        )

        return np.array(
            selected[
                :maximum_candidates
            ],
            dtype=int
        )

    def _classify_policy_sentences(
        self,
        bill_text,
        title,
        draft_summary
    ):
        source_sentences = [
            self.ensure_sentence_ending(
                sentence
            )
            for sentence in (
                self.split_sentences(
                    bill_text
                )
            )
            if (
                6
                <= self.count_words(
                    sentence
                )
                <= 130
            )
        ]

        if not source_sentences:
            return pd.DataFrame()

        sentence_embeddings = (
            self.evidence_model.encode(
                source_sentences,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                np.float32,
                copy=False
            )
        )

        policy_similarity = (
            sentence_embeddings
            @ self.policy_label_embeddings.T
        )

        non_policy_similarity = (
            sentence_embeddings
            @ self.non_policy_label_embeddings.T
        )

        best_policy_semantic = (
            policy_similarity.max(
                axis=1
            )
        )

        best_non_policy_semantic = (
            non_policy_similarity.max(
                axis=1
            )
        )

        candidate_indices = (
            self._choose_policy_candidates(
                best_policy_semantic,
                best_non_policy_semantic
            )
        )

        premises = []
        hypotheses = []
        metadata = []

        for candidate_index in (
            candidate_indices
        ):
            candidate_index = int(
                candidate_index
            )

            sentence = (
                source_sentences[
                    candidate_index
                ]
            )

            top_policy_indices = (
                np.argsort(
                    -policy_similarity[
                        candidate_index
                    ]
                )[:2]
            )

            best_non_policy_index = int(
                np.argmax(
                    non_policy_similarity[
                        candidate_index
                    ]
                )
            )

            for policy_index in (
                top_policy_indices
            ):
                policy_index = int(
                    policy_index
                )

                premises.append(
                    sentence
                )

                hypotheses.append(
                    self.policy_descriptions[
                        policy_index
                    ]
                )

                metadata.append(
                    {
                        "candidate_index": (
                            candidate_index
                        ),
                        "kind": "policy",
                        "label_index": (
                            policy_index
                        )
                    }
                )

            premises.append(
                sentence
            )

            hypotheses.append(
                self.non_policy_descriptions[
                    best_non_policy_index
                ]
            )

            metadata.append(
                {
                    "candidate_index": (
                        candidate_index
                    ),
                    "kind": "non_policy",
                    "label_index": (
                        best_non_policy_index
                    )
                }
            )

        probabilities = (
            self._score_nli_pairs(
                premises,
                hypotheses,
                batch_size=16
            )
        )

        result_lookup = {}

        for pair_metadata, pair_probs in zip(
            metadata,
            probabilities
        ):
            candidate_index = int(
                pair_metadata[
                    "candidate_index"
                ]
            )

            entry = (
                result_lookup.setdefault(
                    candidate_index,
                    {
                        "policy_results": [],
                        "non_policy_results": []
                    }
                )
            )

            result = {
                "label_index": int(
                    pair_metadata[
                        "label_index"
                    ]
                ),
                "entailment": float(
                    pair_probs[
                        self.entailment_index
                    ]
                ),
                "contradiction": float(
                    pair_probs[
                        self.contradiction_index
                    ]
                )
            }

            if (
                pair_metadata[
                    "kind"
                ]
                == "policy"
            ):
                entry[
                    "policy_results"
                ].append(
                    result
                )

            else:
                entry[
                    "non_policy_results"
                ].append(
                    result
                )

        draft_embedding = None

        if draft_summary:
            draft_embedding = (
                self.evidence_model.encode(
                    [draft_summary],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
            )

        title_embedding = None

        if title:
            title_embedding = (
                self.evidence_model.encode(
                    [title],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )[0]
            )

        rows = []

        for candidate_index in (
            candidate_indices
        ):
            candidate_index = int(
                candidate_index
            )

            result_entry = (
                result_lookup.get(
                    candidate_index
                )
            )

            if not result_entry:
                continue

            policy_results = (
                result_entry[
                    "policy_results"
                ]
            )

            if not policy_results:
                continue

            best_policy_result = max(
                policy_results,
                key=lambda row: (
                    row["entailment"]
                )
            )

            non_policy_results = (
                result_entry[
                    "non_policy_results"
                ]
            )

            best_non_policy_result = (
                max(
                    non_policy_results,
                    key=lambda row: (
                        row["entailment"]
                    )
                )
                if non_policy_results
                else {
                    "label_index": 0,
                    "entailment": 0.0,
                    "contradiction": 0.0
                }
            )

            sentence = (
                source_sentences[
                    candidate_index
                ]
            )

            complete_clause = (
                self._has_complete_clause(
                    sentence
                )
            )

            policy_entailment = float(
                best_policy_result[
                    "entailment"
                ]
            )

            non_policy_entailment = float(
                best_non_policy_result[
                    "entailment"
                ]
            )

            semantic_score = float(
                policy_similarity[
                    candidate_index,
                    best_policy_result[
                        "label_index"
                    ]
                ]
            )

            draft_similarity = (
                float(
                    sentence_embeddings[
                        candidate_index
                    ]
                    @ draft_embedding
                )
                if draft_embedding
                is not None
                else 0.0
            )

            title_similarity = (
                float(
                    sentence_embeddings[
                        candidate_index
                    ]
                    @ title_embedding
                )
                if title_embedding
                is not None
                else 0.0
            )

            classification_score = (
                0.46
                * policy_entailment
                + 0.20
                * semantic_score
                + 0.18
                * draft_similarity
                + 0.08
                * title_similarity
                + (
                    0.08
                    if complete_clause
                    else 0.0
                )
                - 0.18
                * non_policy_entailment
            )

            is_policy = (
                (
                    complete_clause
                    and policy_entailment
                    >= 0.40
                    and non_policy_entailment
                    <= policy_entailment
                    + 0.08
                )
                or (
                    policy_entailment
                    >= 0.62
                    and non_policy_entailment
                    < 0.62
                )
            )

            rows.append(
                {
                    "sentence_position": (
                        candidate_index
                    ),
                    "source_sentence": (
                        sentence
                    ),
                    "policy_function": (
                        self.policy_names[
                            best_policy_result[
                                "label_index"
                            ]
                        ]
                    ),
                    "policy_entailment": (
                        policy_entailment
                    ),
                    "policy_contradiction": float(
                        best_policy_result[
                            "contradiction"
                        ]
                    ),
                    "non_policy_function": (
                        self.non_policy_names[
                            best_non_policy_result[
                                "label_index"
                            ]
                        ]
                    ),
                    "non_policy_entailment": (
                        non_policy_entailment
                    ),
                    "semantic_score": (
                        semantic_score
                    ),
                    "draft_similarity": (
                        draft_similarity
                    ),
                    "title_similarity": (
                        title_similarity
                    ),
                    "complete_clause": (
                        complete_clause
                    ),
                    "selection_score": float(
                        classification_score
                    ),
                    "is_policy": bool(
                        is_policy
                    )
                }
            )

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(
                rows
            )
            .sort_values(
                [
                    "is_policy",
                    "selection_score"
                ],
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

    def _select_diverse_policy_rows(
        self,
        policy_df,
        minimum_items=3,
        maximum_items=5
    ):
        if (
            policy_df is None
            or policy_df.empty
        ):
            return pd.DataFrame()

        ranked = (
            policy_df.loc[
                policy_df[
                    "is_policy"
                ]
            ]
            .sort_values(
                "selection_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        if len(ranked) < minimum_items:
            relaxed = (
                policy_df.loc[
                    (
                        policy_df[
                            "complete_clause"
                        ]
                    )
                    & (
                        policy_df[
                            "policy_entailment"
                        ]
                        >= 0.30
                    )
                    & (
                        policy_df[
                            "non_policy_entailment"
                        ]
                        < 0.62
                    )
                ]
                .sort_values(
                    "selection_score",
                    ascending=False
                )
            )

            ranked = (
                pd.concat(
                    [
                        ranked,
                        relaxed
                    ],
                    ignore_index=True
                )
                .drop_duplicates(
                    subset=[
                        "source_sentence"
                    ]
                )
                .sort_values(
                    "selection_score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

        selected_rows = []
        selected_sentences = []
        function_counts = Counter()

        for pass_name in [
            "new_function",
            "fill"
        ]:
            for row in (
                ranked.itertuples(
                    index=False
                )
            ):
                sentence = (
                    row.source_sentence
                )

                if any(
                    self.sentence_jaccard(
                        sentence,
                        existing
                    )
                    >= 0.64
                    for existing in (
                        selected_sentences
                    )
                ):
                    continue

                if (
                    pass_name
                    == "new_function"
                    and function_counts[
                        row.policy_function
                    ] > 0
                ):
                    continue

                selected_rows.append(
                    row._asdict()
                )

                selected_sentences.append(
                    sentence
                )

                function_counts[
                    row.policy_function
                ] += 1

                if (
                    len(
                        selected_rows
                    )
                    >= maximum_items
                ):
                    break

            if (
                len(
                    selected_rows
                )
                >= maximum_items
            ):
                break

        return pd.DataFrame(
            selected_rows
        )

    def _source_fallback_rows(
        self,
        bill_text,
        count=3
    ):
        sentences = [
            self.ensure_sentence_ending(
                sentence
            )
            for sentence in (
                self.split_sentences(
                    bill_text
                )
            )
            if (
                8
                <= self.count_words(
                    sentence
                )
                <= 100
            )
            and self._has_complete_clause(
                sentence
            )
        ]

        if not sentences:
            sentences = [
                self.ensure_sentence_ending(
                    sentence
                )
                for sentence in (
                    self.split_sentences(
                        bill_text
                    )[:count]
                )
            ]

        if len(sentences) <= count:
            chosen = sentences

        else:
            indices = np.linspace(
                0,
                len(sentences) - 1,
                num=count,
                dtype=int
            )

            chosen = [
                sentences[
                    int(
                        index_value
                    )
                ]
                for index_value
                in indices
            ]

        return pd.DataFrame(
            [
                {
                    "sentence_position": (
                        position
                    ),
                    "source_sentence": (
                        sentence
                    ),
                    "policy_function": (
                        "Source provision"
                    ),
                    "policy_entailment": 1.0,
                    "policy_contradiction": 0.0,
                    "non_policy_function": "",
                    "non_policy_entailment": 0.0,
                    "semantic_score": 0.0,
                    "draft_similarity": 0.0,
                    "title_similarity": 0.0,
                    "complete_clause": True,
                    "selection_score": 0.0,
                    "is_policy": True
                }
                for position, sentence in enumerate(
                    chosen
                )
            ]
        )

    def _build_grounded_summary(
        self,
        bill_text,
        selected_policy_df
    ):
        if (
            selected_policy_df is None
            or selected_policy_df.empty
        ):
            selected_policy_df = (
                self._source_fallback_rows(
                    bill_text,
                    count=3
                )
            )

        if (
            len(
                selected_policy_df
            )
            < self.minimum_summary_claims
        ):
            fallback_df = (
                self._source_fallback_rows(
                    bill_text,
                    count=(
                        self.minimum_summary_claims
                    )
                )
            )

            selected_policy_df = (
                pd.concat(
                    [
                        selected_policy_df,
                        fallback_df
                    ],
                    ignore_index=True
                )
                .drop_duplicates(
                    subset=[
                        "source_sentence"
                    ]
                )
                .head(
                    self.maximum_summary_claims
                )
            )

        selected_policy_df = (
            selected_policy_df.head(
                self.maximum_summary_claims
            )
        )

        summary_claims = [
            self.ensure_sentence_ending(
                sentence
            )
            for sentence in (
                selected_policy_df[
                    "source_sentence"
                ].tolist()
            )
        ]

        return (
            " ".join(
                summary_claims
            ),
            selected_policy_df
        )

    def _select_policy_changes(
        self,
        policy_df,
        summary_policy_df
    ):
        rows = []

        if (
            summary_policy_df is not None
            and not summary_policy_df.empty
        ):
            rows.extend(
                summary_policy_df.to_dict(
                    orient="records"
                )
            )

        if (
            policy_df is not None
            and not policy_df.empty
        ):
            rows.extend(
                policy_df.loc[
                    policy_df[
                        "is_policy"
                    ]
                ]
                .sort_values(
                    "selection_score",
                    ascending=False
                )
                .to_dict(
                    orient="records"
                )
            )

        selected = []
        selected_sentences = []
        functions = Counter()

        for row in rows:
            sentence = (
                self.ensure_sentence_ending(
                    row[
                        "source_sentence"
                    ]
                )
            )

            if any(
                self.sentence_jaccard(
                    sentence,
                    existing
                )
                >= 0.64
                for existing
                in selected_sentences
            ):
                continue

            policy_function = str(
                row.get(
                    "policy_function",
                    "Policy change"
                )
            )

            # Prefer function diversity first, but
            # allow a second provision from the same
            # function when it adds distinct content.
            if (
                functions[
                    policy_function
                ] >= 2
            ):
                continue

            selected.append(
                {
                    "policy_change": (
                        sentence
                    ),
                    "policy_function": (
                        policy_function
                    ),
                    "source_sentence": (
                        sentence
                    ),
                    "support_score": float(
                        row.get(
                            "policy_entailment",
                            1.0
                        )
                    )
                }
            )

            selected_sentences.append(
                sentence
            )

            functions[
                policy_function
            ] += 1

            if (
                len(selected)
                >= self.maximum_policy_changes
            ):
                break

        return pd.DataFrame(
            selected
        )

    def _collect_stakeholder_candidates(
        self,
        policy_change_df
    ):
        if (
            policy_change_df is None
            or policy_change_df.empty
        ):
            return []

        candidate_rows = []

        allowed_entity_labels = {
            "PERSON",
            "ORG",
            "GPE",
            "NORP",
            "FAC"
        }

        allowed_dependencies = {
            "nsubj",
            "nsubjpass",
            "dobj",
            "obj",
            "pobj",
            "agent",
            "attr",
            "oprd",
            "dative"
        }

        for context_rank, row in enumerate(
            policy_change_df.head(
                10
            ).itertuples(
                index=False
            )
        ):
            context = str(
                row.source_sentence
            )

            document = (
                self.language_processor(
                    context
                )
            )

            phrases = []

            for entity in document.ents:
                if (
                    entity.label_
                    in allowed_entity_labels
                ):
                    phrases.append(
                        entity.text
                    )

            for noun_chunk in (
                document.noun_chunks
            ):
                root = noun_chunk.root

                if (
                    root.dep_
                    in allowed_dependencies
                    and root.pos_
                    in {
                        "NOUN",
                        "PROPN",
                        "PRON"
                    }
                ):
                    phrases.append(
                        noun_chunk.text
                    )

            for phrase in phrases:
                phrase = (
                    self.normalize_text(
                        phrase
                    )
                )

                phrase = re.sub(
                    r"^(?:the|a|an)\s+",
                    "",
                    phrase,
                    flags=re.IGNORECASE
                ).strip(
                    " ,.;:-"
                )

                if (
                    not phrase
                    or not (
                        1
                        <= self.count_words(
                            phrase
                        )
                        <= 9
                    )
                ):
                    continue

                if (
                    phrase.casefold()
                    in {
                        "it",
                        "they",
                        "he",
                        "she",
                        "this",
                        "that",
                        "this act",
                        "this section",
                        "the act",
                        "the section",
                        "law",
                        "provision"
                    }
                ):
                    continue

                candidate_rows.append(
                    {
                        "candidate": (
                            phrase
                        ),
                        "context": (
                            context
                        ),
                        "context_rank": (
                            context_rank
                        )
                    }
                )

        unique_rows = []

        seen_pairs = set()

        for row in candidate_rows:
            key = (
                row[
                    "candidate"
                ].casefold(),
                row[
                    "context"
                ].casefold()
            )

            if key in seen_pairs:
                continue

            seen_pairs.add(
                key
            )

            unique_rows.append(
                row
            )

        return unique_rows

    def _classify_affected_groups(
        self,
        policy_change_df
    ):
        candidates = (
            self._collect_stakeholder_candidates(
                policy_change_df
            )
        )

        if not candidates:
            return pd.DataFrame()

        # Keep the number of NLI role checks bounded
        # while preserving candidates from different
        # policy contexts.
        candidates = candidates[:36]

        type_premises = []
        type_hypotheses = []
        type_metadata = []

        for row in candidates:
            candidate = row[
                "candidate"
            ]

            context = row[
                "context"
            ]

            stakeholder_hypothesis = (
                f'"{candidate}" refers to a person, '
                "group of people, organization, "
                "business, employer, agency, government "
                "body, institution, or other actor that "
                "is directly affected by or acts under "
                "this legal provision."
            )

            abstract_hypothesis = (
                f'"{candidate}" is an abstract concept, '
                "condition, procedure, legal term, "
                "activity, result, or object rather than "
                "a person, group, organization, business, "
                "agency, institution, or government body."
            )

            type_premises.extend(
                [
                    context,
                    context
                ]
            )

            type_hypotheses.extend(
                [
                    stakeholder_hypothesis,
                    abstract_hypothesis
                ]
            )

            type_metadata.append(
                row
            )

        type_probabilities = (
            self._score_nli_pairs(
                type_premises,
                type_hypotheses,
                batch_size=16
            )
        )

        accepted_candidates = []

        for candidate_index, row in enumerate(
            type_metadata
        ):
            stakeholder_probs = (
                type_probabilities[
                    candidate_index * 2
                ]
            )

            abstract_probs = (
                type_probabilities[
                    candidate_index * 2
                    + 1
                ]
            )

            stakeholder_score = float(
                stakeholder_probs[
                    self.entailment_index
                ]
            )

            abstract_score = float(
                abstract_probs[
                    self.entailment_index
                ]
            )

            if (
                stakeholder_score
                >= 0.40
                and stakeholder_score
                >= abstract_score
                + 0.03
            ):
                accepted_candidates.append(
                    {
                        **row,
                        "stakeholder_score": (
                            stakeholder_score
                        ),
                        "abstract_score": (
                            abstract_score
                        )
                    }
                )

        if not accepted_candidates:
            return pd.DataFrame()

        role_names = list(
            STAKEHOLDER_ROLES.keys()
        )

        role_descriptions = [
            STAKEHOLDER_ROLES[
                role_name
            ]
            for role_name
            in role_names
        ]

        role_premises = []
        role_hypotheses = []
        role_metadata = []

        for candidate_row in (
            accepted_candidates
        ):
            candidate = (
                candidate_row[
                    "candidate"
                ]
            )

            context = (
                candidate_row[
                    "context"
                ]
            )

            for role_index, role_description in enumerate(
                role_descriptions
            ):
                role_premises.append(
                    context
                )

                role_hypotheses.append(
                    (
                        f'For "{candidate}": '
                        f"{role_description}"
                    )
                )

                role_metadata.append(
                    (
                        candidate_row,
                        role_index
                    )
                )

        role_probabilities = (
            self._score_nli_pairs(
                role_premises,
                role_hypotheses,
                batch_size=16
            )
        )

        best_role_by_candidate = {}

        for (
            candidate_row,
            role_index
        ), probabilities in zip(
            role_metadata,
            role_probabilities
        ):
            candidate_key = (
                candidate_row[
                    "candidate"
                ].casefold()
            )

            entailment = float(
                probabilities[
                    self.entailment_index
                ]
            )

            current = (
                best_role_by_candidate
                .get(
                    candidate_key
                )
            )

            if (
                current is None
                or entailment
                > current[
                    "role_score"
                ]
            ):
                best_role_by_candidate[
                    candidate_key
                ] = {
                    **candidate_row,
                    "role": (
                        role_names[
                            role_index
                        ]
                    ),
                    "role_score": (
                        entailment
                    )
                }

        ranked = [
            row
            for row in (
                best_role_by_candidate
                .values()
            )
            if row[
                "role_score"
            ] >= 0.42
        ]

        ranked.sort(
            key=lambda row: (
                -row[
                    "role_score"
                ],
                -row[
                    "stakeholder_score"
                ],
                row[
                    "context_rank"
                ],
                -len(
                    row[
                        "candidate"
                    ]
                )
            )
        )

        selected = []

        for row in ranked:
            candidate = row[
                "candidate"
            ]

            candidate_key = re.sub(
                r"[^a-z0-9]+",
                " ",
                candidate.casefold()
            ).strip()

            duplicate = False

            for existing in selected:
                existing_key = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    existing[
                        "affected_group"
                    ].casefold()
                ).strip()

                if (
                    candidate_key
                    == existing_key
                    or (
                        candidate_key
                        and existing_key
                        and (
                            candidate_key
                            in existing_key
                            or existing_key
                            in candidate_key
                        )
                    )
                ):
                    duplicate = True

                    if (
                        len(candidate_key)
                        > len(
                            existing_key
                        )
                        and row[
                            "role_score"
                        ]
                        >= existing[
                            "role_score"
                        ]
                        - 0.05
                    ):
                        existing[
                            "affected_group"
                        ] = candidate

                        existing[
                            "role"
                        ] = row[
                            "role"
                        ]

                        existing[
                            "role_score"
                        ] = row[
                            "role_score"
                        ]

                        existing[
                            "evidence"
                        ] = row[
                            "context"
                        ]

                    break

            if duplicate:
                continue

            selected.append(
                {
                    "affected_group": (
                        candidate
                    ),
                    "role": (
                        row[
                            "role"
                        ]
                    ),
                    "role_score": round(
                        float(
                            row[
                                "role_score"
                            ]
                        ),
                        4
                    ),
                    "stakeholder_score": round(
                        float(
                            row[
                                "stakeholder_score"
                            ]
                        ),
                        4
                    ),
                    "evidence": (
                        row[
                            "context"
                        ]
                    )
                }
            )

            if (
                len(selected)
                >= self.maximum_affected_groups
            ):
                break

        return pd.DataFrame(
            selected
        )

    def _generate_plain_rewrites(
        self,
        source_sentences
    ):
        if not source_sentences:
            return []

        prompts = [
            (
                "Rewrite this U.S. legislative provision "
                "in plain English for a general reader. "
                "Do not change who must, may, or must not "
                "do something. Keep every number, date, "
                "deadline, exception, condition, penalty, "
                "right, and named actor that affects the "
                "meaning. Do not add information. "
                "Use one clear sentence.\n\n"
                f"Provision: {sentence}"
            )
            for sentence
            in source_sentences
        ]

        model = (
            self._load_plain_model()
        )

        rewrites = []

        for batch_start in range(
            0,
            len(prompts),
            2
        ):
            batch_prompts = prompts[
                batch_start:
                batch_start + 2
            ]

            encoded = (
                self.plain_tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                generated_ids = (
                    model.generate(
                        **encoded,
                        max_new_tokens=112,
                        num_beams=4,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                )

            decoded = (
                self.plain_tokenizer
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            )

            rewrites.extend(
                [
                    self.ensure_sentence_ending(
                        rewrite
                    )
                    for rewrite
                    in decoded
                ]
            )

            del encoded
            del generated_ids

        self._release_model(
            "plain"
        )

        return rewrites

    def _create_plain_english_explanation(
        self,
        summary_policy_df
    ):
        if (
            summary_policy_df is None
            or summary_policy_df.empty
        ):
            return {
                "text": "",
                "details": []
            }

        source_sentences = [
            self.ensure_sentence_ending(
                sentence
            )
            for sentence in (
                summary_policy_df[
                    "source_sentence"
                ].tolist()
            )
        ]

        # Release the NLI model before loading the
        # plain-English model to lower peak memory on
        # Streamlit Community Cloud.
        self._release_model(
            "nli"
        )

        rewrites = (
            self._generate_plain_rewrites(
                source_sentences
            )
        )

        details = []
        final_sentences = []

        for source_sentence, rewrite in zip(
            source_sentences,
            rewrites
        ):
            verification = (
                self._verify_rewrite(
                    source_sentence=(
                        source_sentence
                    ),
                    rewritten_sentence=(
                        rewrite
                    )
                )
            )

            if verification[
                "accepted"
            ]:
                final_sentence = rewrite
                fallback_used = False

            else:
                final_sentence = (
                    source_sentence
                )

                fallback_used = True

            final_sentences.append(
                final_sentence
            )

            details.append(
                {
                    "source_sentence": (
                        source_sentence
                    ),
                    "rewrite": rewrite,
                    "final_sentence": (
                        final_sentence
                    ),
                    "entailment": round(
                        verification[
                            "entailment"
                        ],
                        4
                    ),
                    "contradiction": round(
                        verification[
                            "contradiction"
                        ],
                        4
                    ),
                    "fallback_used": (
                        fallback_used
                    )
                }
            )

        return {
            "text": " ".join(
                final_sentences
            ),
            "details": details
        }

    def _filter_similar_bills(
        self,
        result_df,
        query_title,
        top_k
    ):
        if (
            result_df is None
            or result_df.empty
        ):
            return pd.DataFrame()

        query_title_key = (
            self.normalize_title(
                query_title
            )
        )

        filtered_rows = []

        for row in result_df.to_dict(
            orient="records"
        ):
            result_title_key = (
                self.normalize_title(
                    row.get(
                        "title",
                        ""
                    )
                )
            )

            exact_match = (
                query_title_key
                and result_title_key
                and query_title_key
                == result_title_key
            )

            near_match = (
                query_title_key
                and result_title_key
                and SequenceMatcher(
                    None,
                    query_title_key,
                    result_title_key
                ).ratio()
                >= 0.94
            )

            if (
                exact_match
                or near_match
            ):
                continue

            filtered_rows.append(
                row
            )

            if (
                len(filtered_rows)
                >= top_k
            ):
                break

        if not filtered_rows:
            return pd.DataFrame()

        for rank, row in enumerate(
            filtered_rows,
            start=1
        ):
            row["rank"] = rank

        return pd.DataFrame(
            filtered_rows
        )

    @staticmethod
    def _similar_bills_to_records(
        similar_bills_df
    ):
        if (
            similar_bills_df is None
            or similar_bills_df.empty
        ):
            return []

        records = []

        for row in (
            similar_bills_df.to_dict(
                orient="records"
            )
        ):
            records.append(
                {
                    "rank": int(
                        row.get(
                            "rank",
                            len(records) + 1
                        )
                    ),
                    "document_id": str(
                        row.get(
                            "document_id",
                            ""
                        )
                    ),
                    "title": str(
                        row.get(
                            "title",
                            ""
                        )
                    ),
                    "summary": str(
                        row.get(
                            "summary",
                            row.get(
                                "reference_summary",
                                ""
                            )
                        )
                    ),
                    "prepared_split": str(
                        row.get(
                            "prepared_split",
                            ""
                        )
                    ),
                    "similarity_score": round(
                        float(
                            row.get(
                                "similarity_score",
                                0.0
                            )
                        ),
                        4
                    )
                }
            )

        return records

    def analyze_bill(
        self,
        title,
        bill_text,
        top_k=5
    ):
        pipeline_start = (
            time.perf_counter()
        )

        title = self.normalize_text(
            title
        )

        bill_text = self.normalize_text(
            bill_text
        )

        if not bill_text:
            raise ValueError(
                "Bill text is required."
            )

        draft_result = (
            self._hierarchical_t5_draft(
                title=title,
                bill_text=bill_text
            )
        )

        evidence_store = (
            self._build_evidence_store(
                bill_text
            )
        )

        policy_df = (
            self._classify_policy_sentences(
                bill_text=bill_text,
                title=title,
                draft_summary=(
                    draft_result[
                        "draft_summary"
                    ]
                )
            )
        )

        summary_policy_df = (
            self._select_diverse_policy_rows(
                policy_df=policy_df,
                minimum_items=(
                    self.minimum_summary_claims
                ),
                maximum_items=(
                    self.maximum_summary_claims
                )
            )
        )

        grounded_summary, summary_policy_df = (
            self._build_grounded_summary(
                bill_text=bill_text,
                selected_policy_df=(
                    summary_policy_df
                )
            )
        )

        policy_change_df = (
            self._select_policy_changes(
                policy_df=policy_df,
                summary_policy_df=(
                    summary_policy_df
                )
            )
        )

        affected_group_df = (
            self._classify_affected_groups(
                policy_change_df
            )
        )

        explanation_result = (
            self._create_plain_english_explanation(
                summary_policy_df
            )
        )

        # NLI is not needed after explanation
        # verification. Release it before retrieval.
        self._release_model(
            "nli"
        )

        raw_similar_bills = (
            self.similar_bill_search
            .semantic_search(
                title=title,
                summary_or_text=(
                    grounded_summary
                ),
                top_k=(
                    top_k + 15
                )
            )
        )

        similar_bills_df = (
            self._filter_similar_bills(
                result_df=(
                    raw_similar_bills
                ),
                query_title=title,
                top_k=top_k
            )
        )

        pipeline_seconds = (
            time.perf_counter()
            - pipeline_start
        )

        policy_function_count = (
            int(
                policy_change_df[
                    "policy_function"
                ].nunique()
            )
            if (
                policy_change_df
                is not None
                and not policy_change_df.empty
            )
            else 0
        )

        return {
            "title": title,
            "summary": (
                grounded_summary
            ),
            "plain_english_explanation": (
                explanation_result[
                    "text"
                ]
            ),
            "key_policy_changes": (
                policy_change_df[
                    "policy_change"
                ].tolist()
                if (
                    policy_change_df
                    is not None
                    and not policy_change_df.empty
                )
                else []
            ),
            "affected_groups": (
                affected_group_df[
                    "affected_group"
                ].tolist()
                if (
                    affected_group_df
                    is not None
                    and not affected_group_df.empty
                )
                else []
            ),
            "similar_bills": (
                self._similar_bills_to_records(
                    similar_bills_df
                )
            ),
            "generation_metadata": {
                "pipeline_revision": (
                    "source_grounded_v3"
                ),
                "bill_chunks": int(
                    draft_result[
                        "chunk_count"
                    ]
                ),
                "reduction_levels": int(
                    draft_result[
                        "reduction_levels"
                    ]
                ),
                "grounded_summary_claims": int(
                    len(
                        summary_policy_df
                    )
                ),
                "policy_functions_covered": (
                    policy_function_count
                ),
                "affected_groups_found": int(
                    len(
                        affected_group_df
                    )
                    if (
                        affected_group_df
                        is not None
                    )
                    else 0
                ),
                "plain_english_fallbacks": int(
                    sum(
                        1
                        for detail in (
                            explanation_result[
                                "details"
                            ]
                        )
                        if detail[
                            "fallback_used"
                        ]
                    )
                ),
                "pipeline_seconds": round(
                    float(
                        pipeline_seconds
                    ),
                    4
                ),
                "device": (
                    self.device.type
                ),
                "generation_config": (
                    self.generation_config.get(
                        "name",
                        "Notebook 4 selected configuration"
                    )
                )
            },
            "grounding_details": {
                "draft_t5_summary": (
                    draft_result[
                        "draft_summary"
                    ]
                ),
                "summary_policy_provisions": (
                    summary_policy_df.to_dict(
                        orient="records"
                    )
                    if (
                        summary_policy_df
                        is not None
                        and not summary_policy_df.empty
                    )
                    else []
                ),
                "policy_change_details": (
                    policy_change_df.to_dict(
                        orient="records"
                    )
                    if (
                        policy_change_df
                        is not None
                        and not policy_change_df.empty
                    )
                    else []
                ),
                "affected_group_details": (
                    affected_group_df.to_dict(
                        orient="records"
                    )
                    if (
                        affected_group_df
                        is not None
                        and not affected_group_df.empty
                    )
                    else []
                ),
                "plain_english_details": (
                    explanation_result[
                        "details"
                    ]
                ),
                "evidence_sentence_count": int(
                    len(
                        evidence_store[
                            "sentences"
                        ]
                    )
                )
            },
            "disclaimer": (
                DISCLAIMER_TEXT
            )
        }
