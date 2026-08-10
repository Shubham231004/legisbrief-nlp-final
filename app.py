# Streamlit interface for LegisBrief-NLP.

from pathlib import Path
import json
import shutil

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import snapshot_download

from app_helpers import validate_bill_input
from legisbrief_pipeline import LegisBriefPipeline


# This must be the first Streamlit command.
st.set_page_config(
    page_title="LegisBrief-NLP",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)


CURRENT_APP_DIRECTORY = Path(
    __file__
).resolve().parent

APP_ROOT = (
    CURRENT_APP_DIRECTORY
    if (
        CURRENT_APP_DIRECTORY
        / "project_config.json"
    ).exists()
    else CURRENT_APP_DIRECTORY.parent
)


HF_ARTIFACT_REPOSITORY = (
    "HaalandBuu/"
    "legisbrief-grounded-artifacts"
)


# The pipeline expects a complete project structure.
# The large model files will be downloaded here.
RUNTIME_PROJECT_ROOT = Path(
    "/tmp/LegisBrief_Project"
)


@st.cache_resource(
    show_spinner=False
)
def prepare_runtime_project():
    """
    Download the Hugging Face repository and rebuild
    the folder structure expected by the final
    generalized LegisBriefPipeline.
    """

    download_root = Path(
        "/tmp/LegisBrief_HF_Download"
    )

    for folder_path in [
        RUNTIME_PROJECT_ROOT,
        download_root
    ]:
        if folder_path.exists():
            shutil.rmtree(
                folder_path
            )

        folder_path.mkdir(
            parents=True,
            exist_ok=True
        )

    # Small configuration files remain in GitHub.
    local_project_files = [
        Path(
            "project_config.json"
        ),
        Path(
            "structured_output_config.json"
        ),
        (
            Path("outputs")
            / "metrics"
            / "t5_selected_generation_config.json"
        )
    ]

    for relative_path in (
        local_project_files
    ):
        source_path = (
            APP_ROOT
            / relative_path
        )

        destination_path = (
            RUNTIME_PROJECT_ROOT
            / relative_path
        )

        if not source_path.exists():
            raise FileNotFoundError(
                "Required GitHub file was not found:\n"
                f"{source_path}"
            )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    downloaded_repository = Path(
        snapshot_download(
            repo_id=(
                HF_ARTIFACT_REPOSITORY
            ),
            repo_type="model",
            local_dir=str(
                download_root
            )
        )
    )

    def model_directory_is_valid(
        candidate_directory,
        expected_model_type=None
    ):
        config_path = (
            candidate_directory
            / "config.json"
        )

        if not config_path.exists():
            return False

        try:
            with config_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                model_config = (
                    json.load(file)
                )

        except Exception:
            return False

        if expected_model_type:
            model_type = str(
                model_config.get(
                    "model_type",
                    ""
                )
            ).lower()

            if (
                model_type
                != expected_model_type
            ):
                return False

        tokenizer_exists = any(
            file_path.exists()
            for file_path in [
                (
                    candidate_directory
                    / "tokenizer_config.json"
                ),
                (
                    candidate_directory
                    / "tokenizer.json"
                ),
                (
                    candidate_directory
                    / "spiece.model"
                ),
                (
                    candidate_directory
                    / "vocab.json"
                )
            ]
        )

        weights_exist = any(
            file_path.exists()
            for file_path in [
                (
                    candidate_directory
                    / "model.safetensors"
                ),
                (
                    candidate_directory
                    / "pytorch_model.bin"
                )
            ]
        )

        return bool(
            tokenizer_exists
            and weights_exist
        )

    def find_directory_named(
        directory_name,
        marker_name
    ):
        candidates = []

        for marker_path in (
            downloaded_repository.rglob(
                marker_name
            )
        ):
            candidate_directory = (
                marker_path.parent
            )

            if (
                directory_name
                in candidate_directory.parts
            ):
                candidates.append(
                    candidate_directory
                )

        if not candidates:
            return None

        return sorted(
            candidates,
            key=lambda path: (
                len(
                    path.parts
                ),
                str(path)
            )
        )[0]

    # Fine-tuned T5 summarizer.
    t5_source_directory = None

    for config_candidate in (
        downloaded_repository.rglob(
            "config.json"
        )
    ):
        candidate_directory = (
            config_candidate.parent
        )

        if not model_directory_is_valid(
            candidate_directory,
            expected_model_type="t5"
        ):
            continue

        # Prefer the project-trained model folder
        # rather than the separate FLAN-T5 model.
        if (
            "t5_small_legisbrief_final"
            in candidate_directory.parts
        ):
            t5_source_directory = (
                candidate_directory
            )
            break

    if t5_source_directory is None:
        raise FileNotFoundError(
            "The fine-tuned LegisBrief T5 model "
            "could not be located in the "
            "Hugging Face repository."
        )

    t5_destination_directory = (
        RUNTIME_PROJECT_ROOT
        / "models"
        / "t5_small_legisbrief_final"
    )

    shutil.copytree(
        t5_source_directory,
        t5_destination_directory,
        dirs_exist_ok=True
    )

    # Similar-bill retrieval artifacts.
    faiss_candidates = list(
        downloaded_repository.rglob(
            "faiss_index.bin"
        )
    )

    if not faiss_candidates:
        raise FileNotFoundError(
            "faiss_index.bin was not found in "
            "the Hugging Face repository."
        )

    similarity_source_directory = (
        faiss_candidates[0].parent
    )

    similarity_destination_directory = (
        RUNTIME_PROJECT_ROOT
        / "artifacts"
        / "similarity_search"
    )

    shutil.copytree(
        similarity_source_directory,
        similarity_destination_directory,
        dirs_exist_ok=True
    )

    modules_candidates = list(
        downloaded_repository.rglob(
            "modules.json"
        )
    )

    if not modules_candidates:
        raise FileNotFoundError(
            "The SentenceTransformer modules.json "
            "file was not found."
        )

    modules_candidates = sorted(
        modules_candidates,
        key=lambda path: (
            "sentence_transformer_model"
            not in path.parts,
            len(
                path.parts
            )
        )
    )

    sentence_transformer_source = (
        modules_candidates[0].parent
    )

    sentence_transformer_destination = (
        similarity_destination_directory
        / "sentence_transformer_model"
    )

    if not (
        sentence_transformer_destination
        / "modules.json"
    ).exists():
        shutil.copytree(
            sentence_transformer_source,
            sentence_transformer_destination,
            dirs_exist_ok=True
        )

    # General grounded-inference artifacts produced
    # by final Notebook 8.
    grounded_destination = (
        RUNTIME_PROJECT_ROOT
        / "artifacts"
        / "grounded_inference"
    )

    grounded_destination.mkdir(
        parents=True,
        exist_ok=True
    )

    nli_source = find_directory_named(
        "nli_model",
        "config.json"
    )

    plain_source = (
        find_directory_named(
            "plain_english_model",
            "config.json"
        )
    )

    spacy_source = (
        find_directory_named(
            "spacy_en_core_web_sm",
            "meta.json"
        )
    )

    if nli_source is None:
        raise FileNotFoundError(
            "The Notebook 8 NLI model folder "
            "was not found in the Hugging Face "
            "repository."
        )

    if plain_source is None:
        raise FileNotFoundError(
            "The Notebook 8 plain-English model "
            "folder was not found in the "
            "Hugging Face repository."
        )

    if spacy_source is None:
        raise FileNotFoundError(
            "The Notebook 8 spaCy model folder "
            "was not found in the Hugging Face "
            "repository."
        )

    shutil.copytree(
        nli_source,
        grounded_destination
        / "nli_model",
        dirs_exist_ok=True
    )

    shutil.copytree(
        plain_source,
        grounded_destination
        / "plain_english_model",
        dirs_exist_ok=True
    )

    shutil.copytree(
        spacy_source,
        grounded_destination
        / "spacy_en_core_web_sm",
        dirs_exist_ok=True
    )

    grounding_config_candidates = [
        candidate
        for candidate in (
            downloaded_repository.rglob(
                "grounding_config.json"
            )
        )
        if (
            "grounded_inference"
            in candidate.parts
        )
    ]

    if grounding_config_candidates:
        shutil.copy2(
            grounding_config_candidates[0],
            grounded_destination
            / "grounding_config.json"
        )

    required_runtime_files = [
        t5_destination_directory
        / "config.json",
        similarity_destination_directory
        / "faiss_index.bin",
        similarity_destination_directory
        / "bill_metadata.csv",
        (
            similarity_destination_directory
            / "sentence_transformer_model"
            / "modules.json"
        ),
        grounded_destination
        / "nli_model"
        / "config.json",
        grounded_destination
        / "plain_english_model"
        / "config.json",
        grounded_destination
        / "spacy_en_core_web_sm"
        / "meta.json",
        (
            RUNTIME_PROJECT_ROOT
            / "outputs"
            / "metrics"
            / "t5_selected_generation_config.json"
        )
    ]

    missing_runtime_files = [
        str(
            file_path
        )
        for file_path in (
            required_runtime_files
        )
        if not file_path.exists()
    ]

    if missing_runtime_files:
        raise FileNotFoundError(
            "The final grounded runtime is "
            "missing required files:\n"
            + "\n".join(
                missing_runtime_files
            )
        )

    return RUNTIME_PROJECT_ROOT

@st.cache_resource(
    show_spinner=False
)
def load_pipeline():
    """
    Load the LegisBrief pipeline once per
    Streamlit application runtime.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    runtime_project_root = (
        prepare_runtime_project()
    )

    return LegisBriefPipeline(
        project_root=runtime_project_root,
        device=device
    )


def display_policy_list(
    items,
    empty_message
):
    if not items:
        st.info(empty_message)
        return

    for item in items:
        st.markdown(
            f"- {item}"
        )


def display_similar_bills(
    similar_bills
):
    if not similar_bills:
        st.info(
            "No similar bills were returned."
        )
        return

    for fallback_rank, bill in enumerate(
        similar_bills,
        start=1
    ):
        rank = bill.get(
            "rank",
            fallback_rank
        )

        title = bill.get(
            "title",
            "Untitled bill"
        )

        score = float(
            bill.get(
                "similarity_score",
                0.0
            )
        )

        with st.expander(
            f"{rank}. {title} — "
            f"similarity {score:.3f}"
        ):
            summary = bill.get(
                "summary",
                bill.get(
                    "reference_summary",
                    ""
                )
            )

            if summary:
                st.write(summary)

            details = {
                "Document ID": bill.get(
                    "document_id",
                    ""
                ),
                "Dataset split": bill.get(
                    "prepared_split",
                    ""
                ),
                "Similarity score": round(
                    score,
                    4
                )
            }

            st.dataframe(
                pd.DataFrame(
                    [details]
                ),
                hide_index=True,
                use_container_width=True
            )


def main():
    st.title(
        "LegisBrief-NLP"
    )

    st.subheader(
        "Legislative Bill Summarization and "
        "Similar-Bill Search"
    )

    st.write(
        "Paste a legislative bill below to receive "
        "a concise summary, a plain-English explanation, "
        "key policy changes, affected groups, and "
        "similar bills."
    )

    with st.sidebar:
        st.header("About")

        st.write(
            "LegisBrief-NLP uses a fine-tuned T5-small "
            "summarizer and SentenceTransformer plus "
            "FAISS semantic retrieval."
        )

        st.write(
            "The summary is model-generated. The "
            "structured fields use transparent "
            "rule-based post-processing."
        )

        st.warning(
            "This tool provides general information "
            "and does not provide legal advice."
        )

    title = st.text_input(
        "Bill title (optional)",
        placeholder=(
            "Example: Consumer Data Protection Act"
        )
    )

    bill_text = st.text_area(
        "Bill text",
        height=340,
        max_chars=500_000,
        placeholder=(
            "Paste the bill text here. "
            "At least 20 words are required."
        )
    )

    word_count = len(
        bill_text.split()
    )

    character_count = len(
        bill_text
    )

    count_column_1, count_column_2 = (
        st.columns(2)
    )

    count_column_1.caption(
        f"Words: {word_count:,}"
    )

    count_column_2.caption(
        f"Characters: {character_count:,}"
    )

    analyze_button = st.button(
        "Analyze Bill",
        type="primary",
        use_container_width=True
    )

    if not analyze_button:
        st.info(
            "Enter a bill and select Analyze Bill."
        )
        return

    validation = validate_bill_input(
        title=title,
        bill_text=bill_text
    )

    if not validation["is_valid"]:
        for error_message in validation[
            "errors"
        ]:
            st.error(error_message)

        return

    for warning_message in validation[
        "warnings"
    ]:
        st.warning(warning_message)

    try:
        with st.spinner(
            "Downloading the model files and "
            "analyzing the bill. The first run "
            "may take several minutes..."
        ):
            pipeline = load_pipeline()

            result = pipeline.analyze_bill(
                title=validation["title"],
                bill_text=validation[
                    "bill_text"
                ],
                top_k=5
            )

    except Exception as error:
        st.error(
            "The bill could not be analyzed. "
            "Check the model files and try again."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(
                str(error)
            )

        return

    st.success(
        "Bill analysis completed."
    )

    st.header(
        "Generated Summary"
    )

    st.write(
        result.get(
            "summary",
            ""
        )
    )

    st.header(
        "Plain-English Explanation"
    )

    st.write(
        result.get(
            "plain_english_explanation",
            ""
        )
    )

    left_column, right_column = (
        st.columns(2)
    )

    with left_column:
        st.header(
            "Key Policy Changes"
        )

        display_policy_list(
            result.get(
                "key_policy_changes",
                []
            ),
            (
                "No policy changes "
                "were extracted."
            )
        )

    with right_column:
        st.header(
            "Affected Groups"
        )

        display_policy_list(
            result.get(
                "affected_groups",
                []
            ),
            (
                "No affected groups "
                "were extracted."
            )
        )

    st.header(
        "Similar Bills"
    )

    display_similar_bills(
        result.get(
            "similar_bills",
            []
        )
    )

    generation_metadata = result.get(
        "generation_metadata",
        {}
    )

    if generation_metadata:
        with st.expander(
            "Generation Details"
        ):
            st.dataframe(
                pd.DataFrame(
                    [generation_metadata]
                ),
                hide_index=True,
                use_container_width=True
            )

    st.divider()

    st.caption(
        result.get(
            "disclaimer",
            (
                "This output is for informational "
                "purposes only and is not legal advice."
            )
        )
    )


if __name__ == "__main__":
    main()