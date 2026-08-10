# LegisBrief-NLP

LegisBrief-NLP accepts a legislative bill title and bill text and returns:

- A generated T5 summary
- A plain-English explanation
- Key policy changes
- Affected groups
- Five semantically similar bills

## Model components

- Fine-tuned `google-t5/t5-small` summarizer
- `sentence-transformers/all-mpnet-base-v2` embeddings
- FAISS exact cosine-similarity search
- Transparent rule-based structured-output extraction

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required structure

```text
LegisBrief_Deployment/
├── app.py
├── app_helpers.py
├── legisbrief_pipeline.py
├── structured_outputs.py
├── similar_bill_search.py
├── project_config.json
├── requirements.txt
├── runtime.txt
├── models/
│   └── t5_small_legisbrief_final/
├── artifacts/
│   └── similarity_search/
└── outputs/
    └── metrics/
        └── t5_selected_generation_config.json
```

## Limitations

The generated summary may omit details or produce inaccurate wording. BillSum directly supervises only the main summary. The plain-English explanation, policy changes, and affected groups are generated with transparent post-processing rules. Similarity scores indicate semantic closeness and do not establish legal equivalence.

## Disclaimer

This application provides information only and is not legal advice.
