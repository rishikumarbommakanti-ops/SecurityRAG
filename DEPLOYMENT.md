# SecurityRAG Deployment Guide

## Streamlit Cloud Deployment

1. Create a Streamlit Cloud app and connect it to this GitHub repository.
2. Set the main file to `app/app.py`.
3. Use Python version `3.11`.
4. Add the following repository files:
   - `runtime.txt`
   - `requirements.txt`

## Required Secrets

In Streamlit Cloud, configure the following secrets:

- `GOOGLE_API_KEY`: Your Google Gemini API key.
- `GEMINI_MODEL`: Optional Gemini model name (default: `gemini-1.5-flash-002`).
- `ENABLE_GEMINI_FALLBACK`: `true` or `false` to allow fallback messaging when Gemini fails.
- `DEMO_MODE`: `true` to enable demo mode and disable live updates.
- `EMBEDDING_MODEL_NAME`: Optional embedding model name.
- `CHROMA_COLLECTION_NAME`: Optional collection name.
- `RETRIEVER_TOP_K`: Optional retrieval top-k.
- `ENABLE_LIVE_UPDATES`: Optional; demo mode forces this to `false`.

## Environment and Secrets Priority

`config.py` reads configuration in this order:
1. `st.secrets`
2. environment variables

This means Streamlit secrets take precedence over OS environment variables.

## Demo Mode

Set `DEMO_MODE=true` to:
- disable live updates
- skip Gemini answer generation
- load a sample knowledge set for local testing
- avoid failures if the Gemini key is missing

## Health Dashboard

The app now displays deployment health information in the sidebar:
- Demo mode status
- Gemini API key presence
- ChromaDB readiness
- Knowledge cache readiness
- Last validation timestamp

## Troubleshooting

### Gemini API key missing
- Ensure `GOOGLE_API_KEY` is set in Streamlit secrets.
- If not using Gemini, set `DEMO_MODE=true` for demo testing.

### Chroma database initialization fails
- Verify write access to `chroma_db` in the app root.
- If using Streamlit Cloud, ensure persistence is enabled.

### Knowledge cache unavailable
- Verify `data/` directory exists and is writable.
- The app will attempt to create `data/knowledge_cache.json` automatically.

### Test failures
Run locally:
```bash
cd /workspaces/SecurityRAG
PYTHONPATH=. python -m pytest -q
```

## Notes

- If you are using a newer Gemini SDK, update `app/llm/gemini_client.py` accordingly.
- `requirements.txt` is curated for deployment and tested with Python 3.11.
