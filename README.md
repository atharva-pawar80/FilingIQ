# FilingIQ

Grounded Q&A over SEC 10-K filings, built as a full-stack RAG application.

Ask questions like *"What are Apple's main risk factors?"* or *"How did
JPMorgan's net income change year over year?"* and get answers backed by
cited source passages — not hallucinated numbers.

Companion project to [RiskLens](#) (structured/tabular fraud detection ML).
FilingIQ applies the same risk-intelligence focus to unstructured
financial documents.

## Status

🚧 Under active development. Current phase: backend RAG pipeline.

- [x] Project scaffolding, CI, config
- [x] Ingestion: EDGAR fetch (optional) + section-aware HTML splitting
- [x] Chunking: narrative (recursive splitter) + tables (row-to-sentence)
- [ ] Embeddings + Chroma vector store wiring
- [ ] Hybrid retrieval (semantic + keyword)
- [ ] LLM generation via Groq + LangChain, grounded prompting
- [ ] Evaluation harness (RAGAS, 15-20 Q&A pairs)
- [ ] FastAPI endpoints (`/ingest`, `/ask`, `/history`)
- [ ] React frontend
- [ ] Deployment (Render + Vercel) + CI/CD

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | `sentence-transformers` (bge-large-en-v1.5), local, free |
| Vector DB | Chroma |
| Orchestration | LangChain |
| Backend | FastAPI + Docker |
| Frontend | React (Vite) |
| Eval | RAGAS |
| CI/CD | GitHub Actions |

## Local setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd filingiq/backend

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and add your real GROQ_API_KEY
# (get one free at https://console.groq.com/keys)

# 5. Add your downloaded 10-K filings
# Place the .htm files you downloaded from SEC EDGAR into:
#   backend/data/raw_filings/

# 6. Run the API locally
uvicorn app.main:app --reload
# visit http://localhost:8000/docs

# 7. Run tests
pytest tests/ -v
```

## Project structure

```
filingiq/
├── .github/workflows/ci.yml   # GitHub Actions: tests + eval suite
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── config.py          # env/config, loaded once
│   │   ├── ingestion/         # EDGAR fetch, section split, chunking
│   │   ├── routers/           # API endpoints (Phase 3)
│   │   └── services/          # retrieval, generation, eval logic
│   ├── data/raw_filings/      # downloaded 10-K HTML (gitignored)
│   ├── vectorstore/           # Chroma persistence (gitignored)
│   ├── eval/                  # eval dataset + RAGAS harness (Phase 2)
│   ├── tests/                 # pytest suite
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
└── frontend/                  # React app (Phase 4)
```

## Why these design choices

See project documentation / interview notes for the reasoning behind:
section-aware chunking, dual sentence/table representation for financial
tables, why Groq + local embeddings keep this fully free-tier, and the
evaluation methodology for measuring groundedness.
