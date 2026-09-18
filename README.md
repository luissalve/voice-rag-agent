# voice-rag-agent — ask a PDF by phone or web

**Ask a PDF by phone or web, with grounded answers and page citations.**

## Why this exists

Most public RAG demos stop at a chat box. Most public voice-agent demos stop at "the AI can talk."
This scaffold wires both halves together over one shared retrieval core, because that is
the shape production voice-support systems actually take: one knowledge base, multiple front doors.

## How an inbound call flows

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant Twilio as Twilio Number
    participant Webhook as FastAPI voice webhook
    participant ElevenLabs as ElevenLabs Conv. AI
    participant API as FastAPI /ask
    participant RAG as src/rag.py
    participant PG as Postgres + pgvector
    participant Claude as Anthropic Claude

    Note over Twilio,Webhook: No such route exists in src/app.py yet — planned.
    Caller->>Twilio: Dials phone number
    Twilio->>Webhook: Audio stream (planned)
    Webhook->>ElevenLabs: Audio handoff (planned)

    Note over ElevenLabs,API: /ask is a typed handler in src/app.py;<br/>its body calls into src/rag.py.
    ElevenLabs->>API: POST /ask {question, document_id} (planned)
    API->>RAG: embed_query(question) — NotImplementedError in v1
    RAG->>PG: ORDER BY embedding <=> $1 LIMIT k (planned)
    PG-->>RAG: top-k chunks (planned)
    RAG->>Claude: messages.create(context, question) (planned)
    Claude-->>RAG: grounded answer + page numbers (planned)
    RAG-->>API: Answer{text, citations}

    API-->>ElevenLabs: {answer, citations} (planned)
    ElevenLabs->>Caller: Speaks answer (planned)
```

The web front door (`POST /ask` directly) shares the same retrieval + answering code — `src/rag.py`
is the single place that decides what "grounded" means.

## Quickstart

```bash
git clone https://github.com/luissalve/voice-rag-agent
cd voice-rag-agent
cp .env.example .env  # fill in the keys you actually need; the rest have defaults
pip install -e ".[dev]"
uvicorn src.app:app --reload
```

The test path runs offline — no API keys required:

```bash
pytest
```

The full `/ask` and `/ingest` endpoints need the keys listed in `.env.example`; today they return
`501 Not Implemented` because the embedding, retrieval, and Claude-calling code is a typed skeleton
— see [Status & roadmap](#status--roadmap) below.

## What to look at

- `chunk_text` in `src/rag.py` — pure, dependency-free, the one function in this scaffold with real behavior; fully unit-tested.
- `ask` in `src/app.py` — the single shared entry point for both front doors; today it returns a clear `501` rather than fabricating an answer.
- `retrieve_top_k` in `src/rag.py` — the planned seam between pgvector and `answer_with_citations`; its typed signature tells you the contract the rest of the app relies on.

## Design decisions

- **Chunking strategy** — `chunk_text` runs per page and every `Chunk` carries its source `page_number`, so page citations come back without post-processing.
- **Embedding choice** — BGE-M3 is the default (via `EMBEDDING_MODEL` in `.env.example`); picked for multilingual + code-aware coverage.
- **Grounding enforcement** — the planned `answer_with_citations` builds a prompt from the retrieved chunks (text + page number) and instructs Claude to answer only from that context and to say "I don't know" when the context is insufficient.
- **Phone vs web** — both front doors hit the same `/ask`; the phone path wraps it with Twilio (telephony) and ElevenLabs (STT/TTS) — planned in v1.

## Status & roadmap

This is a **v1 scaffold**. The retrieval core, the embedding client, the pgvector storage, the Twilio
webhook, the ElevenLabs integration, and the Claude call are typed skeletons with explicit TODOs —
they return `501 Not Implemented` rather than fabricating fake behavior. The proposed next commits are
tracked in [`CHANGELOG-PROPOSAL.md`](CHANGELOG-PROPOSAL.md):

1. `test(rag): add ranking tests with a deterministic fake embedder`
2. `feat(security): verify Twilio webhook signatures`
3. `ci: add GitHub Actions workflow running ruff + pytest on Python 3.12`
4. `feat(eval): add a tiny offline evaluation script over a fixture set`

## License

MIT — see [LICENSE](LICENSE).
