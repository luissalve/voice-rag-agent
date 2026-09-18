# CHANGELOG — proposed next commits

These are the next four commits proposed for `voice-rag-agent` after the v1 scaffold landed.
Each is sized small enough to review in one sitting and has an acceptance check the maintainer
can run offline (no paid API keys, no Twilio account, no Anthropic account). Order is the
suggested order; commits 1–4 do not depend on each other and can land in any order.

---

## 1. `test(rag): add ranking tests with a deterministic fake embedder`

**Why now:** credibility. A scaffold with one passing test reads thin; the retrieval ranking core
is the easiest place to add honest coverage because it is pure math (no network, no DB, no API keys).

**Files touched:**
- `src/rag.py` — add `cosine_similarity(a, b) -> float` and `rank_by_similarity(query_embedding, candidates) -> list[EmbeddedChunk]` (both pure, no I/O). `chunk_text` and the existing `NotImplementedError` stubs are unchanged.
- `tests/conftest.py` — new, exposes a `FakeEmbedder` class with a small fixed vocabulary mapped to orthogonal 4-D vectors, plus `make_chunk` / `make_embedded` factories and `sample_chunks` / `fake_embedder` / `make_embedded` pytest fixtures.
- `tests/test_rag.py` — add tests for similarity edge cases (identical, orthogonal, opposite, zero-vector on either side, length mismatch) and for ranking order, stability, empty input, and iterator input. Existing `chunk_text` tests are preserved unchanged.

**Acceptance check (offline):**
```bash
pip install -e ".[dev]"
pytest -v
```
All tests pass. No API keys, no network, no database required.

**Estimated size:** ~190 lines added (src/rag.py +60, conftest.py +75, test_rag.py ~+90 net additions).

---

## 2. `feat(security): verify Twilio webhook signatures`

**Why now:** once a Twilio number is pointed at this service, every public webhook is a potential
spoofing surface. Verifying `X-Twilio-Signature` before doing anything else is the smallest change
that makes the phone path production-shaped.

**Files touched:**
- `src/security.py` — new, exposes `verify_twilio_signature(auth_token: str, full_url: str, form_params: dict[str, str], signature_header: str | None) -> bool` using `hmac.compare_digest` (constant-time comparison, not `==`).
- `src/app.py` — add `POST /twilio/webhook` that calls the helper and returns `200` on success, `403` on failure; the existing `/ask`, `/ingest`, and `/health` routes are unchanged.
- `tests/test_security.py` — new, tests a valid signature, a tampered body, a missing header, a wrong auth token, and a sanity check that `hmac.compare_digest` is used (not `==`).

**Acceptance check (offline):**
```bash
pytest tests/test_security.py -v
```
All tests pass. No Twilio account, no network required.

**Estimated size:** ~130 lines added (security.py ~40, app.py ~20, test_security.py ~70).

---

## 3. `ci: add GitHub Actions workflow running ruff + pytest on Python 3.12`

**Why now:** credibility on the green badge. Recruiters look at the CI badge; the README already
declares `python 3.12+`, so the matrix should match.

**Files touched:**
- `.github/workflows/ci.yml` — new. Triggers on `push` and `pull_request` to `main`; single job, `runs-on: ubuntu-latest`, `python-version: "3.12"`. Steps: `actions/checkout@v4`, `actions/setup-python@v5` with pip cache, `pip install -e ".[dev]"`, `ruff check src tests`, `ruff format --check src tests`, `pytest -v`. No secrets required.

**Acceptance check (offline):**
```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
pytest -v
```
All commands exit 0. The workflow file is plain YAML with no secrets required.

**Estimated size:** ~35 lines added (one new file).

---

## 4. `feat(eval): add a tiny offline evaluation script over a fixture set`

**Why now:** credibility on the "grounded" claim. A fixture set of question/expected-pages lets
the maintainer detect regressions in retrieval quality even before the live Claude call is wired.

**Files touched:**
- `scripts/eval.py` — new. Loads `tests/fixtures/eval_cases.json`, runs each question through a stubbed pipeline (the `FakeEmbedder` from `tests/conftest.py` for embedding + a hand-coded mock retriever that returns the expected chunks + a hand-coded mock Claude that returns the expected answer), and prints a per-case score plus an aggregate summary (`recall@5`, keyword hit rate, pass/fail count).
- `tests/fixtures/eval_cases.json` — new. Five fixture cases: each has `question`, `expected_pages`, `expected_keywords`, and `expected_chunks`. No PII, no real document content — synthetic content only.

**Acceptance check (offline):**
```bash
python scripts/eval.py
```
Exits 0, prints a summary like `eval: 5/5 passed (recall@5 = 1.0)`. Uses no network and no API keys — the retriever and Claude answer are stubbed in `scripts/eval.py` itself.

**Estimated size:** ~150 lines added (eval.py ~110, eval_cases.json ~20, README note ~10).
