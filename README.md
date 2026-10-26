# Adaptive Document Processing - AI Service (Plus Tier)

The Plus tier AI service of the Aura document platform. It reads PDF files and images,
sends them to Google Gemini 2.5 Flash for one combined analysis (text, type, fields,
tables, signatures, stamps, summary), stores the result in MongoDB, and makes the
documents searchable and answerable in natural language.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Status](#project-status)
3. [How It Works](#how-it-works)
4. [Processing Pipeline](#processing-pipeline)
5. [Components](#components)
6. [AI Models](#ai-models)
7. [Differences from Tier 1 (Basic)](#differences-from-tier-1-basic)
8. [Technology Stack](#technology-stack)
9. [Project Structure](#project-structure)
10. [Getting Started](#getting-started)
11. [Configuration](#configuration)
12. [API Reference](#api-reference)
13. [Security](#security)
14. [Testing](#testing)
15. [Known Issues](#known-issues)
16. [License](#license)

---

## Overview

### Purpose

Companies receive many paper and PDF documents: invoices, contracts, delivery notes,
letters, forms and more. Typing their key data into business systems and finding a
document again later costs a lot of manual work.

This service automates that work. It is a Python REST API (FastAPI) that:

- reads the text of scanned or digital documents, including handwriting,
- classifies each document into one of 36 canonical document types (including `other`),
- extracts structured fields (for example invoice number, dates, amounts, parties),
- extracts entities (companies, people, dates, amounts, addresses, e-mail addresses,
  phone numbers),
- extracts tables, signatures and stamps, with position boxes and small image snippets,
- writes a short summary and detects the document language,
- creates a search vector for each document, so users can search by meaning,
- answers questions about one document or about all documents of a company.

The main analysis is done by **Google Gemini 2.5 Flash** through the Google Gen AI API.
Only the search embeddings are created locally.

### Intended users

- **The Aura backend**, which uploads documents and polls for the results.
- **The Aura Control Panel**, which monitors the tier services (`/api/v1/status`).
- **Developers** who need document AI through a REST API.

### Part of the Aura platform

The Aura AI service exists in three tiers. Each tier is its own repository and service.
According to the code, this service mirrors the API contract of the Pro tier, so the Aura
backend can route requests to it without code changes.

| Tier | Repository / engine | Default port |
|------|---------------------|--------------|
| Basic (Tier 1) | `AuraAITier1`, local models (Llama 3.2 3B, PaddleOCR) | 9000 |
| **Plus (this repository)** | **Google Gemini 2.5 Flash** | **9001** |
| Pro | Google Gemini 2.5 Pro (per the comments in `config/settings.py`) | 9002 |

The service shares the MongoDB database `aura` (collection `documents`) with the Aura backend.

---

## Project Status

Version 1.0.0 (`API_VERSION` in `config/settings.py`). The repository has two commits
(April 2026). The core pipeline is implemented; there are no tests and no deployment files.

| Feature | Status |
|---------|--------|
| PDF and image upload | Working |
| Text extraction from digital PDFs (PyMuPDF) | Working |
| Analysis with Gemini 2.5 Flash (one call per file) | Working |
| Classification with label normalisation | Working |
| Structured fields, entities, tables, signatures, stamps | Working (as returned by Gemini) |
| Image snippets of signatures and stamps | Working when Gemini returns a position box |
| Semantic search | Working (brute-force cosine similarity over MongoDB) |
| Question answering and conversational search | Working |
| Splitting one PDF into several logical documents | Not implemented; one file = one document |
| Physical scanner input | Not available on the Plus tier |
| Authentication | Not implemented |
| Tests | None |
| Docker / deployment files | None |

---

## How It Works

```
   Aura backend / Control Panel
                |
                | REST (JSON, multipart)
                v
   +-------------------------------------------+
   |  Plus Tier AI Service (FastAPI, port 9001) |
   |                                           |
   |  /api/v1/upload  -->  background task      |
   |                          |                |
   |                          v                |
   |        validate > text layer > images >    |
   |        Gemini analysis > normalise >       |
   |        embed > store                       |
   +------+-----------------+---------------+--+
          |                 |               |
          v                 v               v
   +-------------+  +----------------+  +--------------------+
   | MongoDB     |  | Google Gemini  |  | Local embedding    |
   | db `aura`   |  | 2.5 Flash      |  | model (MiniLM,     |
   | `documents` |  | (cloud API)    |  | 384 dimensions)    |
   +-------------+  +----------------+  +--------------------+
```

1. The Aura backend uploads a file to `/api/v1/upload` (or `/api/v1/upload-and-process`)
   together with a `doc_id`. The file is saved as `data/uploads/<doc_id>.<ext>`.
   A file that is already in `data/uploads` or `data/temp` can be queued with `/api/v1/process`.
2. The service returns `{"status": "queued"}` at once and runs the pipeline as a background task.
3. The pipeline sets the record status to `PROCESSING`, analyses the file and writes the
   result into `processed_data`. The status becomes `COMPLETE` (or `FAILED` with an `error`).
4. The client polls `/api/v1/documents/{doc_id}/status` and receives the results when the
   status is `COMPLETE`.
5. The client can then use `/api/v1/search`, `/api/v1/qa` and `/api/v1/conversational-search`.

The service updates an existing record in the `documents` collection; it does not create
one. The record (with the same `_id`) is expected to be created by the Aura backend first.

---

## Processing Pipeline

Implemented in `app/core/pipeline.py` (`run_pipeline`).

| Step | What it does | Technology |
|------|--------------|------------|
| 1. Validate | Checks that the file exists and is not larger than `MAX_PDF_SIZE_MB`; sets status `PROCESSING` | Python |
| 2. Text layer | For PDFs, reads the native text of every page (empty for scans) | PyMuPDF (`fitz`) |
| 3. Page images | Converts PDF pages to JPEG images at `IMAGE_DPI`, up to `MAX_PAGES_PER_DOCUMENT` pages. Image files (`.png`, `.jpg`, `.tif`, ...) are loaded directly as one page | pdf2image + Poppler, Pillow |
| 4. Gemini analysis | Sends all page images plus up to 20,000 characters of the text layer in **one** request. Gemini returns one JSON object | Gemini 2.5 Flash |
| 5. Normalise | Maps field name aliases to canonical snake_case keys (for example invoice number, dates, vendor, customer, amounts, currency) and maps translated type labels (for example `rechnung` to `invoice`) to the English type list; unknown types become `other` | Python rules |
| 6. Embedding | Creates a 384-dimension vector from the extracted text | paraphrase-multilingual-MiniLM-L12-v2 |
| 7. Store | Writes one entry to `processed_data`, sets status `COMPLETE` and `processing_completed` | MongoDB |

### Gemini request details

- System prompt asks for raw JSON with these keys: `ocr_text`, `language` (ISO 639-1),
  `handwriting_detected`, `page_quality`, `classification` (`type`, `confidence`,
  `reasoning`), `structured_fields`, `entities`, `tables`, `signature`, `stamps`, `summary`.
- The prompt has extra rules for handwritten documents and for documents in any language
  (keys stay English, values stay in the source language, dates become `YYYY-MM-DD`).
- `temperature=0.1`, `response_mime_type="application/json"`,
  `max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS` (default 32768).
- If the answer is cut off because of the token limit, the service retries once with a
  larger budget (at least 60000 tokens) and can also try to repair truncated JSON.
- For the signature and every stamp with a position box, the service crops a snippet from
  the page image and stores it as Base64 (`snippet_base64`).

### Stored result (one entry in `processed_data`)

Main fields: `logical_doc_num` (always 1), `page_range`, `page_count`, `text` (first
2000 characters), `full_text`, `embedding`, `language`, `handwriting_detected`,
`page_quality`, `classification` (`label`, `confidence`, `reasoning`), `structured_data`,
`extraction_confidence`, `entities`, `tables`, `table_count`, `signature_detected`,
`signature`, `stamps`, `stamp_count`, `summary`, `ocr_validation`, `processed_at`.

### Document types

`invoice`, `credit_note`, `debit_note`, `payment_voucher`, `remittance_advice`,
`quotation`, `purchase_order`, `receipt`, `delivery_note`, `packing_slip`,
`bill_of_lading`, `shipping_document`, `contract`, `agreement`,
`non_disclosure_agreement`, `memorandum_of_understanding`, `business_letter`, `letter`,
`memo`, `proposal`, `tender_document`, `bank_statement`, `financial_statement`,
`tax_document`, `certificate`, `license`, `permit`, `form`, `report`, `statement`,
`resume`, `job_application`, `employment_contract`, `id_document`, `medical_record`, `other`.

---

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry point | `main.py` | Starts Uvicorn with `app.api.endpoints:app` on `API_HOST:API_PORT` |
| REST API | `app/api/endpoints.py` | FastAPI app, CORS, MongoDB connection, upload, status, results, search, Q&A |
| Pipeline | `app/core/pipeline.py` | Runs the steps above, normalises fields and types, writes to MongoDB |
| Gemini service | `app/services/gemini_service.py` | Prompt, Gemini client, PDF text extraction, JSON parsing and repair, snippet cropping |
| Embeddings | `app/services/embeddings.py` | Lazy-loaded Sentence Transformers model, `embed(text)` |
| Settings | `config/settings.py` | Reads environment variables (with `python-dotenv`) and creates data folders |

On startup the service connects to MongoDB (5 second timeout) and pre-loads the embedding
model. If MongoDB is not reachable, the service still starts, but database endpoints
return `503 Database not available`.

---

## AI Models

| Model | Where it runs | Used for |
|-------|---------------|----------|
| Google Gemini 2.5 Flash (`GEMINI_MODEL`, default `gemini-2.5-flash`) | Google cloud API | Reading text and handwriting, classification, fields, entities, tables, signatures, stamps, summary, Q&A, conversational answers |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Locally (CPU or GPU) | 384-dimension vectors for semantic search; downloaded from Hugging Face on first start |

The model name can be changed with `GEMINI_MODEL`. The comments in the code name
`gemini-2.5-pro` as the model of the Pro tier.

---

## Differences from Tier 1 (Basic)

| Topic | Tier 1 (`AuraAITier1`) | Plus tier (this repository) |
|-------|------------------------|-----------------------------|
| Default port | 9000 | 9001 |
| Main AI engine | Local: PaddleOCR, TrOCR, Llama 3.2 3B (Llamafile) | Cloud: Google Gemini 2.5 Flash |
| Data leaves the server | No | Yes, page images and text are sent to Google |
| Pipeline | Many separate steps with on/off toggles | One multimodal Gemini call per file, then normalisation |
| PDF splitting into several documents | Yes | No, one file = one document |
| Document types | 31 | 36 (including `other`) |
| Tables, stamps, position boxes, snippets | Tables disabled; signatures from text patterns | Tables, signature and stamps from Gemini, with boxes and image snippets |
| Search | FAISS index | Cosine similarity computed over all MongoDB records per request |
| API key header | `X-API-Key` | None |
| Runtime configuration API | Yes | No |
| Scanner support | Windows WIA | None (`/api/v1/scanners` returns an empty list) |

---

## Technology Stack

| Area | Technology |
|------|------------|
| Language | Python 3.10 or newer |
| Web framework | FastAPI, Uvicorn |
| AI (cloud) | `google-genai` SDK, Gemini 2.5 Flash |
| AI (local) | `sentence-transformers` |
| PDF and images | PyMuPDF, pdf2image (needs Poppler), Pillow |
| Database | MongoDB through `pymongo` |
| Other | `numpy`, `python-dotenv`, `python-multipart` |

The versions in `requirements.txt` are not pinned.

---

## Project Structure

```
adaptive-Dokumentenverarbeitung-Plus/
|-- main.py                     Entry point (Uvicorn)
|-- requirements.txt            Python dependencies
|-- config/
|   |-- __init__.py
|   `-- settings.py             Environment variables and defaults
|-- app/
|   |-- __init__.py
|   |-- api/
|   |   |-- __init__.py
|   |   `-- endpoints.py        FastAPI app and all endpoints
|   |-- core/
|   |   |-- __init__.py
|   |   `-- pipeline.py         Processing pipeline and normalisation
|   `-- services/
|       |-- __init__.py
|       |-- gemini_service.py   Gemini prompt, call and JSON parsing
|       `-- embeddings.py       Sentence Transformers embeddings
`-- data/
    |-- uploads/                Uploaded files (created at startup, not stored in Git)
    `-- temp/                   Temporary files (created at startup, not stored in Git)
```

`.env`, uploaded files, logs and `__pycache__` folders are excluded by `.gitignore`.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer (the code uses the `X | None` type syntax)
- MongoDB (local or remote)
- Poppler (needed by `pdf2image` to convert PDF pages; on Windows set `POPPLER_PATH`)
- A Google Gemini API key
- Internet access for the Gemini API and for the first download of the embedding model

### Install

```bash
git clone https://github.com/ahmadkobah/adaptive-Dokumentenverarbeitung-Plus-public.git
cd adaptive-Dokumentenverarbeitung-Plus-public
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root (see [Configuration](#configuration)).
At least `GEMINI_API_KEY` must be set; without it every analysis fails with
`GEMINI_API_KEY is not set`.

### Run

```bash
python main.py
```

The service listens on `http://0.0.0.0:9001` by default. The interactive API
documentation of FastAPI is available at `http://localhost:9001/docs`.

Check that it runs:

```bash
curl http://localhost:9001/health
```

---

## Configuration

All settings are read from environment variables (a `.env` file is loaded automatically).

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Address the server binds to |
| `API_PORT` | `9001` | Server port |
| `MONGODB_URI` | `mongodb://localhost:27017/` | MongoDB connection string |
| `DATABASE_NAME` | `aura` | MongoDB database name |
| `GEMINI_API_KEY` | (empty) | Google Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `GEMINI_MAX_OUTPUT_TOKENS` | `32768` | Output token limit for the analysis call |
| `IMAGE_DPI` | `200` | Resolution for PDF page images |
| `MAX_PDF_SIZE_MB` | `500` | Largest accepted file size |
| `MAX_PAGES_PER_DOCUMENT` | `100` | Highest number of pages converted to images |
| `POPPLER_PATH` | (not set) | Path to the Poppler binaries, if they are not on the PATH |
| `LOG_LEVEL` | `INFO` | Python log level |

Fixed values in the code: Q&A and conversational answers use `temperature=0.3` and
`max_output_tokens=1024`; Q&A context is limited to 50,000 characters; search
returns only matches with a similarity of at least 0.15 (conversational search: 0.10).

---

## API Reference

All endpoints use JSON unless noted. No authentication is required.

### Health and status

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service name, version and tier (`plus`) |
| GET | `/health` | `status`, `tier`, `model`, `timestamp` |
| GET | `/api/v1/status` | Status for Control Panel monitoring: tier, model, loaded models, `embedding_model_ready` |
| GET | `/api/v1/scanners` | Always an empty list; scanners are not available on the Plus tier |

### Processing

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/api/v1/upload` | multipart: `file`, `doc_id`, optional `company_id` | Saves the file and queues processing (alias used by the Aura backend) |
| POST | `/api/v1/upload-and-process` | same as above | Same behaviour as `/api/v1/upload` |
| POST | `/api/v1/process` | `{"doc_id", "file_path", "config_override"}` | Queues a file that is already inside `data/uploads` or `data/temp`; other paths return `403` |

Accepted upload types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`.
`doc_id` must be a valid MongoDB ObjectId string. `company_id` and `config_override`
are accepted but not used by the pipeline.

Response: `{"status": "queued", "doc_id": "...", "file": "<doc_id>.<ext>"}`.

### Status and results

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/status/{doc_id}` | `status`, `error`, `completed_at` |
| GET | `/api/v1/documents/{doc_id}/status` | Same, plus `processed_data` and `document_count` when the status is `COMPLETE` (polled by Aura) |
| GET | `/api/v1/results/{doc_id}` | `status`, `processed_data`, `document_count` |

Status values: `PROCESSING`, `COMPLETE`, `FAILED` (or `UNKNOWN` if not set).

### Search and questions

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/api/v1/search` | `{"query", "top_k": 10, "company_id"}` | Semantic search over all `COMPLETE` documents; returns score, file name, summary, classification, language, page range |
| POST | `/api/v1/qa` | `{"question", "doc_id", "company_id"}` | Gemini answers a question using the full text of one document (`doc_id`) or of the company's documents |
| POST | `/api/v1/conversational-search` | `{"query", "top_k", "company_id", "doc_id"}` | Finds matching documents, then Gemini writes one answer from them; returns `answer` and `source_documents`. If Gemini fails, the answer says so and the matches are still returned |

Example:

```bash
curl -X POST http://localhost:9001/api/v1/upload \
  -F "file=@invoice.pdf" \
  -F "doc_id=<24-character ObjectId>"

curl http://localhost:9001/api/v1/documents/<doc_id>/status
```

---

## Security

- **No authentication.** Every endpoint is open to anyone who can reach the port. Run the
  service only in a private network, behind the Aura backend or a gateway.
- **CORS** allows all origins, methods and headers, with credentials.
- **Path check:** `/api/v1/process` only accepts files inside `data/uploads` and `data/temp`.
  Uploaded files are saved under the name `<doc_id>.<ext>`, not under the client file name.
- **Data protection:** page images and document text are sent to Google (Gemini API).
  Check that this is allowed for your documents (GDPR / DSGVO) before you use this tier.
  Tier 1 keeps all processing local.
- **Secrets:** keep the Gemini API key and the MongoDB connection string in a local `.env`
  file and never commit it.

---

## Testing

There are no automated tests in this repository. To check the service by hand:

1. Start MongoDB and the service.
2. Create a record in the `documents` collection (or let the Aura backend create it).
3. Upload a sample PDF with the same `doc_id` and poll the status endpoint.
4. Try `/api/v1/search` and `/api/v1/qa` on the result.

---

## Known Issues

- One file always becomes one logical document. PDFs that contain several documents are
  not split.
- The service never creates a `documents` record. If no record with the given `doc_id`
  exists, the result is not stored anywhere.
- `run_pipeline` is an `async` function that makes blocking calls (Gemini, PDF conversion,
  embeddings). While a file is being processed, other requests can be delayed.
- `/api/v1/search` accepts `doc_id` but ignores it.
- `/api/v1/qa` has no error handling around the Gemini call; a Gemini error returns a
  server error.
- Search loads all matching records from MongoDB and compares vectors one by one. This
  becomes slow with many documents.
- `MAX_PAGES_PER_DOCUMENT` limits only the page images; the PyMuPDF text layer is read
  from all pages.
- Some names still refer to an earlier DeepSeek-based version: the docstring in
  `app/services/embeddings.py` and the stored flag `deepseek_powered: true`.
- `requirements.txt` has no version pins, and there is no Dockerfile.

---

## License

Private. All rights reserved.

Author: Ahmad Kobah
