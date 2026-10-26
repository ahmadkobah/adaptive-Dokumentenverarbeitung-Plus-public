"""
Gemini Service — Plus Tier
Single entry point for all AI processing in the Plus tier.

Uses Google Gemini 2.5 Flash — the cheaper sibling of Pro's gemini-2.5-pro.
~10× cheaper, same vision/multilingual/tables/handwriting capabilities.

One call per logical document (all pages as images + one structured prompt).
PyMuPDF text (when available) is appended as supplemental context to help
on digital PDFs that have a selectable text layer.
"""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger("gemini_service")

# ── Gemini client (lazy init) ─────────────────────────────
_client: Optional[genai.Client] = None


def _ensure_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Prompt ────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert document analyst with deep knowledge of business, legal, and administrative documents in all languages, especially German.

Analyze the provided document pages and return ONLY a valid JSON object — no markdown fences, no extra text before or after.

The JSON must have exactly these keys:

{
  "ocr_text": "<full extracted text of all pages combined>",
  "language": "<ISO 639-1 code, e.g. de, en, fr, ar>",
  "handwriting_detected": <true|false>,
  "page_quality": <0.0-1.0 estimate of text/scan quality>,

  "classification": {
    "type": "<invoice|credit_note|debit_note|payment_voucher|remittance_advice|quotation|purchase_order|receipt|delivery_note|packing_slip|bill_of_lading|contract|agreement|non_disclosure_agreement|memorandum_of_understanding|business_letter|letter|memo|proposal|tender_document|bank_statement|financial_statement|tax_document|certificate|license|permit|form|report|statement|resume|job_application|employment_contract|id_document|medical_record|other>",
    "confidence": <0.0-1.0>,
    "reasoning": "<one sentence>"
  },

  "structured_fields": {
    "<field_name>": "<value or null>"
  },

IMPORTANT — structured_fields MUST use these EXACT snake_case keys (never spaces, never camelCase, never uppercase):

  invoice              → invoice_number, invoice_date, due_date, vendor_name, customer_name,
                          subtotal, tax_amount, total_amount, currency, payment_terms,
                          bank_account, iban, bic
  purchase_order       → po_number, order_date, vendor_name, customer_name,
                          delivery_date, total_amount, currency, payment_terms
  receipt              → receipt_number, date, merchant_name, total_amount, currency,
                          payment_method
  delivery_note        → delivery_note_number, delivery_date, recipient_name, sender_name,
                          po_number
  contract / agreement → contract_number, contract_date, party_a, party_b,
                          start_date, end_date, contract_value, currency
  bank_statement       → account_number, statement_period, opening_balance, closing_balance,
                          currency, iban, bic
  letter / business_letter → date, sender_name, sender_company, recipient_name, subject
  credit_note          → credit_note_number, date, vendor_name, total_amount, currency
  debit_note           → debit_note_number, date, vendor_name, total_amount, currency
  payment_voucher      → voucher_number, date, payer_name, payee_name, total_amount, currency
  remittance_advice    → remittance_number, date, payer_name, payee_name, total_amount, currency,
                          payment_method, invoice_numbers (list of invoices being paid)
  quotation            → quotation_number, date, customer_name, total_amount, currency,
                          valid_until
  tax_document         → tax_id, tax_year, tax_period, total_tax, currency

Do NOT invent variant key names like "purchase_order_number", "invoice_no", "InvoiceNumber",
"order number". Always use the canonical snake_case keys above. If a field is not present,
use null.

  "entities": {
    "companies":     ["<name>"],
    "people":        ["<full name>"],
    "dates":         ["<YYYY-MM-DD or original string>"],
    "amounts":       ["<numeric string with currency if available>"],
    "addresses":     ["<full address string>"],
    "emails":        ["<email>"],
    "phone_numbers": ["<phone>"]
  },

  "tables": [
    {
      "title":   "<table title or 'Table N'>",
      "page":    <1-based page number>,
      "headers": ["<col1>", "<col2>"],
      "rows":    [["<cell>", "<cell>"]],
      "bbox":    {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
    }
  ],

  "signature": {
    "found":          <true|false>,
    "signer_name":    "<full name or null>",
    "signer_title":   "<job title or null>",
    "signer_company": "<company or null>",
    "page":           <1-based page number or null>,
    "position":       "<bottom-right|bottom-left|center|other or null>",
    "bbox":           {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
  },

  "stamps": [
    {
      "issuing_authority": "<authority name or null>",
      "stamp_type":        "<official|company|received|approved|other>",
      "text_on_stamp":     "<visible stamp text or null>",
      "date_on_stamp":     "<YYYY-MM-DD or null>",
      "page":              <1-based page number>,
      "position":          "<bottom-left|bottom-right|top-right|other>",
      "bbox":              {"x": <0-1>, "y": <0-1>, "w": <0-1>, "h": <0-1>}
    }
  ],

  "summary": "<one concise paragraph summarising the document>"
}

Rules:
- structured_fields must contain ALL relevant fields for the document type (invoice_number, dates, amounts, parties, payment terms, bank details, etc.)
- bbox values are fractions of the page dimensions (0.0 to 1.0)
- If a field is not present, use null — do not omit the key
- tables must include ALL tables, including borderless/implicit ones
- stamps and signatures arrays may be empty ([]) if none found
- Return the raw JSON only — no ```json fences

Handwriting rules:
- If the document is fully or partially handwritten, read every handwritten field carefully.
- If no explicit invoice number label is present, look for ORDER NO., ORDER #, REF, REFERENCE, JOB NO., FILE NO. and use that value as invoice_number.
- For handwritten dates, normalize to YYYY-MM-DD.
- For handwritten amounts, strip currency symbols and commas — return digits only (e.g. "281.25" not "$281.25").
- Set handwriting_detected = true whenever any handwritten text is found.

Foreign-language rules:
- The document may be in any language (Arabic, German, French, Spanish, Chinese, Turkish, etc.).
- ALWAYS use the canonical English snake_case keys listed above — NEVER translate or transliterate them.
- Extract field VALUES as they appear in the source language but normalize dates to YYYY-MM-DD and amounts to digit strings.
- For classification.type, ALWAYS use the English type labels (invoice, contract, receipt, etc.) regardless of document language.
- For language, return the ISO 639-1 code (de, ar, fr, es, zh, tr, …).
"""


# ── Image helpers ─────────────────────────────────────────

def _pil_to_part(image: Image.Image) -> types.Part:
    """Convert a PIL image to a Gemini Part (JPEG)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(
        data=buf.getvalue(),
        mime_type="image/jpeg",
    )


def _crop_snippet(image: Image.Image, bbox: Dict[str, float]) -> Optional[str]:
    """
    Crop a region from a PIL image using normalised bbox and return base64 PNG.
    bbox keys: x, y, w, h — all in 0-1 range relative to image size.
    Returns None if bbox is invalid.
    """
    if not bbox or not all(k in bbox for k in ("x", "y", "w", "h")):
        return None
    try:
        iw, ih = image.size
        x1 = int(bbox["x"] * iw)
        y1 = int(bbox["y"] * ih)
        x2 = int((bbox["x"] + bbox["w"]) * iw)
        y2 = int((bbox["y"] + bbox["h"]) * ih)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Snippet crop failed: {e}")
        return None


# ── Text extraction helper (PyMuPDF) ──────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract native text from a PDF using PyMuPDF (fitz).
    Returns concatenated text from all pages, or "" if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages_text)
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available — text extraction skipped")
        return ""
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return ""


# ── JSON parser ───────────────────────────────────────────


class _TruncatedJSONError(ValueError):
    """Raised when Gemini returned a JSON string that was cut off mid-token."""


def _salvage_truncated_json(text: str) -> Dict[str, Any] | None:
    """Best-effort recovery of a JSON object cut mid-string by token cap."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in body:
        out.append(ch)
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_string:
        if escape:
            out.pop()
        out.append('"')
    repaired = "".join(out)
    repaired = re.sub(r",\s*$", "", repaired)
    while stack:
        repaired = re.sub(r'[,\s]*(?:"[^"]*"\s*:\s*)?$', "", repaired)
        repaired += stack.pop()
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse JSON from Gemini response, tolerating markdown fences and truncation."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(text[start:], start):
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        salvaged = _salvage_truncated_json(text)
        if salvaged is not None:
            logger.warning(
                f"Gemini JSON was truncated; salvaged a partial object "
                f"(len={len(text)} chars, error={e})"
            )
            return salvaged
        raise _TruncatedJSONError(
            f"Could not parse Gemini JSON response: {e}\nRaw (first 500): {raw[:500]}"
        )


# ── Main entry point ──────────────────────────────────────

def analyse_document(
    pages: List[Image.Image],
    extracted_text: str = "",
) -> Dict[str, Any]:
    """
    Send all pages of a logical document to Gemini 2.5 Flash in a single call.

    Args:
        pages: List of PIL Image objects (one per page, already at correct DPI).
        extracted_text: Optional native PDF text (PyMuPDF) appended as supplemental
                        context — useful for digital PDFs with selectable text.

    Returns:
        Parsed result dict with all fields described in _SYSTEM_PROMPT.
        Adds 'snippet_base64' to signature and each stamp if bbox is present.
    """
    client = _ensure_client()

    parts: List[Any] = [_SYSTEM_PROMPT]

    # Optional supplemental text layer (helps with digital PDFs)
    if extracted_text and extracted_text.strip():
        snippet = extracted_text.strip()
        if len(snippet) > 20000:
            snippet = snippet[:20000] + "\n... (truncated)"
        parts.append(
            "\n--- Supplemental text layer (from PDF, may be incomplete) ---\n"
            + snippet
            + "\n--- End supplemental text ---\n"
        )

    for i, page in enumerate(pages):
        parts.append(f"\n--- Page {i + 1} ---\n")
        parts.append(_pil_to_part(page))

    logger.info(f"Sending {len(pages)} page(s) to Gemini ({GEMINI_MODEL})")

    def _call(max_tokens: int):
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )

    response = _call(GEMINI_MAX_OUTPUT_TOKENS)

    raw_text = response.text
    logger.debug(f"Gemini raw response (first 300): {raw_text[:300]}")

    # Detect truncation due to token cap
    hit_max_tokens = False
    try:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and str(finish_reason).upper().endswith("MAX_TOKENS"):
            hit_max_tokens = True
            logger.warning(
                f"Gemini response hit MAX_TOKENS ({GEMINI_MAX_OUTPUT_TOKENS})."
            )
    except (AttributeError, IndexError):
        pass

    try:
        result = _parse_json_response(raw_text)
    except _TruncatedJSONError as exc:
        # Retry once with the largest available budget so the model can finish.
        big_budget = max(int(GEMINI_MAX_OUTPUT_TOKENS), 60000)
        logger.warning(
            f"JSON truncated, retrying with max_output_tokens={big_budget}"
        )
        response = _call(big_budget)
        raw_text = response.text
        result = _parse_json_response(raw_text)
    if hit_max_tokens:
        logger.info("Recovered partial result from MAX_TOKENS response.")

    # ── Attach image snippets for signature and stamps ────
    sig = result.get("signature", {})
    if sig and sig.get("found") and sig.get("bbox"):
        page_idx = (sig.get("page") or 1) - 1
        if 0 <= page_idx < len(pages):
            sig["snippet_base64"] = _crop_snippet(pages[page_idx], sig["bbox"])

    for stamp in result.get("stamps", []):
        if stamp.get("bbox"):
            page_idx = (stamp.get("page") or 1) - 1
            if 0 <= page_idx < len(pages):
                stamp["snippet_base64"] = _crop_snippet(pages[page_idx], stamp["bbox"])

    logger.info(
        f"Gemini analysis complete — type={result.get('classification', {}).get('type')}, "
        f"lang={result.get('language')}, "
        f"tables={len(result.get('tables', []))}, "
        f"sig={result.get('signature', {}).get('found')}, "
        f"stamps={len(result.get('stamps', []))}"
    )

    return result
