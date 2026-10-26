"""
Pipeline — Plus Tier
PDF → text extraction (PyMuPDF) → images → Gemini 2.5 Flash → structured result → MongoDB
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Union

from bson import ObjectId
from pdf2image import convert_from_path
from pymongo.database import Database

from config.settings import (
    IMAGE_DPI,
    MAX_PDF_SIZE_MB,
    MAX_PAGES,
    POPPLER_PATH,
)
from app.services.gemini_service import analyse_document, extract_text_from_pdf
from app.services.embeddings import embed

logger = logging.getLogger("pipeline")


# ── Field-name normalisation ─────────────────────────────────────────────
# Maps free-form field aliases that Gemini may emit → canonical
# snake_case keys expected by Aura's autoFilingService FILING_RULES.
# Covers: English variants, German, French, Spanish, Italian, Dutch, Turkish, Arabic, Chinese.
_FIELD_ALIAS_MAP: Dict[str, str] = {
    # ── Invoice number ────────────────────────────────────
    "invoice_no": "invoice_number", "invoice_nr": "invoice_number",
    "invoicenumber": "invoice_number", "invoice number": "invoice_number",
    "bill_number": "invoice_number", "bill_no": "invoice_number",
    "reference_number": "invoice_number", "ref_number": "invoice_number",
    "doc_number": "invoice_number",
    "rechnungsnummer": "invoice_number", "rechnungsnr": "invoice_number",          # de
    "numéro_de_facture": "invoice_number", "numero_de_facture": "invoice_number",   # fr
    "n_de_facture": "invoice_number", "facture_n": "invoice_number",
    "número_de_factura": "invoice_number", "numero_de_factura": "invoice_number",  # es
    "factura_número": "invoice_number", "factura_numero": "invoice_number",
    "numero_fattura": "invoice_number", "n_fattura": "invoice_number",              # it
    "factuurnummer": "invoice_number",                                              # nl
    "fatura_no": "invoice_number", "fatura_numarası": "invoice_number",            # tr
    "رقم_الفاتورة": "invoice_number",                                            # ar
    "发票号码": "invoice_number", "发票号": "invoice_number",                          # zh
    # ── Invoice date ──────────────────────────────────────
    "invoice date": "invoice_date", "bill_date": "invoice_date",
    "issue_date": "invoice_date", "doc_date": "invoice_date",
    "rechnungsdatum": "invoice_date",                                               # de
    "date_de_facture": "invoice_date", "date_facture": "invoice_date",              # fr
    "fecha_de_factura": "invoice_date", "fecha_factura": "invoice_date",            # es
    "data_fattura": "invoice_date",                                                 # it
    "factuurdatum": "invoice_date",                                                 # nl
    "fatura_tarihi": "invoice_date",                                                # tr
    "تاريخ_الفاتورة": "invoice_date",                                          # ar
    "发票日期": "invoice_date",                                                       # zh
    # ── Purchase order ────────────────────────────────────
    "purchase_order_number": "po_number", "purchaseorder_number": "po_number",
    "purchase order number": "po_number", "po_no": "po_number", "ponumber": "po_number",
    "order_number": "po_number", "order_no": "po_number",
    "job_number": "po_number", "file_number": "po_number",
    "bestellnummer": "po_number", "bestellnr": "po_number",                         # de
    "numéro_de_commande": "po_number", "numero_de_commande": "po_number",            # fr
    "número_de_pedido": "po_number", "numero_de_pedido": "po_number",                # es
    "numero_ordine": "po_number",                                                   # it
    "bestelnummer": "po_number",                                                    # nl
    "sipariş_no": "po_number",                                                      # tr
    "رقم_الطلب": "po_number", "رقم_أمر_الشراء": "po_number",                       # ar
    "订单号": "po_number", "采购单号": "po_number",                                # zh
    # ── Order date ────────────────────────────────────────
    "order date": "order_date", "purchase_date": "order_date",
    "bestelldatum": "order_date",                                                   # de
    "date_de_commande": "order_date",                                               # fr
    "fecha_de_pedido": "order_date",                                                # es
    "sipariş_tarihi": "order_date",                                                 # tr
    "تاريخ_الطلب": "order_date",                                                # ar
    # ── Vendor / supplier ─────────────────────────────────
    "supplier_name": "vendor_name", "supplier": "vendor_name", "vendor": "vendor_name",
    "seller_name": "vendor_name", "seller": "vendor_name",
    "from_name": "vendor_name", "issuer_name": "vendor_name",
    "biller_name": "vendor_name", "service_provider": "vendor_name",
    "company_name": "vendor_name",
    "lieferant": "vendor_name", "verkäufer": "vendor_name",                          # de
    "fournisseur": "vendor_name", "vendeur": "vendor_name",                         # fr
    "proveedor": "vendor_name", "vendedor": "vendor_name",                          # es
    "fornitore": "vendor_name", "venditore": "vendor_name",                         # it
    "leverancier": "vendor_name",                                                   # nl
    "satıcı": "vendor_name", "tedarikçi": "vendor_name",                            # tr
    "البائع": "vendor_name", "اسم_البائع": "vendor_name", "المورد": "vendor_name",      # ar
    "供应商": "vendor_name", "卖方": "vendor_name",                                  # zh
    # ── Customer / buyer ──────────────────────────────────
    "buyer_name": "customer_name", "client_name": "customer_name",
    "bill_to": "customer_name", "purchaser_name": "customer_name",
    "recipient": "customer_name",
    "kunde": "customer_name", "käufer": "customer_name",                            # de
    "client": "customer_name", "acheteur": "customer_name",                         # fr
    "cliente": "customer_name", "comprador": "customer_name",                       # es / it
    "klant": "customer_name",                                                       # nl
    "müşteri": "customer_name", "alıcı": "customer_name",                            # tr
    "العميل": "customer_name", "اسم_العميل": "customer_name", "المشتري": "customer_name",     # ar
    "客户": "customer_name", "买方": "customer_name",                                  # zh
    # ── Amounts ───────────────────────────────────────────
    "total": "total_amount", "grand_total": "total_amount",
    "amount_due": "total_amount", "amount": "total_amount",
    "gesamtbetrag": "total_amount", "endbetrag": "total_amount",                    # de
    "montant_total": "total_amount", "total_ttc": "total_amount",                   # fr
    "importe_total": "total_amount", "total_factura": "total_amount",               # es
    "importo_totale": "total_amount", "totale": "total_amount",                     # it
    "totaalbedrag": "total_amount",                                                 # nl
    "toplam": "total_amount", "genel_toplam": "total_amount",                       # tr
    "الإجمالي": "total_amount", "المجموع": "total_amount",                            # ar
    "总金额": "total_amount", "总计": "total_amount",                                  # zh
    "subtotal_amount": "subtotal",
    "zwischensumme": "subtotal",                                                    # de
    "sous_total": "subtotal",                                                       # fr
    "subtotal_factura": "subtotal",                                                 # es
    "ara_toplam": "subtotal",                                                       # tr
    "小计": "subtotal",                                                            # zh
    "tax": "tax_amount", "vat": "tax_amount", "vat_amount": "tax_amount",
    "mwst": "tax_amount", "mehrwertsteuer": "tax_amount",                           # de
    "tva": "tax_amount",                                                            # fr
    "iva": "tax_amount", "impuesto": "tax_amount",                                  # es / it
    "btw": "tax_amount",                                                            # nl
    "kdv": "tax_amount", "vergi": "tax_amount",                                     # tr
    "الضريبة": "tax_amount", "ضريبة_القيمة_المضافة": "tax_amount",            # ar
    "税金": "tax_amount", "增值税": "tax_amount",                                  # zh
    # ── Receipts / delivery ────────────────────────────────
    "receipt_no": "receipt_number",
    "delivery_no": "delivery_note_number", "delivery_number": "delivery_note_number",
    "lieferscheinnummer": "delivery_note_number",                                   # de
    # ── Generic dates ─────────────────────────────────────
    "document_date": "date", "datum": "date",                                       # de
    "date_du_document": "date",                                                     # fr
    "fecha": "date",                                                                # es
    "data": "date",                                                                 # it
    "tarih": "date",                                                                # tr
    "التاريخ": "date",                                                            # ar
    "日期": "date",                                                                # zh
    # ── Currency ──────────────────────────────────────────
    "währung": "currency",                                                          # de
    "devise": "currency",                                                           # fr
    "moneda": "currency",                                                           # es
    "valuta": "currency",                                                           # it / nl
    "para_birimi": "currency",                                                      # tr
    "العملة": "currency",                                                          # ar
    "货币": "currency",                                                            # zh
}

# Canonical English document types. classification.type from Gemini must be one of these;
# anything else is normalised to a canonical key when possible, otherwise "other".
_VALID_DOC_TYPES = {
    "invoice", "contract", "agreement", "non_disclosure_agreement",
    "memorandum_of_understanding", "receipt", "letter", "business_letter",
    "purchase_order", "delivery_note", "packing_slip", "bill_of_lading",
    "shipping_document", "bank_statement", "financial_statement",
    "tax_document", "id_document", "medical_record", "form", "report",
    "credit_note", "debit_note", "payment_voucher", "remittance_advice", "quotation",
    "tender_document", "resume", "job_application", "employment_contract",
    "certificate", "license", "permit", "memo", "proposal", "statement", "other",
}

# Maps localised classification labels (German/French/Spanish/Arabic/Chinese/etc.) to
# canonical English types in case Gemini slips and returns a translated label.
_CLASSIFICATION_LANG_MAP: Dict[str, str] = {
    "rechnung": "invoice", "rechnungen": "invoice",                                 # de
    "facture": "invoice",                                                           # fr
    "factura": "invoice",                                                           # es
    "fattura": "invoice",                                                           # it
    "factuur": "invoice",                                                           # nl
    "fatura": "invoice",                                                            # tr
    "فاتورة": "invoice",                                                          # ar
    "发票": "invoice",                                                              # zh
    "bestellung": "purchase_order",
    "bon_de_commande": "purchase_order", "commande": "purchase_order",
    "pedido": "purchase_order", "orden_de_compra": "purchase_order",
    "ordine": "purchase_order", "ordine_di_acquisto": "purchase_order",
    "sipariş": "purchase_order", "satın_alma_emri": "purchase_order",
    "أمر_شراء": "purchase_order", "طلب_شراء": "purchase_order",
    "采购单": "purchase_order", "订单": "purchase_order",
    "quittung": "receipt", "beleg": "receipt",
    "reçu": "receipt", "recibo": "receipt", "ricevuta": "receipt",
    "makbuz": "receipt", "fiş": "receipt",
    "إيصال": "receipt", "收据": "receipt",
    "vertrag": "contract", "vereinbarung": "contract",
    "contrat": "contract", "accord": "contract",
    "contrato": "contract", "contratto": "contract",
    "sözleşme": "contract",
    "عقد": "contract", "اتفاقية": "contract",
    "合同": "contract", "协议": "contract",
    "lieferschein": "delivery_note",
    "bon_de_livraison": "delivery_note",
    "albarán": "delivery_note", "nota_de_entrega": "delivery_note",
    "bolla_di_consegna": "delivery_note",
    "irsaliye": "delivery_note",
    "إيصال_تسليم": "delivery_note", "送货单": "delivery_note",
    "kontoauszug": "bank_statement", "bankauszug": "bank_statement",
    "relevé_bancaire": "bank_statement",
    "extracto_bancario": "bank_statement",
    "estratto_conto": "bank_statement",
    "banka_ekstresi": "bank_statement", "hesap_ekstresi": "bank_statement",
    "كشف_حساب": "bank_statement", "银行对账单": "bank_statement",
    "steuerunterlage": "tax_document", "steuerbescheid": "tax_document",
    "document_fiscal": "tax_document", "avis_d_imposition": "tax_document",
    "documento_fiscal": "tax_document",
    "vergi_belgesi": "tax_document",
    "وثيقة_ضريبية": "tax_document",
    "brief": "letter", "lettre": "letter", "carta": "letter",
    "lettera": "letter", "mektup": "letter",
    "خطاب": "letter", "رسالة": "letter", "信函": "letter",
    "angebot": "quotation", "devis": "quotation",
    "cotización": "quotation", "presupuesto": "quotation",
    "preventivo": "quotation", "teklif": "quotation",
    "عرض_أسعار": "quotation", "عرض_سعر": "quotation",
    "报价单": "quotation",
}

_DATE_DDMMYYYY_RE = re.compile(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})$")
_DATE_YYYYMMDD_RE = re.compile(r"^(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})$")
_AMOUNT_EU_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*,\d{1,2}$")
_AMOUNT_US_RE = re.compile(r"^-?\d{1,3}(,\d{3})*\.\d{1,2}$")


def _norm_value(v: Any) -> Any:
    """Best-effort normalise a single field value.

    - Dates: 31.12.2024 / 31-12-2024 / 31/12/2024 → 2024-12-31; ISO already correct.
    - Amounts: 1.234,56 → 1234.56 ; 1,234.56 → 1234.56 ; pure decimals stay float.
    - Strings: trimmed.
    """
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return s
    # Date DD.MM.YYYY → ISO
    m = _DATE_DDMMYYYY_RE.match(s)
    if m:
        d, mo, y = m.groups()
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return s
    # Already ISO-like
    m = _DATE_YYYYMMDD_RE.match(s)
    if m:
        y, mo, d = m.groups()
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return s
    # EU amount 1.234,56
    if _AMOUNT_EU_RE.match(s):
        try:
            return float(s.replace(".", "").replace(",", "."))
        except ValueError:
            return s
    # US amount 1,234.56
    if _AMOUNT_US_RE.match(s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return s
    return s


def _normalise_structured_fields(gemini_result: Dict[str, Any]) -> None:
    """
    Rewrite keys in gemini_result['structured_fields'] to canonical snake_case
    and normalise common date / amount formats. Mutates in place.
    """
    fields = gemini_result.get("structured_fields")
    if not isinstance(fields, dict):
        return

    normalised: Dict[str, Any] = {}
    renamed: Dict[str, str] = {}

    for raw_key, value in fields.items():
        canonical = (raw_key or "").strip().lower().replace("-", "_")
        canonical = _FIELD_ALIAS_MAP.get(canonical, canonical)
        canonical = canonical.replace(" ", "_")
        if canonical != raw_key:
            renamed[raw_key] = canonical
        value = _norm_value(value)
        # Don't overwrite an existing real value with null from a duplicate alias
        if canonical in normalised and (value in (None, "", [])):
            continue
        normalised[canonical] = value

    if renamed:
        logger.info(f"Field-name normalisation applied: {renamed}")

    gemini_result["structured_fields"] = normalised


def _normalise_classification(gemini_result: Dict[str, Any]) -> None:
    """
    Force classification.type to a canonical English value from _VALID_DOC_TYPES.
    Translates localised labels (e.g. 'Rechnung' → 'invoice') and falls back to 'other'.
    Mutates in place.
    """
    cls = gemini_result.get("classification")
    if not isinstance(cls, dict):
        gemini_result["classification"] = {"type": "other", "confidence": 0.0, "reasoning": "missing classification"}
        return
    raw = (cls.get("type") or "other").strip().lower().replace(" ", "_").replace("-", "_")
    if raw in _VALID_DOC_TYPES:
        cls["type"] = raw
        return
    if raw in _CLASSIFICATION_LANG_MAP:
        translated = _CLASSIFICATION_LANG_MAP[raw]
        logger.info(f"Classification language fallback: {raw!r} → {translated!r}")
        cls["type"] = translated
        return
    logger.warning(f"Unknown classification.type {raw!r}, forcing to 'other'")
    cls["type"] = "other"


def _to_object_id(doc_id: Union[str, ObjectId]) -> ObjectId:
    if isinstance(doc_id, ObjectId):
        return doc_id
    return ObjectId(doc_id)


def _set_status(collection, doc_id: ObjectId, status: str, error: str = None) -> None:
    update: Dict[str, Any] = {"status": status}
    if error:
        update["error"] = error
    if status == "PROCESSING":
        update["processing_started"] = datetime.utcnow()
        update["processed_data"] = []
        update["error"] = None
    elif status in ("COMPLETE", "FAILED"):
        update["processing_completed"] = datetime.utcnow()
    collection.update_one({"_id": doc_id}, {"$set": update})


async def run_pipeline(
    doc_id: Union[str, ObjectId],
    pdf_path: str,
    db: Database,
) -> None:
    """
    Full Plus-tier pipeline for a single uploaded document.

    Steps:
      1. Validate file
      2. Extract native text from PDF using PyMuPDF
      3. Convert PDF pages to images (for visual snippets)
      4. Send pages to Gemini for structured analysis
      5. Generate search embedding from OCR text
      6. Attach image snippets (signature + stamps)
      7. Save to MongoDB
    """
    documents = db["documents"]
    doc_id = _to_object_id(doc_id)

    # ── 1. Validate ───────────────────────────────────────
    if not os.path.exists(pdf_path):
        _set_status(documents, doc_id, "FAILED", f"File not found: {pdf_path}")
        raise FileNotFoundError(pdf_path)

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        msg = f"File too large: {size_mb:.1f} MB (limit: {MAX_PDF_SIZE_MB} MB)"
        _set_status(documents, doc_id, "FAILED", msg)
        raise ValueError(msg)

    _set_status(documents, doc_id, "PROCESSING")
    logger.info(f"[{doc_id}] Pipeline started — {size_mb:.1f} MB")

    try:
        ext = os.path.splitext(pdf_path)[1].lower()
        is_image = ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

        if is_image:
            # ── 2/3. Image input: skip PyMuPDF + Poppler ──────
            from PIL import Image
            logger.info(f"[{doc_id}] Image input ({ext}) — loading directly")
            img = Image.open(pdf_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            pages = [img]
            extracted_text = ""  # no native text layer for raster images
            logger.info(f"[{doc_id}] 1 image page loaded ({img.size[0]}x{img.size[1]})")
        else:
            # ── 2. Extract text (PyMuPDF) ─────────────────────
            logger.info(f"[{doc_id}] Extracting text via PyMuPDF")
            extracted_text = extract_text_from_pdf(pdf_path)
            logger.info(f"[{doc_id}] Extracted {len(extracted_text)} chars from PDF")

            # ── 3. PDF → images (for visual snippets) ─────────
            logger.info(f"[{doc_id}] Converting PDF to images at {IMAGE_DPI} DPI")
            convert_kwargs: Dict[str, Any] = {
                "dpi": IMAGE_DPI,
                "last_page": MAX_PAGES,
                "fmt": "jpeg",
            }
            if POPPLER_PATH and os.path.exists(POPPLER_PATH):
                convert_kwargs["poppler_path"] = POPPLER_PATH

            pages = convert_from_path(pdf_path, **convert_kwargs)
            logger.info(f"[{doc_id}] {len(pages)} page(s) converted")

        # ── 4. Gemini analysis ──────────────────────────
        gemini_result = analyse_document(pages, extracted_text)

        # ── 4b. Normalise field names (alias → canonical snake_case) ──
        _normalise_structured_fields(gemini_result)
        _normalise_classification(gemini_result)

        # ── 5. Embedding (for semantic search) ────────────
        ocr_text: str = gemini_result.get("ocr_text", "") or extracted_text
        logger.info(f"[{doc_id}] Generating search embedding ({len(ocr_text)} chars)")
        search_embedding = embed(ocr_text)

        # ── 6. Build processed_doc ────────────────────────
        processed_doc = {
            "logical_doc_num": 1,
            "page_range": f"0-{len(pages) - 1}",
            "page_count": len(pages),

            # OCR / extracted text
            "text": ocr_text[:2000] + "..." if len(ocr_text) > 2000 else ocr_text,
            "full_text": ocr_text,
            "embedding": search_embedding,

            # Language & quality
            "language": gemini_result.get("language", "unknown"),
            "handwriting_detected": gemini_result.get("handwriting_detected", False),
            "page_quality": gemini_result.get("page_quality", 1.0),

            # Classification
            "classification": {
                "label": gemini_result.get("classification", {}).get("type", "other"),
                "confidence": gemini_result.get("classification", {}).get("confidence", 0.0),
                "reasoning": gemini_result.get("classification", {}).get("reasoning", ""),
            },

            # Structured field extraction
            "structured_data": gemini_result.get("structured_fields", {}),
            "extraction_confidence": gemini_result.get("classification", {}).get("confidence", 0.0),

            # Entities
            "entities": gemini_result.get("entities", {
                "companies": [], "people": [], "dates": [],
                "amounts": [], "addresses": [], "emails": [], "phone_numbers": [],
            }),

            # Tables
            "tables": gemini_result.get("tables", []),
            "table_count": len(gemini_result.get("tables", [])),

            # Signature
            "signature_detected": gemini_result.get("signature", {}).get("found", False),
            "signature": gemini_result.get("signature", {}),

            # Stamps
            "stamps": gemini_result.get("stamps", []),
            "stamp_count": len(gemini_result.get("stamps", [])),

            # Summary
            "summary": gemini_result.get("summary", ""),

            # Quality metadata
            "ocr_validation": {
                "quality_score": gemini_result.get("page_quality", 1.0),
                "corrections_made": 0,
                "issues_found": [],
            },

            "deepseek_powered": True,
            "processed_at": datetime.utcnow().isoformat(),
        }

        # ── 7. Save to MongoDB ────────────────────────────
        documents.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "processed_data": [processed_doc],
                    "status": "COMPLETE",
                    "processing_completed": datetime.utcnow(),
                    "error": None,
                }
            },
        )
        logger.info(f"[{doc_id}] Pipeline complete — saved to MongoDB")

    except Exception as e:
        logger.error(f"[{doc_id}] Pipeline failed: {e}", exc_info=True)
        _set_status(documents, doc_id, "FAILED", str(e))
        raise
