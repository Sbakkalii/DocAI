"""
Pydantic models for schema-injected VLM extraction.

Each document type has a dedicated model. The model's json_schema()
is passed to Ollama's `format` parameter for strict structured output.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[str] = None
    uom: Optional[str] = None
    unit_price: Optional[str] = None
    sub_total: Optional[str] = None


class InvoiceSchema(BaseModel):
    NUMBER: Optional[str] = None
    SUPPLIER: Optional[str] = None
    ADDRESS: Optional[str] = None
    INVOICE_DATE: Optional[str] = None
    TOTAL: Optional[str] = None
    TOTAL_AMOUNT: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)


class ContractSchema(BaseModel):
    CONTRACT_DATE: Optional[str] = None
    PARTIES: Optional[str] = None
    EFFECTIVE_DATE: Optional[str] = None
    TERMINATION_CLAUSE: Optional[str] = None
    SIGNATORY: Optional[str] = None
    CONTRACT_VALUE: Optional[str] = None
    SCOPE_OF_WORK: Optional[str] = None
    GOVERNING_LAW: Optional[str] = None


class PurchaseOrderSchema(BaseModel):
    PO_NUMBER: Optional[str] = None
    SUPPLIER: Optional[str] = None
    ORDER_DATE: Optional[str] = None
    DELIVERY_DATE: Optional[str] = None
    TOTAL: Optional[str] = None
    SHIPPING_ADDRESS: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)


class DeliveryNoteSchema(BaseModel):
    DN_NUMBER: Optional[str] = None
    SUPPLIER: Optional[str] = None
    DELIVERY_DATE: Optional[str] = None
    RECEIVER_NAME: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    GOODS_RECEIVED_BY: Optional[str] = None


class BankStatementSchema(BaseModel):
    ACCOUNT_NUMBER: Optional[str] = None
    STATEMENT_DATE: Optional[str] = None
    OPENING_BALANCE: Optional[str] = None
    CLOSING_BALANCE: Optional[str] = None
    BANK_NAME: Optional[str] = None
    IBAN: Optional[str] = None


class IDCardSchema(BaseModel):
    DOCUMENT_ID: Optional[str] = None
    FULL_NAME: Optional[str] = None
    DATE_OF_BIRTH: Optional[str] = None
    NATIONALITY: Optional[str] = None
    EXPIRY_DATE: Optional[str] = None
    DOCUMENT_NUMBER: Optional[str] = None
    GENDER: Optional[str] = None
    PLACE_OF_BIRTH: Optional[str] = None


DOCUMENT_TYPE_SCHEMAS: Dict[str, type] = {
    "invoice": InvoiceSchema,
    "contract": ContractSchema,
    "purchase_order": PurchaseOrderSchema,
    "delivery_note": DeliveryNoteSchema,
    "bank_statement": BankStatementSchema,
    "id_card": IDCardSchema,
}


def build_schema_for_document_type(doc_type: str) -> Dict[str, Any]:
    """Build a JSON schema for the given document type using Pydantic model_json_schema().

    For types with line items, the schema uses a consolidated line_items array
    instead of separate LINE/* fields.
    For unknown types, builds a union schema from all known document types.
    """
    model_class = DOCUMENT_TYPE_SCHEMAS.get(doc_type)
    if model_class is None:
        return _build_union_schema()

    schema = model_class.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    if "items" in schema.get("properties", {}).get("line_items", {}):
        schema["properties"]["line_items"]["items"].pop("title", None)
        for sub_prop in schema["properties"]["line_items"]["items"].get("properties", {}).values():
            sub_prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema


def _build_union_schema() -> Dict[str, Any]:
    """Build a union schema containing all fields from all known document types."""
    all_properties: Dict[str, Any] = {}
    all_defs: Dict[str, Any] = {}

    for model_class in DOCUMENT_TYPE_SCHEMAS.values():
        sub_schema = model_class.model_json_schema()
        props = sub_schema.get("properties", {})
        defs = sub_schema.get("$defs", {})
        all_defs.update(defs)
        for key, val in props.items():
            if key not in all_properties:
                val_copy = dict(val)
                val_copy.pop("title", None)
                if "items" in val_copy:
                    val_copy["items"] = dict(val_copy["items"])
                    val_copy["items"].pop("title", None)
                    for sub_prop in val_copy["items"].get("properties", {}).values():
                        sub_prop.pop("title", None)
                all_properties[key] = val_copy

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": all_properties,
        "additionalProperties": False,
    }
    if all_defs:
        schema["$defs"] = all_defs
    return schema


def get_fields_for_document_type(doc_type: str) -> List[str]:
    """Return the flat target field list for a document type (backward compat)."""
    from pipeline.config import DOCUMENT_TYPE_FIELDS
    return list(DOCUMENT_TYPE_FIELDS.get(doc_type, DOCUMENT_TYPE_FIELDS["invoice"]))


def build_output_schema(target_fields: List[str]) -> Dict[str, Any]:
    """Build a JSON schema for structured invoice extraction (legacy)."""
    properties: Dict[str, Any] = {}

    has_line_items = any(f.startswith("LINE/") for f in target_fields)
    if has_line_items:
        properties["line_items"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "quantity": {"type": ["string", "null"]},
                    "uom": {"type": ["string", "null"]},
                    "unit_price": {"type": ["string", "null"]},
                    "sub_total": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        }

    for f in target_fields:
        if f.startswith("LINE/"):
            continue
        properties[f] = {"type": ["string", "null"]}

    properties["_evidence"] = {
        "type": "object",
        "additionalProperties": {"type": ["string", "null"]},
    }

    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": [],
    }


def build_vlm_schema(target_fields: List[str]) -> Dict[str, Any]:
    """Legacy schema for VLM: each LINE/* is a separate array."""
    properties: Dict[str, Any] = {}
    for f in target_fields:
        if f.startswith("LINE/"):
            properties[f] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "quantity": {"type": ["string", "null"]},
                        "uom": {"type": ["string", "null"]},
                        "unit_price": {"type": ["string", "null"]},
                        "sub_total": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            }
        else:
            properties[f] = {"type": ["string", "null"]}
    properties["_evidence"] = {
        "type": "object",
        "additionalProperties": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": [],
    }


def build_ollama_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a JSON schema into Ollama's response_format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Invoice",
            "schema": schema,
            "strict": True,
        },
    }
