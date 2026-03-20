from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from pypdf import PdfReader


def _get_anthropic_api_key() -> str | None:
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
                return key
    return None


def _parse_json_from_response(content: str) -> dict:
    text = (content or "").strip()
    if not text:
        raise ValueError("Anthropic returned empty content")

    decoder = json.JSONDecoder()

    def _as_dict(parsed: object) -> dict | None:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return None

    def _extract_balanced_json_snippet(raw: str) -> str | None:
        start_positions = [i for i, ch in enumerate(raw) if ch in "[{"]
        for start in start_positions:
            opener = raw[start]
            closer = "}" if opener == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return raw[start : i + 1]
        return None

    try:
        parsed, _ = decoder.raw_decode(text)
        as_dict = _as_dict(parsed)
        if as_dict is not None:
            return as_dict
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            parsed, _ = decoder.raw_decode(candidate)
            as_dict = _as_dict(parsed)
            if as_dict is not None:
                return as_dict
        except json.JSONDecodeError:
            snippet = _extract_balanced_json_snippet(candidate)
            if snippet:
                try:
                    parsed = json.loads(snippet)
                    as_dict = _as_dict(parsed)
                    if as_dict is not None:
                        return as_dict
                except json.JSONDecodeError:
                    pass

    snippet = _extract_balanced_json_snippet(text)
    if snippet:
        try:
            parsed = json.loads(snippet)
            as_dict = _as_dict(parsed)
            if as_dict is not None:
                return as_dict
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from Anthropic response")


def _expected_schema_keys(schema_hint: str) -> list[str]:
    try:
        parsed = json.loads(schema_hint)
    except Exception:
        return []

    if isinstance(parsed, dict):
        return [str(k) for k in parsed.keys()]

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return [str(k) for k in parsed[0].keys()]

    return []


def _normalize_to_schema_keys(payload: dict[str, Any], expected_keys: list[str]) -> dict[str, Any]:
    if not expected_keys:
        return payload
    out = dict(payload)
    for key in expected_keys:
        if key not in out:
            out[key] = None
    return out


def _validate_payload(
    payload: dict[str, Any],
    expected_keys: list[str],
    required_keys: list[str] | None,
) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload_not_dict"

    if required_keys:
        missing_required = [k for k in required_keys if k not in payload]
        if missing_required:
            return False, f"missing_required_keys:{','.join(missing_required)}"

    if expected_keys:
        overlap = sum(1 for k in expected_keys if k in payload)
        min_overlap = max(1, int(len(expected_keys) * 0.5))
        if overlap < min_overlap:
            return False, f"low_schema_overlap:{overlap}/{len(expected_keys)}"

    return True, "ok"


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def anthropic_extract_structured(
    text: str,
    schema_hint: str,
    model: str,
    strict_validation: bool = False,
    max_retries: int = 2,
    required_keys: list[str] | None = None,
) -> dict:
    api_key = _get_anthropic_api_key()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Please provide it to enable AI extraction.")

    client = Anthropic(api_key=api_key)
    prompt = (
        "You extract structured healthcare data from unstructured text. "
        "Return strict JSON only. Keep unknown fields as null. "
        "Do not infer unsupported clinical facts.\n\n"
        f"Schema hint:\n{schema_hint}\n\n"
        f"Input text:\n{text[:12000]}"
    )

    expected_keys = _expected_schema_keys(schema_hint)
    attempts = 0
    last_error = "unknown"
    last_content = ""

    while attempts <= max_retries:
        attempts += 1
        try:
            if attempts == 1:
                req_prompt = prompt
            else:
                req_prompt = (
                    "Your previous output failed strict validation. "
                    "Return strict JSON only, matching the schema keys exactly. "
                    "Include all schema keys with null when unknown. "
                    "No prose, no markdown.\n\n"
                    f"Validation failure: {last_error}\n\n"
                    f"Schema hint:\n{schema_hint}\n\n"
                    f"Input text:\n{text[:12000]}\n\n"
                    f"Previous output:\n{last_content[:12000]}"
                )

            message = client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0,
                messages=[{"role": "user", "content": req_prompt}],
            )

            content = "\n".join([block.text for block in message.content if hasattr(block, "text")])
            last_content = content
            parsed = _parse_json_from_response(content)

            if not strict_validation:
                return parsed

            normalized = _normalize_to_schema_keys(parsed, expected_keys)
            valid, reason = _validate_payload(normalized, expected_keys, required_keys)
            if valid:
                return normalized
            last_error = reason
            continue

        except Exception as ex:
            last_error = str(ex)
            continue

    raise ValueError(f"AI extraction failed strict validation after retries: {last_error}")


def extract_from_pdf_with_ai(
    pdf_path: str,
    schema_hint: str,
    model: str,
    strict_validation: bool = False,
    max_retries: int = 2,
    required_keys: list[str] | None = None,
) -> dict:
    text = extract_pdf_text(pdf_path)
    return anthropic_extract_structured(
        text=text,
        schema_hint=schema_hint,
        model=model,
        strict_validation=strict_validation,
        max_retries=max_retries,
        required_keys=required_keys,
    )
