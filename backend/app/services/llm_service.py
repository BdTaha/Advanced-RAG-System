"""
LLM service — OpenAI ChatGPT API (primary) with flan-t5-base fallback.
Used for text generation, entity extraction, and graph construction.
"""

import torch
from typing import Optional
from app.core.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY, OPENAI_LLM_MODEL

# ── Global model cache ─────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None
_openai_client = None


def _load_openai():
    """Lazy-load the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"[LLM] OpenAI client configured (model: {OPENAI_LLM_MODEL}).")
        except Exception as e:
            print(f"[LLM] Failed to configure OpenAI: {e}")
            _openai_client = None
    return _openai_client


def _load_model():
    """Lazy-load the flan-t5-base model and tokenizer (fallback)."""
    global _model, _tokenizer, _device
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = DEFAULT_LLM_MODEL  # google/flan-t5-base
        print(f"[LLM] Loading {model_name} on {_device} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device)
        _model.eval()
        print(f"[LLM] {model_name} loaded successfully.")
    return _model, _tokenizer, _device


def generate_answer(prompt: str, max_new_tokens: int = 512) -> str:
    """Generate text using OpenAI API (primary) or flan-t5-base (fallback)."""
    # Try OpenAI first
    client = _load_openai()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tu es un assistant expert. Réponds de manière précise, détaillée et structurée. Réponds dans la même langue que la question. Si la réponse n'est pas dans le contexte, dis juste 'Information non trouvée'."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_new_tokens,
            )
            if response and response.choices:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] OpenAI generation failed: {e}, falling back to flan-t5")

    # Fallback to flan-t5-base
    model, tokenizer, device = _load_model()
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def generate_answer_gemini(prompt: str, max_tokens: int = 1024) -> Optional[str]:
    """Generate text using ONLY OpenAI API. Returns None on failure.
    (Function name kept as generate_answer_gemini for backward compatibility
     with agent_service and retrieval_service.)
    """
    client = _load_openai()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant expert. Réponds de manière précise, détaillée et structurée. Réponds dans la même langue que la question. Si la réponse n'est pas dans le contexte, dis juste 'Information non trouvée'."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        if response and response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] OpenAI-only generation failed: {e}")
    return None


def extract_entities_and_relations(text: str) -> dict:
    """Use flan-t5-base to extract entities and relations from a text chunk.

    Returns:
        {"entities": [{"name": ..., "type": ...}],
         "relations": [{"source": ..., "target": ..., "type": ...}]}
    """
    model, tokenizer, device = _load_model()

    # Prompt for entity extraction
    entity_prompt = (
        f"Extract the main entities (persons, species, places, concepts) "
        f"from this scientific text. List them as comma-separated items.\n\n"
        f"Text: {text[:400]}\n\nEntities:"
    )
    inputs = tokenizer(entity_prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128,
                                  num_beams=2, early_stopping=True)
    entity_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse entities
    raw_entities = [e.strip() for e in entity_text.split(",") if e.strip()]
    entities = []
    for e in raw_entities[:10]:  # Limit
        entities.append({"name": e, "type": "Concept"})

    # Prompt for relation extraction
    if len(entities) >= 2:
        entity_names = ", ".join([e["name"] for e in entities[:6]])
        rel_prompt = (
            f"Given these entities: {entity_names}\n"
            f"From this text: {text[:300]}\n"
            f"List relationships between them as: entity1 -> relationship -> entity2\n"
            f"Relations:"
        )
        inputs = tokenizer(rel_prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128,
                                      num_beams=2, early_stopping=True)
        rel_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse relations
        relations = _parse_relations(rel_text, entities)
    else:
        relations = []

    return {"entities": entities, "relations": relations}


def _parse_relations(rel_text: str, entities: list[dict]) -> list[dict]:
    """Parse LLM output into structured relations."""
    import re
    relations = []
    entity_names = {e["name"].lower() for e in entities}

    # Try to parse "A -> rel -> B" patterns
    parts = re.split(r'[;\n]', rel_text)
    for part in parts:
        part = part.strip()
        if "->" in part:
            segments = [s.strip() for s in part.split("->")]
            if len(segments) >= 3:
                relations.append({
                    "source": segments[0],
                    "target": segments[2],
                    "type": segments[1],
                })
        elif " is " in part.lower() or " are " in part.lower():
            # Simple "A is B" pattern
            for sep in [" is ", " are "]:
                if sep in part.lower():
                    idx = part.lower().index(sep)
                    src = part[:idx].strip()
                    tgt = part[idx + len(sep):].strip()
                    if src and tgt:
                        relations.append({
                            "source": src,
                            "target": tgt,
                            "type": "IS_A",
                        })
                    break

    # If no relations parsed, create co-occurrence relations
    if not relations and len(entities) >= 2:
        for i in range(min(len(entities) - 1, 3)):
            relations.append({
                "source": entities[i]["name"],
                "target": entities[i + 1]["name"],
                "type": "RELATED_TO",
            })

    return relations[:10]  # Limit
