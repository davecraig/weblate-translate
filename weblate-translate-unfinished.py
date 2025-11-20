#!/usr/bin/env python3
import os
import csv
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
from openai import OpenAI

try:
    import wlc
    from wlc.config import WeblateConfig
except ImportError:
    print("Please `pip install wlc` first.", file=sys.stderr)
    sys.exit(1)


import wlc
from urllib.parse import urlparse
def _to_path(base_url: str, full_url: str) -> str:
    """
    Convert full URL from Weblate `next` field into a path
    suitable for Weblate.get().
    """
    if not full_url:
        return None
    # strip base_url prefix if present
    if full_url.startswith(base_url):
        return full_url[len(base_url):]
    # otherwise, just return path+query
    parsed = urlparse(full_url)
    path = parsed.path.lstrip("/")
    if parsed.query:
        path += "?" + parsed.query
    return path


def get_units(client: wlc.Weblate,
              project_slug: str,
              component_slug: str,
              language_code: str,
              query: str):
    """
    Returns a list of untranslated units for one translation: project/component/language.
    """
    base_url = client.url
    path = (
        f"translations/{project_slug}/{component_slug}/{language_code}/units/"
        f"?q={query}"
    )

    units = []

    while path:
        data = client.get(path)
        for unit in data["results"]:
            # `source` and `target` are arrays to support plurals. :contentReference[oaicite:4]{index=4}
            source_list = unit.get("source") or [""]
            source_text = source_list[0]
            target_list = unit.get("target") or [""]
            target_text = target_list[0]

            units.append(
                {
                    "id": unit["id"],
                    "source": source_text,
                    "target": target_text,
                    "context": unit.get("context"),
                    "note": unit.get("note")
                }
            )

        next_url = data.get("next")
        path = _to_path(base_url, next_url) if next_url else None

    return units


# pip install tiktoken
from typing import List, Dict, Any, Tuple
import tiktoken

# --- Pricing per 1,000 tokens (USD). Update if pricing changes. ---
MODEL_PRICING = {
    "gpt-5.1":     {"input": 1.25/1000,  "output": 10.00/1000},
    "gpt-5-mini":  {"input": 0.25/1000,  "output": 2.00/1000},
    "gpt-5-nano":  {"input": 0.05/1000,  "output": 0.40/1000},
    "gpt-4.1":     {"input": 3.00/1000,  "output": 12.00/1000},
    "gpt-4.1-mini":{"input": 0.80/1000,  "output": 3.20/1000},
    "gpt-4.1-nano":{"input": 0.20/1000,  "output": 0.80/1000},
    "gpt-4o-mini": {"input": 4.0/1000, "output": 16.00/1000},
}

# --- Per-message token overhead heuristics for chat formatting ---
# These are based on OpenAI/tiktoken examples for chat models.
# If a model isn't listed, we fall back to a sensible default.
CHAT_OVERHEAD = {
    # name: (tokens_per_message, tokens_per_name, priming_tokens)
    "gpt-5.1":       (3, 1, 3),
    "gpt-5-mini":    (3, 1, 3),
    "gpt-5-nano":    (3, 1, 3),
    "gpt-4.1":       (3, 1, 3),
    "gpt-4.1-mini":  (3, 1, 3),
    "gpt-4.1-nano":  (3, 1, 3),
    "gpt-4o-mini":   (3, 1, 3),
}

def _encoding_for(model: str):
    """Get a tiktoken encoding for the given model, with a safe fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Safe fallback for most modern OpenAI chat models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Final fallback if tiktoken is very old
            return tiktoken.get_encoding(tiktoken.list_encoding_names()[0])

def count_tokens(text: str, model: str) -> int:
    """Count tokens for a single string."""
    enc = _encoding_for(model)
    return len(enc.encode(text))

def count_message_tokens(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> int:
    """
    Count tokens for a chat conversation.
    messages: list like [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello"}]
    """
    enc = _encoding_for(model)
    tpm, tpn, priming = CHAT_OVERHEAD.get(model, (3, 1, 3))

    total = priming  # e.g., <|start|> system priming, etc.

    for m in messages:
        total += tpm
        # role isn't usually counted through enc.encode in older recipes; it's part of overhead
        # content + name are encoded normally:
        content = m.get("content", "")
        total += len(enc.encode(content))
        name = m.get("name")
        if name:
            total += tpn
    # Assistant reply typically gets an extra 3 tokens in older templates; we skip here,
    # and instead let callers pass expected output tokens separately.
    return total

def estimate_chat_cost(
    messages: List[Dict[str, Any]],
    model: str,
    expected_output_tokens: int = 0
) -> Tuple[int, float]:
    """
    Return (input_tokens, estimated_total_cost_usd)
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model '{model}'. Please add it to MODEL_PRICING.")

    input_tokens = count_message_tokens(messages, model)
    pricing = MODEL_PRICING[model]

    input_cost  = (input_tokens / 1000) * pricing["input"]
    output_cost = (expected_output_tokens / 1000) * pricing["output"]
    return input_tokens, round(input_cost + output_cost, 6)

def translateText(target_language: str,
                  previously_translated_strings: list,
                  payload: list[dict[str, Any]],
                  openAiModel: str):
    system_msg = (
        "You are an expert translator with specific expertise in UK English to "
        f"{target_language} translations for audio-centric apps for visually impaired users.\n\n"
        f"Task: For each item translate `source` text into {target_language}, using the further context provided in `context`."
        "The translations will all be used within the Soundscape app which the user provides a description of."
        "It's important that terms are translated consistently and so the user will provide a "
        "list of already translated that can be used."
        "You must preserve any markdown and line breaks."
        "Return ONLY valid JSON with this exact shape:\n"
        "{ \"results\": [ {\"index\": <int>, \"key\": <string>, \"source\": <string>, \"target\": <string>}, ... ] }\n"
        "Do not add explanations or extra keys.\n\n"
        "Where a direct translation cannot be found, focus on creating clear, concise, "
        f"and contextually appropriate descriptive phrases in {target_language}, especially considering "
        "the aims of the Soundscape app."
    )

    soundscape_description_path = Path(f"soundscape-description.md")
    soundscape_msg = (
        "Here is an explanation of the Soundscape app." +
        soundscape_description_path.read_text()
    )

    previously_translated_strings_msg = (
        "Here is some JSON describing strings which have already been translated. source contains the original\n"
        "English, target contains the translation and note contains some extra context for translators:" +
        json.dumps(previously_translated_strings, ensure_ascii=False)
    )

    user_msg = (
        "Here is the JSON to translate. Again source contains the original English and note contains extra context to aid the translation:\n" +
        json.dumps(payload, ensure_ascii=False)
    )

    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": soundscape_msg},        
        {"role": "user", "content": previously_translated_strings_msg},
        {"role": "user", "content": user_msg},
    ]

    # Estimate the cost of translation
    in_tokens = count_message_tokens(conversation, openAiModel)
    print("Input tokens:", in_tokens)

    # The number of output tokens is likely to be similar to the number of input tokens
    conversation_output = [
        {"role": "user", "content": user_msg},
    ]
    expected_out = count_message_tokens(conversation_output, openAiModel)
    in_tokens, cost = estimate_chat_cost(conversation, openAiModel, expected_out)
    print(f"Estimated cost (incl. ~{expected_out} output tokens): ${cost}")

    if True:
        openAiClient = OpenAI(
            api_key = "",
        )
        resp = openAiClient.chat.completions.create(
            model=openAiModel,
            messages=conversation,
            response_format={"type": "json_object"}    # Ask the model to return strict JSON (no prose)
        )

        content = resp.choices[0].message.content
        data = json.loads(content)

        output_tokens = count_tokens(content, openAiModel)
        print("Actual output tokens:", output_tokens)

        return data
    

def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def client_from_config(path="weblate.ini"):
    cfg = WeblateConfig()
    cfg.read(path)
    return wlc.Weblate(config=cfg)

def load_or_fetch_units(
    filename,
    project,
    component,
    language,
    query,
    config_path
):
    if os.path.exists(filename):
        print(f"Loading units from cache: {filename}")
        return load_json(filename)

    print("No JSON cache found — fetching from Weblate…")
    client = client_from_config(config_path)
    units = get_units(client, project, component, language, query)
    write_json(filename, units)
    return units


def main():
    p = argparse.ArgumentParser(description="Report Weblate translation states via wlc.")
    p.add_argument("--project", default="soundscape-android", help="Project slug (as seen in URL).")
    p.add_argument("--component", default="android-app", help="Limit to a single component slug.")
    p.add_argument("--langs", nargs="*", help="Optional list of language codes to include (e.g. cs de fr).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of a table.")
    args = p.parse_args()

    # If we wanted to iterate over all the translations then we could use:
    #   https://hosted.weblate.org/api/components/soundscape-android/android-app/translations/
    # 
    # That returns a list of all the available translations (paste it into Chrome to see the results)
    LANGUAGES = {
        "arz":  "Egyptian Arabic",
        "da":   "Danish",
        "nl":   "Dutch",
        "fi":   "Finnish",
        "fr":   "French",
        "fr_CA":"Canadian French",
        "de":   "German",
        "el":   "Greek",
        "is":   "Icelandic",
        "it":   "Italian",
        "ja":   "Japanese",
        "nb_NO":"Norwegian Bokmål",
        "fa":   "Persian",
        "pl":   "Polish",
        "pt":   "Portuguese",
        "pt_BR":"Brazillian Portuguese",
        "ru":   "Russian",
        "es":   "Spanish",
        "sv":   "Swedish",
        "uk":   "Ukrainian"
    }

    for language in LANGUAGES:
        language_code = language
        language_string = LANGUAGES[language]

        print(f"Translating {language_string}...")

        untranslated_units = load_or_fetch_units(
            filename = f"{language_code}-untranslated.json",
            project = "soundscape-android",
            component = "android-app",
            language = language_code,
            query = "is:untranslated",
            config_path="weblate.ini"
        )
        translated_units = load_or_fetch_units(
            filename = f"{language_code}-translated.json",
            project = "soundscape-android",
            component = "android-app",
            language = language_code,
            query = "is:translated",
            config_path="weblate.ini"
        )


        print(f"Found {len(untranslated_units)} untranslated units")
        print(f"Found {len(translated_units)} translated units")

        # Translate them
        payload = [
            {
                "index": it.get("id", ""),
                "key":it.get("context", ""),
                "source": it.get("source", ""),
                "context": it.get("note")
            }
            for i, it in enumerate(untranslated_units)
        ]

        openAiModel="gpt-5-mini"

        # We're going to process a number of strings at a time. There's a small cost hit for
        # this, as the glossary and system instructions cost us on each call.
        target_language = language_string
        strings_at_a_time = 25
        start = 0
        end = strings_at_a_time
        accumulated_results = {}
        timeout_count = 0
        while start < end:
            try:
                results = translateText(target_language, translated_units, payload[start:end], openAiModel)
                if results != None:
                    # Transform results into simple JSON format that can be accepted by Weblate
                    for resource in results["results"]:
                        accumulated_results[resource["key"]] = resource["target"]

                    start = start + len(results["results"])
                else:
                    start = start + strings_at_a_time

                end = start + strings_at_a_time
                if end > len(payload):
                    end = len(payload)
            except Exception as inst:
                print(type(inst))    # the exception type
                print(inst.args)     # arguments stored in .args
                print(inst)

                timeout_count = timeout_count + 1
                time.sleep(10)
                # Try a smaller number of strings
                strings_at_a_time = strings_at_a_time = 10
                end = start + strings_at_a_time
            
            if timeout_count > 5:
                break

        out_path = Path(f"{target_language}-{openAiModel}-translations.json")
        out_path.write_text(json.dumps(accumulated_results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
