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

WEBLATE_URL=""
WEBLATE_TOKEN=""

try:
    import wlc
except ImportError:
    print("Please `pip install wlc` first.", file=sys.stderr)
    sys.exit(1)


def get_required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing environment variable: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def api_get_all(client: "wlc.Weblate", path: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate through a paginated Weblate API list endpoint and yield results.
    """
    url = path
    while url:
        page = client.get(url)
        # Weblate list responses have 'results', 'next' fields.
        for item in page.get("results", []):
            yield item
        # next can be absolute or relative; wlc accepts either in .get()
        url = page.get("next")


def list_components(client: "wlc.Weblate", project: str) -> List[Dict[str, Any]]:
    """
    Return all components (objects) within a project.
    """
    path = f"projects/{project}/components/?page_size=1000"
    return list(api_get_all(client, path))


def list_translations_for_component(client: "wlc.Weblate", project: str, component_slug: str
                                    ) -> List[Dict[str, Any]]:
    """
    Return translation objects (per language) for a given component.
    """
    path = f"components/{project}/{component_slug}/translations/?page_size=1000"
    return list(api_get_all(client, path))


def get_translation_statistics(client: "wlc.Weblate", project: str, component: str, language: str
                               ) -> Dict[str, Any]:
    """
    Fetch detailed statistics for a translation.
    """
    path = f"translations/{project}/{component}/{language}/statistics/"
    return client.get(path)


def maybe_filter_languages(items: List[Dict[str, Any]], languages: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not languages:
        return items
    wanted = {lang.lower() for lang in languages}
    out = []
    for t in items:
        # translation object usually has 'language_code' or language url; prefer code if present
        code = t.get("language_code") or t.get("language", "").split("/")[-2]  # fallback parse
        if code and code.lower() in wanted:
            out.append(t)
    return out

def get_weblate_statistics(client: "wlc.Weblate", args: dict):
    components = []
    if args.component:
        # Fetch specific component object quickly from API root to verify it exists
        try:
            comp = client.get(f"components/{args.project}/{args.component}/")
            components = [comp]
        except Exception as e:
            print(f"Error fetching component '{args.component}': {e}", file=sys.stderr)
            sys.exit(3)
    else:
        components = list_components(client, args.project)

    if not components:
        print("No components found.", file=sys.stderr)
        sys.exit(0)

    results = []  # will hold per translation statistics rows

    for comp in components:
        comp_slug = comp.get("slug") or comp.get("name")
        comp_name = comp.get("name") or comp_slug

        translations = list_translations_for_component(client, args.project, comp_slug)
        translations = maybe_filter_languages(translations, args.langs)

        for t in translations:
            language = t.get("language")
            code = language["code"]
            lang_name = language["name"]

            try:
                stats = get_translation_statistics(client, args.project, comp_slug, code)
            except Exception as e:
                print(f"Warning: could not fetch statistics for {comp_slug}:{lang_code}: {e}", file=sys.stderr)
                continue

            # Pull the fields we care about (see Weblate “Statistics” API)
            row = {
                "component": comp_name,
                "component_slug": comp_slug,
                "language": lang_name,
                "language_code": code,
                "translated": stats.get("translated"),
                "translated_percent": stats.get("translated_percent"),
                "fuzzy": stats.get("fuzzy"),
                "approved": stats.get("approved"),
                "readonly": stats.get("readonly"),
                "failing_checks": stats.get("failing"),
                "total": stats.get("total"),
                "total_words": stats.get("total_words"),
                "url": stats.get("url_translate") or t.get("web_url"),
            }
            results.append(row)

    if args.json:
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    # Pretty table output
    if not results:
        print("No matching translations found.")
        return

    # Compute column widths
    cols = [
        ("Name", "language", 24),
        ("Lang", "language_code", 7),
        ("% Tr", "translated_percent", 6),
        ("Tr", "translated", 6),
        ("Fuzzy", "fuzzy", 6),
        ("Approved", "approved", 8),
        ("RO", "readonly", 4),
        ("Fail", "failing_checks", 5),
        ("Total", "total", 6),
    ]
    header = " | ".join(f"{h:<{w}}" for h, _, w in cols)
    sep = "-+-".join("-" * w for _, _, w in cols)
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: (x["component"], x["language_code"])):
        def fmt(name, width):
            val = r.get(name)
            if name == "translated_percent" and val is not None:
                return f"{val:5.1f}%"
            return f"{val if val is not None else '':>{width}}"

        line = " | ".join(
            [
                f"{(r['language'] or ''):<24}",
                f"{(r['language_code'] or ''):<7}",
                f"{(f'{r['translated_percent']:.1f}%' if isinstance(r.get('translated_percent'), (int, float)) else ''):<6}",
                f"{r.get('translated',''):>6}",
                f"{r.get('fuzzy',''):>6}",
                f"{r.get('approved',''):>8}",
                f"{r.get('readonly',''):>4}",
                f"{r.get('failing_checks',''):>5}",
                f"{r.get('total',''):>6}",
            ]
        )
        print(line)


def dump_units(client: "wlc.Weblate", project: str, component: str, lang: str) -> List[Dict[str, Any]]:
    """
    Fetch all units for the chosen base translation, returning key metadata:
    source text, context (msgctxt), key/id, location, and comments/hints.
    """
    base = f"translations/{project}/{component}/{lang}"
    units_path = f"{base}/units/?page_size=200"
    rows = []
    weblate_results = api_get_all(client, units_path)
    for u in weblate_results:
        rows.append({
            "component": component,
            "language_code": lang,
            "unit_id": u.get("id"),
            "source": u.get("source")[0] or "",
            "context": u.get("context") or "",               
            "note": u.get("note") or "",                     
        })
    return rows

def trim_quotes(s: str) -> str:
    s = s.strip()  # remove leading/trailing whitespace
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.strip()  # remove leading/trailing whitespace
    return s

def parse_csv(csv_path: str) -> List[Dict[str, Any]]:

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, strict=True)
        rows = []
        for row in reader:
            source = trim_quotes(row.get("source"))
            context = trim_quotes(row.get("context"))
            developer_comments = trim_quotes(row.get("developer_comments"))
            rows.append({
                "source":  source,
                "context": context,
                "note": developer_comments,
            })

    return rows

# pip install tiktoken
from typing import List, Dict, Any, Tuple
import tiktoken

# --- Pricing per 1,000 tokens (USD). Update if pricing changes. ---
MODEL_PRICING = {
    "gpt-5":       {"input": 1.25/1000,  "output": 10.00/1000},
    "gpt-5-mini":  {"input": 0.25/1000,  "output": 2.60/1000},
    "gpt-4.1":     {"input": 2.00/1000,  "output": 8.00/1000},
    "gpt-4.1-mini":{"input": 0.40/1000,  "output": 1.60/1000},
    "gpt-4.1-nano":{"input": 0.10/1000,  "output": 0.40/1000},
    "gpt-4o":      {"input": 2.50/1000,  "output": 10.00/1000},
    "gpt-4o-mini": {"input": 0.15/1000, "output": 0.60/1000},
}

# --- Per-message token overhead heuristics for chat formatting ---
# These are based on OpenAI/tiktoken examples for chat models.
# If a model isn't listed, we fall back to a sensible default.
CHAT_OVERHEAD = {
    # name: (tokens_per_message, tokens_per_name, priming_tokens)
    "gpt-5":         (3, 1, 3),
    "gpt-5-mini":    (3, 1, 3),
    "gpt-4.1":       (3, 1, 3),
    "gpt-4.1-mini":  (3, 1, 3),
    "gpt-4.1-nano":  (3, 1, 3),
    "gpt-4o":        (3, 1, 3),
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
                  glossary: list,
                  payload: list[dict[str, Any]],
                  openAiModel: str):
    system_msg = (
        "You are an expert translator with specific expertise in UK English to "
        f"{target_language} translations for audio-centric apps for visually impaired users.\n\n"
        f"Task: For each item translate `source` text into {target_language}, using the further context provided in `context`."
        "The translations will all be used within the Soundscape app which the user provides a description of."
        "It's important that terms are translated consistently and so the user will provide a "
        "glossary of common terms that have already been translated and should be used."
        "You must preserve any markdown and line breaks."
        "Return ONLY valid JSON with this exact shape:\n"
        "{ \"results\": [ {\"index\": <int>, \"key\": <string>, \"source\": <string>, \"translation\": <string>}, ... ] }\n"
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

    glossary_msg = (
        "Here is a glossary of important terms which already have been translated:\n" +
        json.dumps(glossary, ensure_ascii=False)
    )

    user_msg = (
        "Items to translate (JSON):\n" +
        json.dumps(payload, ensure_ascii=False)
    )

    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": soundscape_msg},        
        {"role": "user", "content": glossary_msg},
        {"role": "user", "content": user_msg},
    ]

    # Estimate the cost of translation
    in_tokens = count_message_tokens(conversation, openAiModel)
    print("Input tokens:", in_tokens)

    # The number of output tokens is likely to be similar to the number of input tokens
    expected_out = in_tokens 
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
    

def main():
    p = argparse.ArgumentParser(description="Report Weblate translation states via wlc.")
    p.add_argument("--project", default="soundscape-android", help="Project slug (as seen in URL).")
    p.add_argument("--component", default="android-app", help="Limit to a single component slug.")
    p.add_argument("--langs", nargs="*", help="Optional list of language codes to include (e.g. cs de fr).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of a table.")
    args = p.parse_args()

    if False:
        # Get all the strings directly from weblate - this uses way too many API calls
        url = WEBLATE_URL
        token = WEBLATE_TOKEN
        client = wlc.Weblate(key=token, url=url)

        # Get the strings that we want to translate
        units = dump_units(client, args.project, args.component, "en")

    if True:
        # Get all the strings from a CSV file downloaded from weblate
        units = parse_csv("soundscape-android-android-app-en.csv")


    # Translate them
    payload = [
        {"index": i, "key":it.get("context", ""), "source": it.get("source", ""), "context": it.get("note")}
        for i, it in enumerate(units)
    ]

#    glossary = [
#        { "term": "Ahead of Me", "French": "Devant moi" },
#        { "term": "Around Me",   "French": "Autour de moi" },
#        { "term": "My Location", "French": "Mon emplacement" },
#        { "term": "Nearby Markers", "French": "Marqueurs à proximité" },
#        { "term": "Audio Beacon", "French": "Balise sonore" },
#        { "term": "Intersection callouts", "French": "Annonces d’intersections" },
#        { "term": "Sleep button / Sleep mode", "French": "Bouton veille / Mode veille" },
#        { "term": "Marker", "French": "Marqueur" },
#        { "term": "Annotation", "French": "Annotation" },
#        { "term": "Routes", "French": "Itinéraires" },
#        { "term": "Waypoint", "French": "Points de repère" },
#        { "term": "Places Nearby button", "French": "Bouton lieux à proximité" },
#        { "term": "Search bar", "French": "Barre de recherche" },
#        { "term": "Highway", "French": "Autoroute" },
#        { "term": "Voices", "French": "Voix" },
#        { "term": "Offline mode", "French": "Mode hors ligne" },
#        { "term": "Settings", "French": "Réglages" },
#    ]

    glossary = [
        { "term": "Ahead of Me", "Polish": "Przede mną" },
        { "term": "Around Me", "Polish": "Wokół mnie" },
        { "term": "My Location", "Polish": "Moja lokalizacja" },
        { "term": "Nearby Markers", "Polish": "Bliskie znaczniki mapy (pinezki)" },
        { "term": "Audio Beacon", "Polish": "Dźwięk naprowadzający" },
        { "term": "Intersection callouts", "Polish": "Powiadomienia o skrzyżowaniach" },
        { "term": "Sleep", "Polish": "Tryb uśpienia" },
        { "term": "Marker", "Polish": "Znacznik (pinezka)" },
        { "term": "Annotation", "Polish": "Adnotacja" },
        { "term": "Routes", "Polish": "Trasy" },
        { "term": "Waypoint", "Polish": "Punkt trasy" },
        { "term": "Places Nearby", "Polish": "Miejsca w pobliżu" },
        { "term": "Search bar", "Polish": "Pasek wyszukiwania" },
        { "term": "Voices", "Polish": "Wybór głosu TTS" },
        { "term": "Offline mode", "Polish": "Tryb offline" }
    ]

    #openAiModel="gpt-4.1-nano" # Cheapest ~$0.04
    openAiModel="gpt-5"
    #openAiModel="gpt-4.1-mini"  # In between ~$0.13

    # We're going to process 50 strings at a time. There's a small cost hit for
    # this, as the glossary and system instructions cost us on each call.
    target_language = "Polish"
    strings_at_a_time = 50
    start = 0
    end = strings_at_a_time
    accumulated_results = {}
    timeout_count = 0
    while start < end:
        try:
            results = translateText(target_language, glossary, payload[start:end], openAiModel)
            if results != None:
                # Transform results into simple JSON format that can be accepted by Weblate
                for resource in results["results"]:
                    accumulated_results[resource["key"]] = resource["translation"]

                start = start + len(results["results"])
            else:
                start = start + strings_at_a_time

            end = start + strings_at_a_time
            if end > len(payload):
                end = len(payload)
        except:
            timeout_count = timeout_count + 1
            time.sleep(10)
            # Try a smaller number of strings
            strings_at_a_time = strings_at_a_time = 25
            end = start + strings_at_a_time
        
        if timeout_count > 5:
            break

    out_path = Path(f"{target_language}-{openAiModel}-translations.json")
    out_path.write_text(json.dumps(accumulated_results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
