"""
noun_extractor.py — Extract goal object and contextual nouns from aerial navigation instructions.

Input:  A list containing one instruction string in the format used by the Hal-13k dataset.
Output: (goal_object: str, contextual_nouns: list[str])

Uses spaCy noun chunks when available and falls back to lightweight rule-based
chunking so the pipeline still runs in environments where spaCy is not installed.
"""

import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

SEPARATOR = "The description of the target and its surrounding is shown below."

# Terms that are not segmentable in a BEV image — abstract or too general
BLOCKLIST = {
    "environment", "visibility", "landscape", "area", "location", "description",
    "target", "starting point", "surroundings", "mix", "series", "view",
    "vicinity", "setting", "scene", "place", "spot", "site", "space", "zone",
    "distance", "background", "foreground", "proximity", "presence",
    "way", "side", "look", "appearance", "style", "type", "kind", "sort",
    "variety", "collection", "group", "set", "number", "amount", "level",
    "section", "part", "piece", "portion", "segment", "bit", "aspect",
    "feature", "direction", "height", "size", "shape", "pattern", "texture",
    "sides", "greenery", "vegetation", "terrain", "ground",
    "weather", "condition", "atmosphere", "overcast", "visibility", "mist", "fog",
    "sky", "light", "shadow",
    # Function / grammatical words that can leak through
    "and", "or", "with", "near", "along", "beside", "surrounding",
    # Weather/environmental adjectives used as nouns
    "misty", "overcast", "foggy", "sunny", "cloudy", "rainy", "windy",
    "limited", "moderate", "extensive", "dense", "sparse",
    # Generic placeholder nouns used in dataset descriptions
    "object", "subject", "item", "thing", "entity",
    # Scene wrapper terms
    "image", "photo", "picture", "frame",
    # Body parts / clothing that are not useful BEV segments
    "hair", "head", "face", "hand", "hands", "arm", "arms", "leg", "legs",
    "shirt", "shirts", "pants", "jeans", "dress", "coat", "jacket", "hat",
    "helmet", "shoes", "boots", "clothes", "clothing", "sleeve", "sleeves",
    "top", "bottom", "outfit",
}

# Determiners
_DETS = r'(?:a|an|the|some|this|that|these|those|several|multiple|various|another|other|any|one|two|three|many|few)'

# Common verb forms that signal we have left the NP — used for cleanup
_VERB_RE = re.compile(
    r'\b(?:is|are|was|were|be|been|being|has|have|had|can|could|will|would|shall|should|may|might|'
    r'walks?|walked|walking|runs?|ran|running|stands?|stood|standing|sits?|sat|sitting|'
    r'appears?|appeared|appearing|looks?|looked|looking|moves?|moved|moving|'
    r'features?|featured|featuring|surrounds?|surrounded|surrounding|'
    r'borders?|bordered|bordering|lines?|lined|lining|'
    r'situated|located|found|positioned|placed|'
    r'wears?|wearing|dressed|dressing|tied|tying|'
    r'includes?|included|including|leads?|led|leading|'
    r'seen|visible|overlooking)\b'
)
_SCENE_PREFIX_RE = re.compile(
    r'^(?:in|within)\s+(?:the|this)\s+(?:bird\'?s-eye-view\s+)?(?:image|frame|scene)\s*,?\s*',
    re.IGNORECASE,
)
_TRAILING_STOP_RE = re.compile(
    r'\s+(?:the|a|an|some|this|that|these|those|and|or|with|in|on|near|at|by|'
    r'of|to|for|from|which|who|where|when)$',
    re.IGNORECASE,
)
_GOAL_PATTERN = re.compile(
    r'(?i)^(?:a|an|the|this|that|these|those)\s+(.+?)\s+'
    r'(?:is|are|was|were|stands?|standing|walks?|walking|runs?|running|'
    r'sits?|sitting|lies?|lying|rides?|riding|drives?|driving|'
    r'located|situated|positioned|found|placed|appears?|seems?|'
    r'wears?|wearing|has|have|can\s+be)\b'
)
_GOAL_CONNECTOR_RE = re.compile(
    r'(?i)^(?:a|an|the|this|that|these|those)\s+(.+?)\s+'
    r'(?:with|wearing|near|next to|beside|adjacent to|on|in)\b'
)
# Handles "The [generic] is a/an [actual object] [verb/prep]..."
# e.g. "The object is an umbrella table situated on..."
_GOAL_COPULA_PATTERN = re.compile(
    r'(?i)^(?:the|a|an|this|that)\s+\w+\s+is\s+(?:a|an|the)\s+(.+?)\s+'
    r'(?:situated|located|positioned|found|placed|surrounded|bordered|'
    r'on\b|in\b|at\b|with\b|near\b|featuring|visible|,|\Z)'
)
_ADJECTIVE_ENDINGS = {
    "white", "black", "gray", "grey", "red", "green", "blue", "yellow", "purple",
    "orange", "brown", "pink", "small", "large", "tall", "short", "dense",
    "grassy", "rocky", "urban", "residential", "misty", "foggy", "sunlit",
}

# Specificity rank: lower = more specific = higher priority to keep
_SPECIFICITY_RANK = {
    "motorcycle": 0, "bicycle": 0, "bike": 0, "car": 0, "truck": 0, "bus": 0,
    "vehicle": 1, "dog": 0, "cat": 0, "person": 0, "man": 0, "woman": 0,
    "pedestrian": 0, "cyclist": 0, "bin": 0, "bench": 0, "chair": 0,
    "sign": 0, "lamp": 0, "post": 0, "pole": 0, "gate": 0, "fence": 0,
    "pergola": 0, "swing": 0, "crosswalk": 1, "crossroads": 1,
    "bridge": 1, "road": 2, "street": 2, "sidewalk": 2, "pavement": 2,
    "path": 2, "house": 2, "building": 2, "structure": 2, "wall": 2,
    "roof": 2, "lawn": 2, "garden": 2, "park": 2, "plaza": 2,
    "tree": 2, "bush": 2, "shrub": 2, "plant": 2, "grass": 2,
    "river": 2, "lake": 2, "pond": 2, "stream": 2, "canal": 2,
    "greenery": 3, "vegetation": 3, "terrain": 3, "ground": 3,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _strip_leading_det(phrase: str) -> str:
    """Remove a leading determiner from a phrase."""
    return re.sub(rf'^{_DETS}\s+', '', phrase.strip(), flags=re.IGNORECASE).strip()


def _strip_scene_prefix(text: str) -> str:
    """Remove wrapper text such as 'In the image,' from a description span."""
    return _SCENE_PREFIX_RE.sub("", text.strip()).strip()


def _is_blocked(phrase: str) -> bool:
    """Return True if the phrase should be excluded."""
    p = phrase.strip().lower()
    if not p or len(p) <= 2:
        return True
    if p.endswith("'s"):
        return True
    last = p.split()[-1]
    first = p.split()[0]
    if p in BLOCKLIST or last in BLOCKLIST:
        return True
    if first in {"wearing", "tied", "shown", "seen", "visible"}:
        return True
    # If the phrase contains a verb, it slipped through — discard it
    if _VERB_RE.search(p):
        return True
    if last in _ADJECTIVE_ENDINGS:
        return True
    return False


def _clean_np(raw: str) -> str:
    """Strip determiners, trailing stopwords, and leading 'and/or'."""
    s = _strip_scene_prefix(raw.strip().lower())
    # Leading and/or/with/in/on/near
    s = re.sub(r'^(?:and|or|with|in|on|near|at|by|of|to|for|from)\s+', '', s)
    # Leading determiner
    s = _strip_leading_det(s)
    # Trailing function words
    s = re.sub(r'\s+(?:and|or|with|in|on|near|at|by|of|to|for|from|that|which|who)$', '', s)
    s = _TRAILING_STOP_RE.sub("", s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# ─── Clause-based extraction ──────────────────────────────────────────────────

def _normalise_clause(clause: str) -> str:
    """Strip verb-phrase prefixes from a clause, leaving the bare NP."""
    s = _strip_scene_prefix(clause.strip().lower())
    # Strip passive / participial openers
    s = re.sub(r'^(?:is|are|was|were|be)\s+', '', s)
    for vp in ("situated in", "situated on", "situated at", "situated",
               "located in", "located on", "located at", "located",
               "found in", "found on", "found",
               "positioned on", "positioned",
               "placed on", "placed",
               "surrounded by", "bordered by", "lined with",
               "filled with", "covered with",
               "featuring", "including",
               "characterized by", "consists of", "consisting of",
               "which features", "that features", "that includes",
               "visible", "overlooked by",
               ):
        if s.startswith(vp + " ") or s == vp:
            s = s[len(vp):].strip()
            break
    # Strip any leading conjunction or preposition
    s = re.sub(r'^(?:and|or|but|also)\s+', '', s)
    s = re.sub(r'^(?:on|in|near|beside|alongside|along|at|over|under|through|'
               r'between|within|around|across|by|with|of|to|for|from|'
               r'adjacent to|next to|in front of|behind|above|below)\s+', '', s)
    # Strip determiner
    s = _strip_leading_det(s)
    return s


def _np_from_normalised(s: str) -> Optional[str]:
    """Extract up to 3-word NP from a normalised clause string.

    Stops at prepositions, conjunctions, and verb forms so that
    "bridge with yellow lane markings" yields "bridge".
    """
    _INNER_STOP = re.compile(
        r'^(?:with|on|in|near|at|by|of|to|for|from|along|beside|alongside|'
        r'around|over|under|through|between|within|across|behind|above|below|'
        r'and|or|that|which|who|where|when|while|but|if|because|although|'
        r'including|featuring|surrounding|bordering)$'
    )
    # Strip trailing punctuation from input
    s = s.rstrip('.,;:!?')
    words = s.split()
    np_words = []
    for w in words[:4]:  # inspect first 4 tokens, keep up to 3
        w_clean = w.rstrip('.,;:!?')
        if _INNER_STOP.fullmatch(w_clean) or _VERB_RE.fullmatch(w_clean):
            break
        np_words.append(w_clean)
        if len(np_words) == 3:
            break
    if not np_words:
        return None
    np = " ".join(np_words)
    if _VERB_RE.search(np):
        return None
    return np if len(np) >= 3 else None


def _split_and_extract(text: str) -> list:
    """
    Split description into minimal sub-clauses and extract one NP from each.

    Splitting strategy:
      1. commas and semicolons
      2. " and " conjunctions within each comma-clause
      3. " with " prepositional phrases within each part (to separate
         "bridge with yellow lane markings" into "bridge" and "yellow lane markings")
    """
    # Split on prepositional/participial connectors that introduce a new NP
    _WITH_SPLIT = re.compile(
        r'\s+(?:with|featuring|including|near|beside|alongside|along|next to|adjacent to|'
        r'surrounded by|bordered by|lined with)\s+'
    )
    _AND_SPLIT  = re.compile(r'\s+and\s+')

    chunks = []

    # Step 1: split on , ; . :
    for comma_clause in re.split(r'[,:;.]', text):
        comma_clause = comma_clause.strip()
        if not comma_clause:
            continue

        # Step 2: split on " and "
        for and_part in _AND_SPLIT.split(comma_clause):
            and_part = and_part.strip()
            if not and_part:
                continue

            # Step 3: split on " with " to separate prepositional modifiers
            with_parts = _WITH_SPLIT.split(and_part)
            for part in with_parts:
                part = part.strip()
                if not part:
                    continue
                normalised = _normalise_clause(part)
                np = _np_from_normalised(normalised)
                if np:
                    chunks.append(np)

    return chunks


# ─── Goal extraction ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_spacy():
    """Load a small spaCy model when available, otherwise return None."""
    try:
        import spacy  # type: ignore
    except ImportError:
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name, disable=["ner", "textcat"])
        except OSError:
            continue
    return None


def _extract_spacy_chunks(description: str) -> list[str]:
    """Extract noun chunks via spaCy when the dependency is available."""
    nlp = _load_spacy()
    if nlp is None:
        return []

    doc = nlp(description)
    chunks = []
    for chunk in doc.noun_chunks:
        cleaned = _clean_np(chunk.text)
        if cleaned:
            chunks.append(cleaned)
    return chunks


def _trim_goal_candidate(raw: str) -> str:
    """Trim descriptive tails from a goal-object candidate phrase."""
    s = raw.strip().lower()
    s = re.split(
        r'\b(?:with|wearing|near|next to|beside|adjacent to|surrounded by|on|in)\b',
        s,
        maxsplit=1,
    )[0]
    return _clean_np(s)


def _extract_goal(description: str) -> Optional[str]:
    """
    Extract the goal object — the first concrete noun phrase in the description.
    """
    text = _strip_scene_prefix(description.strip())

    match = _GOAL_PATTERN.match(text)
    if match:
        cleaned = _trim_goal_candidate(match.group(1))
        if cleaned and not _is_blocked(cleaned):
            return cleaned

    match = _GOAL_CONNECTOR_RE.match(text)
    if match:
        cleaned = _trim_goal_candidate(match.group(1))
        if cleaned and not _is_blocked(cleaned):
            return cleaned

    match = _GOAL_COPULA_PATTERN.match(text)
    if match:
        cleaned = _trim_goal_candidate(match.group(1))
        if cleaned and not _is_blocked(cleaned):
            return cleaned

    for candidate in _extract_spacy_chunks(text) + _split_and_extract(text):
        cleaned = _clean_np(candidate)
        if cleaned and not _is_blocked(cleaned):
            return cleaned

    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_nouns_from_description(description: str, max_contextual: int = 7):
    """
    Extract goal object and contextual nouns from one description sentence.

    Returns
    -------
    goal_object : str or None
    contextual_nouns : list[str]  (deduplicated, filtered, ranked by specificity)
    """
    goal = _extract_goal(description)

    raw_chunks = _extract_spacy_chunks(description)
    raw_chunks.extend(_split_and_extract(description))

    # Clean, deduplicate, filter
    seen: set[str] = set()
    contextual = []
    for raw in raw_chunks:
        chunk = _clean_np(raw)
        if not chunk:
            continue
        if _is_blocked(chunk):
            continue
        if goal and (chunk == goal or chunk in goal or goal in chunk):
            continue
        if chunk in seen:
            continue
        # Skip if already subsumed by a longer accepted phrase
        if any(chunk in prev for prev in seen):
            continue
        seen.add(chunk)
        contextual.append(chunk)

    # Rank by specificity (lower = better)
    def _rank(phrase: str) -> tuple:
        last = phrase.split()[-1]
        base = _SPECIFICITY_RANK.get(last, 2)
        if len(phrase.split()) == 1 and base >= 3:
            base += 1
        return (base, -len(phrase))

    contextual.sort(key=_rank)
    contextual = contextual[:max_contextual]

    return goal, contextual


def parse_instruction(instruction_json) -> tuple:
    """
    Parse an instruction file or value.

    Parameters
    ----------
    instruction_json : list, str (JSON), or Path
        The raw data (list with one string), a JSON-encoded string, or a file path.

    Returns
    -------
    goal_object : str or None
    contextual_nouns : list[str]
    description : str
    """
    if isinstance(instruction_json, Path):
        with open(instruction_json) as f:
            data = json.load(f)
    elif isinstance(instruction_json, list):
        data = instruction_json
    elif isinstance(instruction_json, str):
        if not instruction_json.strip():
            return None, [], ""
        # Try as file path only if short enough, otherwise treat as plain text
        if len(instruction_json) < 400:
            try:
                p = Path(instruction_json)
                if p.exists():
                    with open(p) as f:
                        data = json.load(f)
                else:
                    data = json.loads(instruction_json)
            except (OSError, ValueError):
                data = instruction_json  # plain description string
        else:
            try:
                data = json.loads(instruction_json)
            except ValueError:
                data = instruction_json  # plain description string
    else:
        raise TypeError(f"Unsupported type: {type(instruction_json)}")

    raw = data[0] if isinstance(data, list) else data

    if SEPARATOR in raw:
        description = raw.split(SEPARATOR, 1)[1].strip()
    else:
        description = raw.strip()

    goal, contextual = extract_nouns_from_description(description)
    return goal, contextual, description


def build_prompt_list(goal: Optional[str], contextual: list) -> list:
    """
    Build the ordered GroundingDINO text prompt list.
    Goal object is always first; contextual nouns follow.
    """
    prompts = []
    if goal:
        prompts.append(goal)
    for noun in contextual:
        if noun != goal:
            prompts.append(noun)
    return prompts


# ─── CLI / quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    samples = [
        ('["Compass north corresponds to the top of the bird\'s-eye-view image. The target location is 47.34 degrees north by east from the starting point. The description of the target and its surrounding is shown below. The motorcycle is purple, located on a bridge with yellow lane markings, and is surrounded by a landscape featuring a river, rocky formations, trees, and light posts; the environment appears to be overcast and misty with limited visibility."]',
         {"goal": "motorcycle", "contextual": ["bridge", "yellow lane markings", "river", "rocky formations", "trees", "light posts"]}),
        ('["Compass north corresponds to the top of the bird\'s-eye-view image. The target location is 26.28 degrees south by east from the starting point. The description of the target and its surrounding is shown below.The man is walking on a road near some bushes and tall evergreen trees, situated in an area that features urban buildings with a visible street lamp, surrounded by a paved street, curved road, and greenery along the sides."]',
         {"goal": "man"}),
        ('["Compass north corresponds to the top of the bird\'s-eye-view image. The target location is 0.93 degrees north by west from the starting point. The description of the target and its surrounding is shown below.The green bin is situated on a grassy lawn in front of a brick house with a tiled roof and windows, bordered by a paved roadway lined with residential structures, trees, and a play area with red swings."]',
         {"goal": "green bin"}),
    ]

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        goal, contextual, desc = parse_instruction(path)
        print(f"Description : {desc}")
        print(f"Goal object : {goal!r}")
        print(f"Contextual  : {contextual}")
        print(f"Prompt list : {build_prompt_list(goal, contextual)}")
    else:
        all_pass = True
        for i, (s, expected) in enumerate(samples, 1):
            goal, contextual, desc = parse_instruction(s)
            prompts = build_prompt_list(goal, contextual)
            ok_goal = goal == expected.get("goal", goal)
            print(f"\n{'='*60}")
            print(f"Sample {i}")
            print(f"Description : {desc[:100]}...")
            print(f"Goal object : {goal!r}  {'[OK]' if ok_goal else '[FAIL]'}")
            print(f"Contextual  : {contextual}")
            print(f"Prompt list : {prompts}")
            if not ok_goal:
                all_pass = False
        print(f"\n{'='*60}")
        print("All goal checks passed!" if all_pass else "Some checks FAILED — review output above.")
