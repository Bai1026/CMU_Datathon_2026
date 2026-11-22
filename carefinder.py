"""Care Finder: which facility offers a service near a place, and why trust the answer.

Pure pandas, no Spark, so the same code runs in a Databricks notebook, a Databricks App
and on a laptop. Everything that is a judgment call lives in three YAML files next to
this module, not in the code:

  scoring.yaml            distances, penalties, grade values
  services.yaml           what counts as evidence for each service
  location_aliases.yaml   other names for places

Stages:
  prepare_facilities   clean each record, resolve its location, attach trust flags
  build_evidence       for every facility x service, collect the exact entries that
                       support it and grade them
  Gazetteer.resolve    "Jaipur" / "302001" / "26.9, 75.8" -> coordinates
  recommend            score_candidates -> pick -> verdict
  render_html          the answer, from the template answer.html.j2
"""
import difflib
import json
import math
import os
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Filled by load_settings() from the YAML files. Kept as module-level dicts so that
# every function sees the same settings; call load_settings() again after editing a file.
CONFIG = {}
SERVICES = {}
SUPPORT = {}
ALIASES = {"states": [], "places": [], "other_scripts": {}, "filler_words": []}
NEGATION = None

TIER_LABEL = {
    "strong": "Corroborated",
    "moderate": "Claimed, not corroborated",
    "weak": "Implied only",
}
TEXT_FIELDS = ["procedure", "capability", "equipment"]
GENERATED_FILE = "services_generated.yaml"


# ----------------------------------------------------------------------------------
# Term matching: how the lists in services.yaml are turned into patterns
# ----------------------------------------------------------------------------------
def term_pattern(term):
    """One YAML term -> regex source.

    'dialy'       starts a word:  dialysis, dialyzer
    '=CT'         whole word only
    'open heart'  spaces and hyphens are interchangeable and optional
    """
    term = str(term).strip()
    whole = term.startswith("=")
    tokens = [t for t in re.split(r"[\s-]+", term.lstrip("=").strip()) if t]
    if not tokens:
        return None
    body = r"[\s-]*".join(re.escape(t) for t in tokens)
    start = r"\b" if re.match(r"\w", tokens[0]) else ""
    end = r"\b" if whole and re.search(r"\w$", tokens[-1]) else ""
    return start + body + end


def compile_terms(terms):
    """List of YAML terms -> one compiled, case-insensitive pattern (None if the list is empty)."""
    parts = [x for x in (term_pattern(t) for t in (terms or [])) if x]
    return re.compile("|".join(parts), re.I) if parts else None


def compile_support(spec):
    """A supports: entry -> {'fn': text -> bool, 'specialty': list of code fragments}."""
    any_re, none_re = compile_terms(spec.get("any_of")), compile_terms(spec.get("none_of"))
    all_res = [compile_terms(group) for group in spec.get("all_of", [])]

    def fn(text):
        if any_re is not None and not any_re.search(text):
            return False
        if any(r is None or not r.search(text) for r in all_res):
            return False
        return not (none_re is not None and none_re.search(text))

    return {"fn": fn, "specialty": [str(x).lower() for x in spec.get("specialty_any", [])]}


def compile_service(svc):
    """Add the compiled patterns a service definition needs. Returns the same dict."""
    for k in ("direct", "equipment", "related", "exclude"):
        svc[k] = [str(t) for t in (svc.get(k) or [])]
        svc[k + "_re"] = compile_terms(svc[k])
    if svc["direct_re"] is None:
        raise ValueError(f"Service '{svc.get('label')}' has no direct terms.")
    svc["any_re"] = compile_terms(svc["direct"] + svc["equipment"] + svc["related"])
    svc["specialties"] = [str(x).lower() for x in (svc.get("specialties") or [])]
    svc["supports"] = list(svc.get("supports") or [])
    svc.setdefault("urgency", "urgent")
    return svc


def load_settings(directory=None):
    """(Re)load scoring.yaml, services.yaml and location_aliases.yaml into the module.

    Also loads services_generated.yaml if it exists: definitions a language model wrote
    for services that are not in services.yaml. Returns a short summary of what was read.
    """
    global NEGATION
    d = directory or BASE_DIR

    def read(name, required=True):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"{name} not found in {d}. It must sit next to carefinder.py.")
            return {}
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    CONFIG.clear(); CONFIG.update(read("scoring.yaml"))
    sv = read("services.yaml")
    NEGATION = compile_terms(sv.get("negation"))
    SUPPORT.clear(); SUPPORT.update({k: compile_support(v) for k, v in (sv.get("supports") or {}).items()})
    SERVICES.clear()
    for key, svc in (sv.get("services") or {}).items():
        SERVICES[key] = compile_service(dict(svc))
    for key, svc in (read(GENERATED_FILE, required=False).get("services") or {}).items():
        if key not in SERVICES:
            svc = dict(svc); svc.setdefault("generated", {"request": svc.get("label", key)})
            SERVICES[key] = compile_service(svc)
    for key, svc in SERVICES.items():
        unknown = [x for x in svc["supports"] if x not in SUPPORT]
        if unknown:
            raise ValueError(f"Service '{key}' uses supports that are not defined: {unknown}")
        if svc["urgency"] not in CONFIG["half_km"]:
            raise ValueError(f"Service '{key}' has urgency '{svc['urgency']}'; use one of {list(CONFIG['half_km'])}")
    al = read("location_aliases.yaml", required=False)
    ALIASES.clear(); ALIASES.update({"states": al.get("states") or [], "places": al.get("places") or [],
                                     "other_scripts": al.get("other_scripts") or {},
                                     "filler_words": [w for g in al.get("filler_words") or []
                                                      for w in (g if isinstance(g, list) else [g])]})
    return {"directory": d, "services": list(SERVICES), "supports": list(SUPPORT),
            "alias_groups": len(ALIASES["places"]), "generated_services": [k for k, v in SERVICES.items() if v.get("generated")]}


def service_to_yaml_dict(svc):
    """The reviewable part of a service definition (no compiled patterns)."""
    keep = ("label", "urgency", "hospital_level", "needs_247", "equipment_is_direct", "small_staff_note",
            "target_minutes", "target_note",
            "direct", "equipment", "related", "exclude", "specialties", "supports", "min_support", "generated")
    return {k: svc[k] for k in keep if k in svc and svc[k] not in (None, [], {})}


def save_generated_services(directory=None):
    """Write every model-written service definition to services_generated.yaml for review."""
    gen = {k: service_to_yaml_dict(v) for k, v in SERVICES.items() if v.get("generated")}
    path = os.path.join(directory or BASE_DIR, GENERATED_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Service definitions written by a language model. NOT clinically reviewed.\n"
                 "# Review an entry, then move it into services.yaml to make it permanent.\n")
        yaml.safe_dump({"services": gen}, fh, sort_keys=False, allow_unicode=True)
    return path


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------
def is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in ("", "null", "none", "nan")


def parse_list(v):
    """A JSON-array string (or list) -> list of non-empty strings."""
    if isinstance(v, (list, tuple, np.ndarray)):
        items = list(v)
    elif is_missing(v):
        return []
    else:
        try:
            items = json.loads(v)
        except (TypeError, ValueError):
            items = [v]
        if not isinstance(items, list):
            items = [items]
    return [str(x).strip() for x in items if not is_missing(x) and str(x).strip()]


def _fold(s):
    """Lower-case; digits of any script become 0-9; Latin letters lose their accents (é -> e).
    Letters of other scripts are kept as they are."""
    s = str(s)
    if s.isascii():
        return s.lower()
    out = []
    for c in unicodedata.normalize("NFKD", s):
        if "\u0300" <= c <= "\u036f" or unicodedata.category(c) == "Cf":      # accents; zero-width joiners
            continue
        d = unicodedata.decimal(c, None)
        out.append(str(d) if d is not None else c)
    return unicodedata.normalize("NFC", "".join(out)).lower()


def _norm(s):
    """The comparison key for names and entries: folded, punctuation replaced by single spaces."""
    s = _fold(s)
    if s.isascii():
        return re.sub(r"[^a-z0-9]+", " ", s).strip()
    return " ".join("".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in s).split())


# ---- text that is not in Latin letters -------------------------------------------------
# Nine Indian scripts share one Unicode layout (each block repeats the same order of letters),
# so one table of readings, applied at each block's starting point, covers all of them.
INDIC_BLOCKS = {"Devanagari": 0x0900, "Bengali": 0x0980, "Gurmukhi": 0x0A00, "Gujarati": 0x0A80, "Odia": 0x0B00,
                "Tamil": 0x0B80, "Telugu": 0x0C00, "Kannada": 0x0C80, "Malayalam": 0x0D00}
_OTHER_SCRIPTS = [("Arabic", 0x0600, 0x06FF), ("Arabic", 0x0750, 0x077F), ("Arabic", 0xFB50, 0xFDFF), ("Arabic", 0xFE70, 0xFEFF),
                  ("Chinese or Japanese", 0x3040, 0x30FF), ("Chinese or Japanese", 0x3400, 0x9FFF),
                  ("Korean", 0xAC00, 0xD7AF), ("Cyrillic", 0x0400, 0x04FF), ("Thai", 0x0E00, 0x0E7F), ("Sinhala", 0x0D80, 0x0DFF)]
# Scripts of languages that usually drop the unwritten "a" at the end of a word (Patna is written pa-ta-naa).
_DROPS_FINAL_A = {"Devanagari", "Bengali", "Gurmukhi", "Gujarati"}
_CONSONANT = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
              0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
              0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
              0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
              0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWEL = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e",
          0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_VOWEL_SIGN = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e",
               0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_NUKTA = {0x1C: "z", 0x2B: "f", 0x21: "r", 0x22: "rh", 0x15: "q", 0x2F: "y"}      # letter + dot below
_FINAL_LETTER = {0x09CE: "t", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}
_SCHWA = "~"      # the unwritten "a" after a consonant; sometimes silent
_NASAL = "\x00"   # a nasal mark, turned into n, m or ng once the next letter is known
_TAMIL_VOICED = {"k": "g", "t": "d", "p": "b"}
_TAMIL_AFTER_NASAL = {"ch": "j"}
_EXTRA_LETTER = {("Odia", 0x71): "w", ("Bengali", 0x70): "r", ("Bengali", 0x71): "w"}      # letters outside the shared layout


def script_of(ch):
    """Name of the script a character belongs to, 'Latin' for a-z, or None for digits and punctuation."""
    o = ord(ch)
    if o < 0x250:
        return "Latin" if ch.isalpha() else None
    for name, base in INDIC_BLOCKS.items():
        if base <= o < base + 0x80:
            return name
    for name, lo, hi in _OTHER_SCRIPTS:
        if lo <= o <= hi:
            return name
    return "another script" if unicodedata.category(ch)[0] == "L" else None


def scripts_in(text):
    """The scripts used in a text, most frequent first, Latin excluded."""
    n = Counter(sc for sc in map(script_of, str(text)) if sc and sc != "Latin")
    return [k for k, _ in n.most_common()]


def _read_word(word):
    syl, script = [], None      # syllables as [consonant, vowel, letter]; vowel None = no vowel follows
    for ch in word:
        o = ord(ch)
        base = next((b for b in INDIC_BLOCKS.values() if b <= o < b + 0x80), None)
        if base is None:
            if ch.isascii() and ch.isalnum():
                syl.append([ch, None, None])
            continue
        script = script or next(k for k, b in INDIC_BLOCKS.items() if b == base)
        k = o - base
        if o in _FINAL_LETTER:
            syl.append([_FINAL_LETTER[o], None, None])
        elif (script, k) in _EXTRA_LETTER:
            syl.append([_EXTRA_LETTER[(script, k)], _SCHWA, None])
        elif k in _CONSONANT:
            c = "j" if (script == "Bengali" and k == 0x2F) else _CONSONANT[k]      # Bengali reads this letter as j
            syl.append([c, _SCHWA, k])
        elif k in _VOWEL:
            syl.append(["", _VOWEL[k], None])
        elif k in _VOWEL_SIGN and syl:
            syl[-1][1] = _VOWEL_SIGN[k]
        elif k == 0x4D and syl:                           # the sign that cancels the unwritten "a"
            syl[-1][1] = None
        elif k == 0x3C and syl and syl[-1][2] in _NUKTA:  # dot below: a different consonant
            syl[-1][0] = _NUKTA[syl[-1][2]]
        elif script == "Bengali" and k == 0x02:
            syl.append(["ng", None, None])
        elif k in (0x01, 0x02) or (script == "Gurmukhi" and k == 0x70):      # nasal marks: the sound depends on what follows
            syl.append([_NASAL, None, None])
    last = len(syl) - 1
    if script in _DROPS_FINAL_A:
        for i in range(last, 0, -1):
            if syl[i][1] != _SCHWA:
                continue
            before = syl[i - 1][1] is not None or syl[i - 1][0] == _NASAL
            if i == last:
                if before:                                # "...ra" after a vowel loses its a; after a cluster it keeps it
                    syl[i][1] = None
            elif before and syl[i + 1][0] and syl[i + 1][1] is not None:
                syl[i][1] = None                          # between two sounded syllables: pa-ta-naa -> patna
    for i in range(1, len(syl)):
        c, v, k = syl[i]
        p_c, p_v = syl[i - 1][0], syl[i - 1][1]
        nasal = p_v is None and p_c in ("n", "m", _NASAL)
        if script == "Tamil":           # one Tamil letter serves for k/g, t/d and p/b: voiced after a nasal or between vowels
            if c in _TAMIL_VOICED and v is not None and (p_v is not None or nasal):
                syl[i][0] = _TAMIL_VOICED[c]
            elif c in _TAMIL_AFTER_NASAL and nasal:
                syl[i][0] = _TAMIL_AFTER_NASAL[c]
        elif script == "Malayalam" and k == 0x1F and p_v is not None and (v is not None or i == last):
            syl[i][0] = "d"             # a single hard t after a vowel is written d in English: Palakkad, Idukki
    for i, (c, v, _) in enumerate(syl):
        if c == _NASAL:
            nxt = syl[i + 1][0] if i < last else None
            syl[i][0] = "m" if (nxt is None or nxt[:1] in ("p", "b", "m")) else "n"
    return "".join(c + ("a" if v == _SCHWA else (v or "")) for c, v, _ in syl)


def romanize(text):
    """A rule-based reading, in Latin letters, of text written in an Indian script ('पटना' -> 'patna').

    It follows the letters, so it gives the local pronunciation, not the English spelling
    ('लखनऊ' reads 'lakhnau', not 'Lucknow'). Text already in Latin letters is returned unchanged.
    """
    return " ".join(w for w in (_read_word(w) for w in _norm(text).split()) if w)


def _loose(name):
    """A spelling with the commonest English/local differences evened out but the vowels kept
    (w/v, ee/i, oo/u, ph/f, z/j), for ranking the places that share a sound key."""
    s = re.sub(r"[^a-z]", "", _fold(name))
    for a, b in (("ph", "f"), ("w", "v"), ("z", "j"), ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ck", "k"), ("y", "i")):
        s = s.replace(a, b)
    return s


def sound_key(name):
    """A loose key for comparing a reading with an English spelling: consonants only, with spelling
    habits evened out (ph/f, w/v, z/j, ck/k, aspirated and doubled letters). 'Muzaffarpur' and the
    reading 'mujaphpharpur' share the key 'mjfrpr'. Deliberately lossy: use it to find candidates,
    then compare the full spellings."""
    s = re.sub(r"[^a-z]", "", _fold(name))
    for a, b in (("chh", "C"), ("ch", "C"), ("sh", "s"), ("zh", "Z"), ("ph", "f"), ("ck", "k"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("rh", "r"), ("w", "v"), ("z", "j"), ("q", "k"),
                 ("x", "ks"), ("c", "k")):
        s = s.replace(a, b)
    s = s.replace("gaon", "gav").replace("ganv", "gav")      # the ending of Jalgaon, Gurgaon: written -gav / -ganv locally
    s = s[:1] + s[1:].replace("y", "")              # y is a consonant only at the start (Yadgir); elsewhere it is ai/ay
    s = re.sub(r"n(?=[pbm])", "m", s)
    s = re.sub(r"(.)\1+", r"\1", s)                 # doubled letters, before the vowels go: Jajpur must not become Jaipur
    return re.sub(r"[aeiou]", "", s).lower()


def dedupe(items):
    """Drop repeats (case and punctuation insensitive), keep first spelling and order."""
    seen, out = set(), []
    for x in items:
        k = _norm(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def domain_of(url):
    """Host name of a URL, or None when the value is missing or is not a URL.

    Some records carry other text in URL fields (for example an equipment list in
    officialWebsite), so anything that does not look like a host is rejected.
    """
    if is_missing(url):
        return None
    u = str(url).strip()
    try:
        host = urlparse(u if "//" in u else "http://" + u).netloc.lower()
    except ValueError:
        return None
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", host) else None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance. Works on scalars or numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def in_india(lat, lon):
    a, b, c, d = CONFIG["india_bbox"]
    return lat is not None and lon is not None and not math.isnan(lat) and not math.isnan(lon) and a <= lat <= b and c <= lon <= d


def extract_pin(v):
    if is_missing(v):
        return None
    m = re.search(r"(?<!\d)[1-9]\d{5}(?!\d)", re.sub(r"(?<=\d)\s+(?=\d)", "", str(v)))
    return m.group(0) if m else None


load_settings()


# ----------------------------------------------------------------------------------
# PIN code directory
# ----------------------------------------------------------------------------------
_PIN_COLS = {
    "pincode": ["pincode", "pin", "pin_code", "postalcode", "postal_code"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long", "lng"],
    "district": ["district", "districtname", "district_name"],
    "state": ["statename", "state", "state_name", "stateorregion"],
    "office": ["officename", "office_name", "office", "postoffice", "post_office"],
}


def standardize_pins(pins_raw):
    """Rename the directory's columns to pincode/lat/lon/district/state/office.

    Column names differ between releases of the India Post directory, so they are
    matched case-insensitively. Raises with the actual column list if pincode is absent.
    """
    lower = {re.sub(r"[^a-z]", "", c.lower()): c for c in pins_raw.columns}
    out = pd.DataFrame(index=pins_raw.index)
    for std, cands in _PIN_COLS.items():
        hit = next((lower[re.sub(r"[^a-z]", "", c)] for c in cands if re.sub(r"[^a-z]", "", c) in lower), None)
        out[std] = pins_raw[hit] if hit else np.nan
    if out["pincode"].isna().all():
        raise ValueError(f"No PIN code column found. Columns are: {list(pins_raw.columns)}")
    out["pincode"] = out["pincode"].map(extract_pin)
    for c in ("lat", "lon"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    a, b, c_, d = CONFIG["india_bbox"]
    bad = ~(out["lat"].between(a, b) & out["lon"].between(c_, d))
    out.loc[bad, ["lat", "lon"]] = np.nan
    for c in ("district", "state", "office"):
        out[c] = out[c].where(~out[c].map(is_missing), None)
        out[c] = out[c].map(lambda s: None if s is None else str(s).strip().title())
    return out.dropna(subset=["pincode"]).reset_index(drop=True)


def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def summarize_pins(pins):
    """One row per PIN: centroid, district, state, and the place names it covers."""
    g = pins.groupby("pincode")
    out = g.agg(pin_lat=("lat", "median"), pin_lon=("lon", "median"),
                pin_district=("district", _mode), pin_state=("state", _mode))
    names = defaultdict(set)
    for p, office, district in zip(pins["pincode"], pins["office"], pins["district"]):
        for x in (office, district):
            if not is_missing(x):
                names[p].add(_norm(x))
    out["pin_names"] = [sorted(names.get(p, ())) for p in out.index]
    return out


def prefix_regions(pins):
    """One row per 3-digit PIN prefix: centre and radius of the area it covers.

    The directory has wrong coordinates for some post offices. A region median over
    hundreds of offices is robust to that, so it is used to sanity-check both the
    single-PIN centroids and the facilities' own coordinates.
    """
    ok = pins.dropna(subset=["lat", "lon"]).copy()
    ok["prefix"] = ok["pincode"].str[:3]
    reg = ok.groupby("prefix").agg(pref_lat=("lat", "median"), pref_lon=("lon", "median"), pref_n=("lat", "size"))
    ok = ok.join(reg, on="prefix")
    ok["d"] = haversine_km(ok["lat"], ok["lon"], ok["pref_lat"], ok["pref_lon"])
    # Median, not a high percentile: a minority of wrong coordinates cannot inflate it, and it
    # grows by itself where post offices really are far apart.
    reg["pref_radius"] = (CONFIG["region_median_factor"] * ok.groupby("prefix")["d"].median()).clip(
        lower=CONFIG["region_min_km"], upper=CONFIG["region_max_km"]) + CONFIG["region_margin_km"]
    reg.loc[reg["pref_n"] < 5, "pref_radius"] = np.nan
    return reg


# ----------------------------------------------------------------------------------
# Stage 1: clean facilities and attach trust flags
# ----------------------------------------------------------------------------------
_SOCIAL = ("facebook.", "instagram.", "x.com", "twitter.", "linkedin.", "youtube.")
# Filled by prepare_facilities: what it found about duplicate rows and ids.
PREP_STATS = {}


def prepare_facilities(raw, pins=None):
    """Return one row per facility with parsed lists, a resolved location and trust flags.

    pins: output of standardize_pins(), or None. Without it, PIN checks are skipped
    and every location relies on the record's own coordinates.
    """
    f = raw.copy()
    # unique_id is not unique in the source table. Drop rows that are exact copies, then
    # give the remaining repeats their own key so nothing downstream mixes two records.
    try:
        exact = f.duplicated(keep="first")
    except TypeError:
        exact = f.astype(str).duplicated(keep="first")
    f = f[~exact].reset_index(drop=True)
    f["source_unique_id"] = f["unique_id"]
    nth = f.groupby("unique_id").cumcount()
    f["id_repeated"] = f.groupby("unique_id")["unique_id"].transform("size") > 1
    f["unique_id"] = [u if n == 0 else f"{u}#{n + 1}" for u, n in zip(f["unique_id"].astype(str), nth)]
    PREP_STATS.update(rows_in=int(len(raw)), exact_duplicate_rows_dropped=int(exact.sum()),
                      rows_sharing_an_id_with_different_content=int(f["id_repeated"].sum()))
    for c in TEXT_FIELDS + ["specialties"]:
        f[c + "_list"] = f[c].map(lambda v: dedupe(parse_list(v)))
    for c in ("numberDoctors", "capacity", "yearEstablished"):
        f[c + "_num"] = pd.to_numeric(f[c].where(~f[c].map(is_missing)), errors="coerce")
    f["description"] = f["description"].map(lambda v: "" if is_missing(v) else str(v))
    # A JSON array sitting in the website field means this row's columns are shifted.
    f["malformed_record"] = f["officialWebsite"].map(lambda v: not is_missing(v) and str(v).lstrip().startswith(("[", "{")))
    f["lat_raw"] = pd.to_numeric(f["latitude"], errors="coerce")
    f["lon_raw"] = pd.to_numeric(f["longitude"], errors="coerce")
    f["coord_in_india"] = [in_india(a, b) for a, b in zip(f["lat_raw"], f["lon_raw"])]
    f["pin"] = f["address_zipOrPostcode"].map(extract_pin)

    f["pin_prefix"] = f["pin"].str[:3]
    if pins is not None and len(pins):
        ps, reg = summarize_pins(pins), prefix_regions(pins)
        # Drop single-PIN centroids that fall outside their own region: directory errors.
        chk = ps.join(reg, on=ps.index.str[:3])
        far = haversine_km(chk["pin_lat"], chk["pin_lon"], chk["pref_lat"], chk["pref_lon"]) > chk["pref_radius"]
        ps.loc[far.fillna(False).values, ["pin_lat", "pin_lon"]] = np.nan
        f = f.join(ps, on="pin").join(reg[["pref_lat", "pref_lon", "pref_radius"]], on="pin_prefix")
    else:
        for c in ("pin_lat", "pin_lon", "pin_district", "pin_state", "pin_names", "pref_lat", "pref_lon", "pref_radius"):
            f[c] = None
    for c in ("pin_lat", "pin_lon", "pref_lat", "pref_lon", "pref_radius"):
        f[c] = f[c].astype(float)
    f["pin_known"] = f["pin_lat"].notna() & f["pin_lon"].notna()
    f["coord_pin_km"] = np.where(
        f["coord_in_india"] & f["pin_known"],
        haversine_km(f["lat_raw"], f["lon_raw"], f["pin_lat"], f["pin_lon"]), np.nan)
    f["coord_region_km"] = np.where(
        f["coord_in_india"] & f["pref_radius"].notna(),
        haversine_km(f["lat_raw"], f["lon_raw"], f["pref_lat"], f["pref_lon"]), np.nan)

    # Merge signals and source quality.
    f["n_phones"] = f["phone_numbers"].map(lambda v: len({re.sub(r"\D", "", p)[-10:] for p in parse_list(v)}))
    f["n_source_ids"] = f["source_ids"].map(lambda v: len(set(parse_list(v))))
    f["source_url_list"] = f["source_urls"].map(lambda v: dedupe(parse_list(v)))
    f["official_domain"] = f["officialWebsite"].map(domain_of)

    def first_party(row):
        od = row["official_domain"]
        if is_missing(od) or od.startswith(_SOCIAL):
            return False
        return any(domain_of(u) == od for u in row["source_url_list"])

    f["has_first_party_source"] = f.apply(first_party, axis=1)
    phone_cut = max(CONFIG["merged_min_phones"], float(f["n_phones"].quantile(CONFIG["merged_phone_quantile"])))
    f["possible_merged"] = f["n_phones"] >= phone_cut
    sizes = f["cluster_id"].where(~f["cluster_id"].map(is_missing)).value_counts()
    f["cluster_size"] = f["cluster_id"].map(sizes).fillna(1).astype(int)

    # Resolve one location per facility and record how it was chosen.
    lat, lon, src, conflict = [], [], [], []
    for r in f.itertuples(index=False):
        near_pin = r.coord_in_india and r.pin_known and r.coord_pin_km <= CONFIG["coord_pin_max_km"]
        in_region = r.coord_in_india and pd.notna(r.coord_region_km) and r.coord_region_km <= r.pref_radius
        can_check = r.pin_known or pd.notna(r.pref_radius)
        if near_pin or in_region:
            lat.append(r.lat_raw); lon.append(r.lon_raw); conflict.append(False)
            src.append("coordinates, consistent with PIN" if near_pin else "coordinates, consistent with PIN region")
        elif r.coord_in_india and not can_check:
            lat.append(r.lat_raw); lon.append(r.lon_raw); src.append("coordinates, no PIN to cross-check"); conflict.append(False)
        elif r.coord_in_india:
            # Coordinates disagree with both the PIN and its region. Use the PIN only if the city agrees with it.
            city = _norm(r.address_city) if not is_missing(r.address_city) else ""
            names = r.pin_names if isinstance(r.pin_names, (list, tuple)) else []
            if r.pin_known and city and any(city in n or n in city for n in names):
                lat.append(r.pin_lat); lon.append(r.pin_lon); src.append("PIN centroid (coordinates disagree with PIN and city)")
            else:
                lat.append(r.lat_raw); lon.append(r.lon_raw); src.append("coordinates (PIN disagrees; city does not match PIN)")
            conflict.append(True)
        elif r.pin_known:
            lat.append(r.pin_lat); lon.append(r.pin_lon); src.append("PIN centroid (coordinates missing or outside India)"); conflict.append(False)
        else:
            lat.append(np.nan); lon.append(np.nan); src.append("none"); conflict.append(False)
    f["lat"], f["lon"], f["loc_source"], f["location_conflict"] = lat, lon, src, conflict
    f["lat"] = f["lat"].astype(float); f["lon"] = f["lon"].astype(float)
    f["loc_approximate"] = f["loc_source"].str.startswith("PIN centroid")
    f["coords_outside_india"] = f["lat_raw"].notna() & ~f["coord_in_india"]
    f["district"] = f["pin_district"]
    f["state"] = f["pin_state"].where(f["pin_state"].notna(), f["address_stateOrRegion"].where(~f["address_stateOrRegion"].map(is_missing)))

    pen = CONFIG["penalties"]
    reasons, trust = [], []
    for r in f.itertuples(index=False):
        rs = []
        if r.location_conflict:
            km, what = (r.coord_pin_km, f"PIN {r.pin}") if pd.notna(r.coord_pin_km) else (r.coord_region_km, f"PIN region {r.pin_prefix}")
            rs.append(("location_conflict", f"Coordinates are {km:.0f} km from the centre of {what}."))
        if r.coords_outside_india:
            rs.append(("coords_outside_india", f"Recorded coordinates ({r.lat_raw:.2f}, {r.lon_raw:.2f}) are outside India."))
        if r.possible_merged:
            rs.append(("possible_merged_record", f"Record lists {r.n_phones} different phone numbers, more than 95% of records, so it may combine more than one facility."))
        if r.malformed_record:
            rs.append(("malformed_record", "This record's fields look shifted (list text in the website field), so entries may be attributed to the wrong field."))
        if not r.has_first_party_source:
            rs.append(("no_first_party_source", "No source page from the facility's own website."))
        reasons.append(rs)
        trust.append(max(CONFIG["trust_floor"], round(1.0 - sum(pen[k] for k, _ in rs), 2)))
    f["trust_reasons"], f["trust"] = reasons, trust
    return f.reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Stage 2: evidence for every facility x service
# ----------------------------------------------------------------------------------
def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def facility_items(row):
    """Every citable unit in a record: (field, index, text)."""
    items = []
    for fld in TEXT_FIELDS:
        items += [(fld, i, t) for i, t in enumerate(row[fld + "_list"])]
    items += [("description", i, t) for i, t in enumerate(_sentences(row["description"]))]
    return items


def assess(row, key, items=None):
    """Grade the evidence that one facility offers one service.

    Returns a dict with tier ('strong' | 'moderate' | 'weak' | None), the evidence
    sentences used, and notes. Every evidence text is an exact entry from the record.
    """
    svc = SERVICES[key]
    items = facility_items(row) if items is None else items
    ev, direct, equip, supports, related, negated = [], [], [], set(), [], []
    for fld, i, t in items:
        if svc["exclude_re"] is not None and svc["exclude_re"].search(t):
            continue
        d = bool(svc["direct_re"].search(t))
        e = bool(svc["equipment_re"] and svc["equipment_re"].search(t))
        if svc.get("equipment_is_direct") and fld == "equipment" and (d or e):
            d = True
        if (d or e) and NEGATION is not None and NEGATION.search(t):
            negated.append((fld, i))
            ev.append({"role": "negative or referral context", "field": fld, "idx": i, "text": t})
            continue
        sup = [k for k in svc["supports"] if SUPPORT[k]["fn"](t)]
        if d:
            direct.append((fld, i))
            ev.append({"role": "states the service", "field": fld, "idx": i, "text": t})
        if e:
            equip.append((fld, i))
            if not d:
                ev.append({"role": "equipment", "field": fld, "idx": i, "text": t})
        for k in sup:
            if k not in supports:
                supports.add(k)
                if not (d or e):
                    ev.append({"role": f"supporting: {k}", "field": fld, "idx": i, "text": t})
        if not (d or e or sup) and svc["related_re"] and svc["related_re"].search(t):
            related.append((fld, i))
            if len(related) <= 2:
                ev.append({"role": "related", "field": fld, "idx": i, "text": t})
    codes = [str(c).lower() for c in row["specialties_list"]]
    spec = [c for c, low in zip(row["specialties_list"], codes) if any(t in low for t in svc["specialties"])]
    for k in svc["supports"]:      # some supports can also be shown by a specialty code (anaesthesia)
        if any(t in low for t in SUPPORT[k]["specialty"] for low in codes):
            supports.add(k)
    for s in spec[:2]:
        ev.append({"role": "specialty tag", "field": "specialties", "idx": row["specialties_list"].index(s), "text": s})

    # Corroboration must come from a different entry than the one stating the service.
    separate_equipment = any(e not in direct for e in equip) or (bool(equip) and len(set(direct)) > 1)
    enough_support = len(supports) >= svc.get("min_support", 99)
    two_fields = svc.get("equipment_is_direct") and len({f for f, _ in direct}) >= 2
    notes = []
    if direct and (separate_equipment or enough_support or two_fields):
        tier = "strong"
        why = "States the service, and " + (
            "a separate entry lists matching equipment." if separate_equipment else
            f"lists supporting capabilities ({', '.join(sorted(supports))})." if enough_support else
            "it appears in both the equipment list and the service lists.")
    elif direct:
        tier, why = "moderate", "States the service, but no separate entry lists matching equipment or enough supporting capabilities."
    elif equip or spec or related:
        tier, why = "weak", "Does not state the service. It is suggested only by " + " and ".join(
            x for x, ok in (("equipment", equip), ("a specialty tag", spec), ("related entries", related)) if ok) + "."
    else:
        tier, why = None, ""
    if tier == "strong" and svc.get("needs_247") and "24/7 emergency care" not in supports:
        tier = "moderate"
        why = "States the service with supporting capabilities, but nothing says emergency care is available 24/7."
        notes.append("No statement of 24/7 emergency availability.")
    if negated:
        notes.append("At least one entry mentions the service in a negative or referral context.")
        if not direct and tier:
            tier = "weak"
    ftype = "" if is_missing(row.get("facilityTypeId")) else str(row["facilityTypeId"]).lower()
    if tier and svc.get("hospital_level") and ftype in ("clinic", "dentist", "doctor", "pharmacy", "farmacy"):
        notes.append(f"The record's facility type is '{ftype}', not a hospital.")
    if tier and svc.get("small_staff_note", True) and pd.notna(row["numberDoctors_num"]) and row["numberDoctors_num"] <= 2:
        notes.append(f"Record lists only {int(row['numberDoctors_num'])} doctor(s).")
    return {"tier": tier, "why": why, "supports": sorted(supports), "notes": notes, "evidence": ev,
            "n_direct": len(set(direct)), "n_equipment": len(set(equip))}


def build_evidence(fac, keys=None):
    """Assess every facility against every service (or only the services in keys).

    Returns (summary, evidence):
      summary   one row per facility x service that has any evidence
      evidence  one row per cited sentence
    """
    srows, erows = [], []
    for row in fac.to_dict("records"):
        items = facility_items(row)
        blob = " \n ".join(t for _, _, t in items)
        specs = " ".join(row["specialties_list"]).lower()
        for key in (keys or SERVICES):
            svc = SERVICES[key]
            if not (svc["any_re"].search(blob) or any(t in specs for t in svc["specialties"])):
                continue
            a = assess(row, key, items)
            if not a["tier"]:
                continue
            srows.append({"unique_id": row["unique_id"], "service": key, "tier": a["tier"],
                          "evidence_value": CONFIG["tier_value"][a["tier"]], "why": a["why"],
                          "supports": a["supports"], "notes": a["notes"],
                          "n_direct": a["n_direct"], "n_equipment": a["n_equipment"]})
            for e in a["evidence"]:
                erows.append({"unique_id": row["unique_id"], "service": key, **e})
    summary = pd.DataFrame(srows, columns=["unique_id", "service", "tier", "evidence_value", "why",
                                           "supports", "notes", "n_direct", "n_equipment"])
    evidence = pd.DataFrame(erows, columns=["unique_id", "service", "role", "field", "idx", "text"])
    return summary, evidence


# ----------------------------------------------------------------------------------
# Stage 3: turn the user's location into coordinates
# ----------------------------------------------------------------------------------
_PLACE_FILLER = {"near", "nearby", "in", "at", "around", "from", "within", "outside", "close", "to", "the", "of",
                 "city", "town", "district", "dist", "area", "region", "taluk", "tehsil", "village", "rural", "urban"}
_KIND_ORDER = {"district": 0, "city": 1, "post office": 2}


class Gazetteer:
    """Place names -> coordinates, from the PIN directory, the facilities and location_aliases.yaml."""

    def __init__(self, fac=None, pins=None):
        self.pin = {}
        self.names = defaultdict(list)
        if fac is not None:
            self._build(fac, pins)
        self._index()

    def _build(self, fac, pins):
        if pins is not None and len(pins):
            ps = summarize_pins(pins).dropna(subset=["pin_lat", "pin_lon"])
            self.pin = {p: (float(r.pin_lat), float(r.pin_lon), r.pin_district, r.pin_state) for p, r in ps.iterrows()}
            ok = pins.dropna(subset=["lat", "lon"])
            dist = ok.dropna(subset=["district"]).groupby(["state", "district"], dropna=False).agg(
                lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size")).reset_index()
            for r in dist.itertuples(index=False):
                self._add(r.district, "district", f"{r.district} district, {r.state}", r.lat, r.lon, r.state, r.n)
            off = ok.dropna(subset=["office"]).copy()
            off["k"] = off["office"].str.replace(r"\s+(S\.?O|B\.?O|H\.?O|G\.?P\.?O)\.?$", "", regex=True, flags=re.I)
            offs = off.groupby(["state", "k"], dropna=False).agg(
                lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size"), district=("district", "first")).reset_index()
            for r in offs.itertuples(index=False):
                self._add(r.k, "post office", f"{r.k} post office area, {r.district}, {r.state}", r.lat, r.lon, r.state, r.n)
        loc = fac.dropna(subset=["lat", "lon"])
        loc = loc[~loc["address_city"].map(is_missing)]
        for (city, st), d in loc.groupby(["address_city", "state"], dropna=False):
            self._add(city, "city", f"{city}, {st} (centre of {len(d)} listed facilities)", d["lat"].median(), d["lon"].median(), st, len(d))
        # Districts the directory could not place (no coordinates there): fall back to facilities.
        for (dist, st), d in loc.dropna(subset=["district"]).groupby(["district", "state"], dropna=False):
            if _norm(dist) not in self.names:
                self._add(dist, "district", f"{dist} district, {st} (centre of {len(d)} listed facilities)", d["lat"].median(), d["lon"].median(), st, len(d))

    def _add(self, name, kind, label, lat, lon, state, n):
        if is_missing(name) or pd.isna(lat) or pd.isna(lon):
            return
        self.names[_norm(name)].append({"kind": kind, "label": label, "lat": float(lat), "lon": float(lon),
                                        "state": None if is_missing(state) else str(state), "n": int(n)})

    def _index(self):
        """Index the alias file against the names that exist in the data."""
        self.alias = defaultdict(list)
        for g in ALIASES.get("places", []):
            for n in g.get("names", []):
                self.alias[_norm(n)].append(g)
        self.state_alias = {_norm(n): names[0] for names in ALIASES.get("states", []) for n in names}
        # Names written in other scripts -> the English name they stand for.
        self.script = {}
        for eng, natives in (ALIASES.get("other_scripts") or {}).items():
            for n in natives or []:
                self.script.setdefault(_norm(n), str(eng))
        self.filler = {_norm(w) for w in ALIASES.get("filler_words") or []} or set(_PLACE_FILLER)
        # Misspellings are matched against districts and cities only; post-office names are too many and too similar.
        self._pool = sorted(k for k, v in self.names.items() if any(c["kind"] in ("district", "city") for c in v))
        self._alias_pool = sorted(k for k in self.alias if k.isascii())
        # The same names by how they sound, for place names typed in an Indian script.
        self._sound = defaultdict(list)
        for k in dict.fromkeys(self._pool + self._alias_pool):
            self._sound[sound_key(k)].append(k)

    # ---- saving and loading -------------------------------------------------------
    def to_tables(self):
        names = pd.DataFrame([{"key": k, **c} for k, cs in self.names.items() for c in cs],
                             columns=["key", "kind", "label", "lat", "lon", "state", "n"])
        pins = pd.DataFrame([{"pin": p, "lat": v[0], "lon": v[1], "district": v[2], "state": v[3]} for p, v in self.pin.items()],
                            columns=["pin", "lat", "lon", "district", "state"])
        return names, pins

    @classmethod
    def from_tables(cls, names, pins):
        g = cls()
        for r in names.itertuples(index=False):
            g.names[r.key].append({"kind": r.kind, "label": r.label, "lat": float(r.lat), "lon": float(r.lon),
                                   "state": None if is_missing(r.state) else str(r.state), "n": int(r.n)})
        for r in pins.itertuples(index=False):
            g.pin[str(r.pin)] = (float(r.lat), float(r.lon), None if is_missing(r.district) else r.district,
                                 None if is_missing(r.state) else r.state)
        g._index()
        return g

    # ---- lookup -------------------------------------------------------------------
    @staticmethod
    def _in_state(cands, state):
        if not state:
            return list(cands)
        s = _norm(state)
        return [c for c in cands if c["state"] and s in _norm(c["state"])] or list(cands)

    def _by_name(self, text, depth=0, other_scripts=True):
        """Candidates for one place name, plus a note saying how it was matched ('' if as typed)."""
        typed = str(text).strip()
        words = _norm(typed).split()
        core = [w for w in words if w not in self.filler]
        # As typed, then without filler words ("near", "district"), then dropping leading words.
        tries = list(dict.fromkeys(t for t in [" ".join(words), " ".join(core)] + [" ".join(core[i:]) for i in range(1, len(core))] if t))
        for t in tries:
            if t in self.names:
                return list(self.names[t]), ""
            eng = self.script.get(t) if other_scripts else None
            if eng:
                c, how = self._by_name(eng, depth, other_scripts=False)
                if c:
                    return c, f"'{typed}' was read as {eng}." + (f" {how}" if how else "")
            for g in self.alias.get(t, []):
                for n in g["names"]:
                    k = _norm(n)
                    if k != t and k in self.names:
                        return self._in_state(self.names[k], g.get("state")), f"'{typed}' is another name for {n}."
                if g.get("within") and depth == 0:
                    c, _ = self._by_name(g["within"], depth=1)
                    if c:
                        return self._in_state(c, g.get("state")), (
                            f"'{typed}' is not in the data by name. Using {g['within']}, the district that contains it.")
        return [], ""

    def _fuzzy(self, text):
        typed = str(text).strip()
        core = " ".join(w for w in _norm(typed).split() if w not in self.filler)
        if len(core) < 4:
            return [], ""
        cut = CONFIG.get("fuzzy_cutoff", 0.86)
        hit = difflib.get_close_matches(core, self._pool, n=1, cutoff=cut)
        if hit:
            c = self.names[hit[0]]
            return list(c), f"No place is named '{typed}'. Using the closest known name: {c[0]['label'].split(',')[0].split(' district')[0]}."
        hit = difflib.get_close_matches(core, self._alias_pool, n=1, cutoff=cut)
        if hit:
            c, _ = self._by_name(hit[0])
            if c:
                return c, f"No place is named '{typed}'. Using the closest known name: {c[0]['label'].split(',')[0].split(' district')[0]}."
        return [], ""

    def _by_sound(self, text):
        """A name typed in an Indian script that the alias file does not list: read it letter by
        letter and look for a district or city that sounds the same. Always says so in the note."""
        typed = str(text).strip()
        reading = romanize(" ".join(w for w in _norm(typed).split() if w not in self.filler))
        if not reading:
            return [], ""
        said = f"'{typed}' was read letter by letter as '{reading}'"
        check = "This is an automatic reading: check that it is the place you meant."
        c, how = self._by_name(reading, other_scripts=False)
        if c:
            return c, f"{said}. {how + ' ' if how else ''}{check}"
        key, flat = sound_key(reading), _loose(reading)
        if len(key) < 2:
            return [], ""
        scored = sorted(((difflib.SequenceMatcher(None, flat, _loose(k)).ratio(), k) for k in self._sound.get(key, [])), reverse=True)
        if scored and scored[0][0] >= CONFIG.get("sound_match_min", 0.85):
            c, _ = self._by_name(scored[0][1], other_scripts=False)
            if c:
                return c, f"{said}, which sounds like {c[0]['label'].split(',')[0].split(' district')[0]}. {check}"
        return [], ""

    def _state_name(self, text):
        """The standard name of a state, from its name in any spelling or Indian script (None if it is not a state)."""
        k = " ".join(w for w in _norm(text).split() if w not in self.filler)
        return self.state_alias.get(k) or (self.state_alias.get(_norm(romanize(k))) if scripts_in(k) else None)

    def resolve(self, query):
        """Returns {'status': 'ok' | 'ambiguous' | 'not_found', 'candidates': [...], 'note': str}.

        Accepts coordinates ("26.9, 75.8"), a 6-digit PIN anywhere in the text, or a place name
        with an optional state after a comma or in brackets ("Aurangabad, Bihar"). Digits and
        names may be written in an Indian script ("पटना", "८००००१").
        """
        q = "".join(str(unicodedata.decimal(c)) if (not c.isascii() and unicodedata.decimal(c, None) is not None) else c
                    for c in unicodedata.normalize("NFKC", str(query or "")))
        q = re.sub(r"\(([^)]*)\)", r", \1", q.replace("،", ",").replace("、", ",")).strip()
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*", q)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if not in_india(lat, lon):
                return {"status": "not_found", "candidates": [], "note": f"Coordinates {lat}, {lon} are outside India."}
            return {"status": "ok", "candidates": [{"kind": "coordinates", "label": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon, "state": None, "n": 1}], "note": "Used the coordinates as given."}
        p = extract_pin(q)      # a 6-digit PIN anywhere in the text, e.g. "near 800001"
        if p:
            if p not in self.pin:
                return {"status": "not_found", "candidates": [], "note": f"PIN {p} is not in the PIN directory."}
            lat, lon, dist, st = self.pin[p]
            return {"status": "ok", "candidates": [{"kind": "PIN", "label": f"PIN {p} ({dist}, {st})", "lat": lat, "lon": lon, "state": st, "n": 1}], "note": "Used the centre of the PIN code area."}
        parts = [x.strip() for x in q.split(",") if x.strip()]
        if not parts:
            return {"status": "not_found", "candidates": [], "note": "Enter a city, district, PIN code or coordinates."}
        name, hint = parts[0], (parts[1] if len(parts) > 1 else "")
        cands, how = self._by_name(name)
        guessed = False      # True when the place was matched by similarity, not by a known name
        if not cands:
            st = self._state_name(name)
            if st:
                return {"status": "not_found", "candidates": [], "note": f"{st} is a state. Add a city, district or PIN code inside it."}
            cands, how = self._fuzzy(name)
            guessed = bool(cands)
        used = scripts_in(name)
        if not cands and used:
            # A listed name whose English form is not in the data must not be matched to a sound-alike.
            eng = self.script.get(" ".join(w for w in _norm(name).split() if w not in self.filler))
            if eng:
                return {"status": "not_found", "candidates": [], "note": (
                    f"'{name}' is {eng}, and no district or city in the data has that name. "
                    "Try the district it belongs to, or its 6-digit PIN code.")}
        if not cands and any(sc in INDIC_BLOCKS for sc in used):
            cands, how = self._by_sound(name)
            guessed = bool(cands)
        if not cands:
            if used:
                return {"status": "not_found", "candidates": [], "note": (
                    f"Could not match '{q}' ({used[0]} script) to a place in the data. "
                    "Type the place in English letters, or give its 6-digit PIN code.")}
            return {"status": "not_found", "candidates": [], "note": f"Could not find '{q}'. Try a district name, a 6-digit PIN code, or 'lat, lon'."}
        if hint:
            cands = self._in_state(cands, self._state_name(hint) or hint)
        best = min(_KIND_ORDER.get(c["kind"], 9) for c in cands)
        # A district match beats a same-named post office elsewhere, so only matches of the best kind compete.
        cands = sorted((c for c in cands if _KIND_ORDER.get(c["kind"], 9) == best), key=lambda c: -c["n"])
        seen, uniq = set(), []
        for c in cands:      # one candidate per state: the same name in two states is a real ambiguity
            if c["state"] not in seen:
                seen.add(c["state"]); uniq.append(c)
        how = (how + " ") if how else ""
        if len(uniq) > 1:
            return {"status": "ambiguous", "candidates": uniq, "guessed": guessed,
                    "note": f"{how}'{name}' exists in more than one state. Using {uniq[0]['label']}. Add the state after a comma to choose another."}
        return {"status": "ok", "candidates": uniq, "guessed": guessed, "note": f"{how}Interpreted as {uniq[0]['label']}."}

    # ---- keeping the alias file honest -----------------------------------------------
    def alias_report(self):
        """How much of location_aliases.yaml matches names that exist in the data."""
        unmatched = []
        for g in ALIASES.get("places", []):
            ok = any(_norm(n) in self.names for n in g["names"]) or (g.get("within") and _norm(g["within"]) in self.names)
            if not ok:
                unmatched.append(g["names"][0])
        n = len(ALIASES.get("places", []))
        other = ALIASES.get("other_scripts") or {}
        lost = [eng for eng in other if not self._by_name(eng, other_scripts=False)[0]]
        return {"groups": n, "matched": n - len(unmatched), "unmatched": unmatched,
                "other_script_places": len(other), "other_script_names": sum(len(v or []) for v in other.values()),
                "other_script_unmatched": lost}

    def suggest_aliases(self, cutoff=0.88, max_km=80.0, limit=100):
        """Pairs of district or city names in the data that look like two spellings of one place
        (similar text, less than max_km apart) and are not yet grouped in the alias file."""
        group_of = {}
        for i, g in enumerate(ALIASES.get("places", [])):
            for n in g["names"]:
                group_of.setdefault(_norm(n), i)
        by_letter = defaultdict(list)
        for k in self._pool:
            if len(k) >= 5:
                by_letter[k[0]].append(k)
        out = []
        for ks in by_letter.values():
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    if abs(len(a) - len(b)) > 2 or (a in group_of and group_of.get(a) == group_of.get(b)):
                        continue
                    sm = difflib.SequenceMatcher(None, a, b)
                    if sm.quick_ratio() < cutoff or sm.ratio() < cutoff:
                        continue
                    ca, cb = self.names[a][0], self.names[b][0]
                    km = float(haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"]))
                    if km <= max_km:
                        out.append({"a": a, "b": b, "similarity": round(sm.ratio(), 2), "km_apart": round(km, 1), "state": ca["state"]})
        out.sort(key=lambda d: (-d["similarity"], d["km_apart"]))
        return out[:limit]


# ----------------------------------------------------------------------------------
# Stage 4: rank. Three steps so each can be changed without touching the others.
# ----------------------------------------------------------------------------------
FAC_QUERY_COLS = ["unique_id", "name", "address_city", "district", "state", "pin", "lat", "lon", "loc_source",
                  "loc_approximate", "trust", "trust_reasons", "officialPhone", "officialWebsite", "source_url_list",
                  "facilityTypeId", "numberDoctors_num", "capacity_num"]


def score_candidates(state, service, lat, lon):
    """Step 1: every located record with evidence for the service, scored.

    score = evidence x trust x proximity, each between 0 and 1.
      evidence   from scoring.yaml tier_value (1.0 corroborated, 0.6 claimed, 0.25 implied)
      trust      1.0 minus penalties for data problems in the record
      proximity  1 / (1 + (km / half_km)^2); half_km depends on the service's urgency
    The same facility listed in several records (same name and PIN) is kept once, best score first.
    Returns (candidates sorted by score, counts).
    """
    svc = SERVICES[service]
    half = CONFIG["half_km"][svc["urgency"]]
    c = state["summary"][state["summary"]["service"] == service].merge(state["fac"][FAC_QUERY_COLS], on="unique_id")
    counts = {"with_any_evidence": int(len(c)), "without_usable_location": int(c["lat"].isna().sum())}
    c = c.dropna(subset=["lat", "lon"]).copy()
    c["evidence_value"] = c["tier"].map(CONFIG["tier_value"]).astype(float)
    c["distance_km"] = haversine_km(lat, lon, c["lat"].values, c["lon"].values)
    c["proximity"] = 1.0 / (1.0 + (c["distance_km"] / half) ** 2)
    c["score"] = c["evidence_value"] * c["trust"] * c["proximity"]
    c = c.sort_values("score", ascending=False)
    c["_k"] = [_norm(n) + "|" + (p if not is_missing(p) else f"{la:.2f},{lo:.2f}")
               for n, p, la, lo in zip(c["name"], c["pin"], c["lat"], c["lon"])]
    c["n_records"] = c.groupby("_k")["unique_id"].transform("size")
    return c.drop_duplicates("_k", keep="first").drop(columns="_k"), counts


def pick(cands, service, k=None):
    """Step 2: decide what to show.

    Search the urgency's radius first. If nothing inside it states the service, widen it by
    scoring.yaml widen_factors, one step at a time. Records that only imply the service are
    shown as answers only when no record in range states it.
    """
    svc = SERVICES[service]
    k = k or CONFIG.get("top_k", 3)
    base = float(CONFIG["max_km"][svc["urgency"]])
    radii = [base] + [base * float(f) for f in CONFIG.get("widen_factors", [])]
    stated_any = cands["tier"] != "weak"
    radius = next((r for r in radii if (stated_any & (cands["distance_km"] <= r)).any()), None)
    if radius is None:      # nothing states it even in the widest radius
        radius = base if (cands["distance_km"] <= base).any() else radii[-1]
    within = cands[cands["distance_km"] <= radius]
    stated = within[within["tier"] != "weak"]
    strong = within[within["tier"] == "strong"].sort_values("distance_km")
    implied = within[within["tier"] == "weak"].sort_values("distance_km")
    top = (stated if len(stated) else implied).head(k)
    shown = set(top["unique_id"])
    alts = {}
    if len(strong) and strong.iloc[0]["unique_id"] not in shown:
        alts["Closest corroborated option"] = strong.iloc[0]
    if len(stated) and len(stated) < k and len(implied):
        alts["Nearby, but the service is only implied"] = implied.iloc[0]
    if not len(top) and len(cands):
        alts["Nearest option beyond the search radius"] = cands.sort_values("distance_km").iloc[0]
    return {"top": [r for _, r in top.iterrows()], "alternatives": alts,
            "base_km": base, "radius_km": float(radius), "widened": bool(radius > base and len(top)),
            "nearest_strong": strong.iloc[0] if len(strong) else None,
            "counts": {"within_radius": int(len(within)), "stating_within_radius": int(len(stated)),
                       "corroborated_within_radius": int(len(strong))}}


def verdict(picked, service):
    """Step 3: the one-sentence answer."""
    svc = SERVICES[service]
    name, top, r = svc["label"], picked["top"], picked["radius_km"]

    def km(d):
        return f"about {max(1, round(d))} km"

    lead = f"Nothing within {picked['base_km']:.0f} km states this service, so the search was widened to {r:.0f} km. " if picked["widened"] else ""
    if not top:
        return f"No facility with evidence of {name} was found within {r:.0f} km."
    first = top[0]
    if first["tier"] == "strong":
        text = f"Best supported option: {first['name']}, {km(first['distance_km'])} away."
    elif first["tier"] == "moderate" and picked["nearest_strong"] is not None:
        s0 = picked["nearest_strong"]
        text = (f"Best placed option: {first['name']}, {km(first['distance_km'])} away, but its record only states the service. "
                f"The nearest corroborated option is {s0['name']}, {km(s0['distance_km'])} away.")
    elif first["tier"] == "moderate":
        n = picked["counts"]["stating_within_radius"]
        text = (f"No corroborated option within {r:.0f} km. {n} record{'s' if n != 1 else ''} within that distance "
                f"state{'' if n != 1 else 's'} the service without supporting detail; the best placed is {first['name']}, "
                f"{km(first['distance_km'])} away. Confirm by phone before sending a patient.")
    else:
        text = (f"No record within {r:.0f} km states that it offers {name}. The options below only imply it "
                f"through a specialty or related entry. Confirm by phone before sending a patient.")
    if picked["widened"] and svc["urgency"] == "emergency":
        text += " At this distance, travel may take too long for an emergency."
    return lead + text


def travel_minutes(km):
    """Rough (fastest, slowest) road travel time in minutes for a straight-line distance.

    An estimate from the assumptions in scoring.yaml (circuity and speeds), not a route calculation.
    """
    t = CONFIG["travel"]
    return (60.0 * km * t["circuity_min"] / t["speed_fast_kmh"], 60.0 * km * t["circuity_max"] / t["speed_slow_kmh"])


def _duration(minutes):
    minutes = max(5, int(round(minutes / 5.0)) * 5)
    if minutes < 90:
        return f"{minutes} min"
    hours = minutes / 60.0
    return f"{hours:.0f} h" if hours >= 10 else f"{hours:.1f} h".replace(".0 h", " h")


def time_check(km, service):
    """Travel estimate for one option, and how it compares with the service's time target (if any)."""
    lo, hi = travel_minutes(km)
    fast, slow = _duration(lo), _duration(hi)
    out = {"travel_min": (lo, hi), "time_status": None, "time_text": "",
           "travel_text": f"roughly {fast} by road" if fast == slow else f"roughly {fast} to {slow} by road"}
    target = SERVICES[service].get("target_minutes")
    if target:
        if hi <= target:
            out["time_status"], out["time_text"] = "within", f"Even the slow estimate is inside the {target}-minute target for this service."
        elif lo <= target:
            out["time_status"], out["time_text"] = "uncertain", f"May or may not be reachable within the {target}-minute target for this service: it depends on roads and traffic."
        else:
            out["time_status"], out["time_text"] = "beyond", f"Even the fast estimate is longer than the {target}-minute target for this service."
    return out


def recommend(state, service, lat, lon, k=None):
    """score_candidates -> pick -> verdict, plus the evidence for each option shown."""
    svc = SERVICES[service]
    cands, counts = score_candidates(state, service, lat, lon)
    picked = pick(cands, service, k)
    ev = state["evidence"]
    ev_by = {uid: g.to_dict("records") for uid, g in ev[ev["service"] == service].groupby("unique_id")}

    def pack(row):
        d = row.to_dict()
        d["evidence"] = ev_by.get(d["unique_id"], [])
        d["notes"] = list(d["notes"])
        d.update(time_check(d["distance_km"], service))
        if d.get("n_records", 1) > 1:
            d["notes"].append(f"The dataset has {int(d['n_records'])} records with this name and PIN; the best-supported one is shown.")
        return d

    # How thinly is this area covered at all? Counted over every facility, not just this service.
    fac = state["fac"].dropna(subset=["lat", "lon"])
    near = int((haversine_km(lat, lon, fac["lat"].values, fac["lon"].values) <= CONFIG["sparse_radius_km"]).sum())
    sparse = ""
    if near < CONFIG["sparse_max_facilities"]:
        sparse = (f"Only {near} facilit{'y' if near == 1 else 'ies'} of any kind {'is' if near == 1 else 'are'} listed within "
                  f"{CONFIG['sparse_radius_km']:.0f} km of this location. The dataset covers rural areas thinly, "
                  f"so there may be options nearby that it does not list.")
    return {"service": service, "label": svc["label"], "urgency": svc["urgency"],
            "half_km": CONFIG["half_km"][svc["urgency"]], "max_km": picked["radius_km"], "base_km": picked["base_km"],
            "widened": picked["widened"], "verdict": verdict(picked, service),
            "top": [pack(r) for r in picked["top"]],
            "alternatives": {h: pack(r) for h, r in picked["alternatives"].items()},
            "counts": {**counts, **picked["counts"]}, "facilities_nearby": near, "sparse_note": sparse,
            "target_minutes": svc.get("target_minutes"), "target_note": svc.get("target_note", ""),
            "travel_assumptions": dict(CONFIG["travel"])}


# ----------------------------------------------------------------------------------
# Explain: build the answer from the template answer.html.j2
# ----------------------------------------------------------------------------------
_FIELD_NAME = {"procedure": "Procedures", "capability": "Capabilities", "equipment": "Equipment",
               "description": "Description", "specialties": "Specialties"}
_TIER_COLOR = {"strong": "#1a7f4b", "moderate": "#9a6400", "weak": "#b3372f"}
_template = None


def _get_template():
    global _template
    if _template is None:
        try:
            import jinja2
        except ImportError as err:      # pragma: no cover
            raise ImportError("The answer template needs jinja2. In a notebook run: %pip install jinja2") from err
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(BASE_DIR), autoescape=True,
                                 trim_blocks=True, lstrip_blocks=True)
        _template = env.get_template("answer.html.j2")
    return _template


def card_view(rec, rank=None, heading=None):
    """Everything one card shows, as plain data for the template."""
    tier = rec["tier"]
    groups = defaultdict(list)
    for ev in rec["evidence"]:
        if ev["role"] == "related" and tier != "weak":
            continue
        groups[ev["role"]].append({"text": ev["text"], "where": f"{_FIELD_NAME.get(ev['field'], ev['field'])}, entry {int(ev['idx']) + 1}"})
    order = sorted(groups, key=lambda r: (not r.startswith("states"), not r.startswith("equip"), r))
    warnings = [t for _, t in rec["trust_reasons"]] + list(rec["notes"])
    if rec["loc_approximate"]:
        warnings.append("Location is approximate (centre of the PIN code area), so the distance is too.")
    own = domain_of(rec.get("officialWebsite"))
    urls = sorted(rec["source_url_list"], key=lambda u: domain_of(u) != own)      # facility's own pages first
    return {
        "heading": heading, "rank": rank, "name": str(rec["name"]),
        "place": ", ".join(dedupe([str(x) for x in (rec.get("address_city"), rec.get("district"), rec.get("state")) if not is_missing(x)])),
        "distance": "under 1 km" if rec["distance_km"] < 1 else f"about {rec['distance_km']:.0f} km",
        "travel": rec.get("travel_text", ""), "time_status": rec.get("time_status"), "time_text": rec.get("time_text", ""),
        "phone": None if is_missing(rec.get("officialPhone")) else str(rec["officialPhone"]),
        "tier_label": TIER_LABEL[tier], "tier_color": _TIER_COLOR[tier], "why": rec["why"],
        "score": f"{rec['score']:.2f}", "evidence_value": f"{rec['evidence_value']:.2f}",
        "trust": f"{rec['trust']:.2f}", "proximity": f"{rec['proximity']:.2f}",
        "groups": [{"role": r[:1].upper() + r[1:], "entries": groups[r][:4], "more": max(0, len(groups[r]) - 4)} for r in order],
        "warnings": warnings,
        "sources": [{"url": u, "text": u[:90]} for u in urls[:3]],
    }


def render_html(result, place_note="", understood=None):
    """A self-contained HTML answer: verdict, ranked cards, alternatives, method.
    `understood` (from ask) is shown on top: how a free-text question was read."""
    svc = SERVICES[result["service"]]
    definition = None
    if svc.get("generated"):
        definition = {"request": svc["generated"].get("request", svc["label"]),
                      "rows": [("Counts as stating the service", svc["direct"]), ("Counts as backing it up", svc["equipment"]),
                               ("Only suggests it", svc["related"]), ("Specialty tags", svc["specialties"])]}
    return _get_template().render(
        message=None, result=result, place_note=place_note, definition=definition, understood=understood,
        cards=[card_view(r, rank=i + 1) for i, r in enumerate(result["top"])],
        alternatives=[card_view(r, heading=h) for h, r in result["alternatives"].items()],
        article="an" if result["urgency"][0] in "aeiou" else "a")


def render_message(text, understood=None):
    """A plain notice in the same visual style (location not found, model unavailable...)."""
    return _get_template().render(message=str(text), understood=understood)


def find_care(state, service, location, k=None, understood=None):
    """Service key + free-text location -> (result dict or None, HTML)."""
    if service not in SERVICES:
        return None, render_message(f"Unknown service '{service}'. Known services: {', '.join(SERVICES)}.", understood)
    loc = state["gaz"].resolve(location)
    if not loc["candidates"]:
        return None, render_message(loc["note"], understood)
    c = loc["candidates"][0]
    res = recommend(state, service, c["lat"], c["lon"], k)
    res["location"] = loc
    return res, render_html(res, loc["note"], understood)


# ----------------------------------------------------------------------------------
# Saving and loading: the state is everything a query needs
# ----------------------------------------------------------------------------------
_LIST_COLS = ["procedure_list", "capability_list", "equipment_list", "specialties_list", "source_url_list", "trust_reasons"]
FAC_SAVE_COLS = ["unique_id", "source_unique_id", "id_repeated", "name", "facilityTypeId", "operatorTypeId",
                 "address_city", "district", "state", "pin", "lat", "lon", "loc_source", "loc_approximate",
                 "location_conflict", "coords_outside_india", "coord_pin_km", "coord_region_km", "possible_merged",
                 "malformed_record", "n_phones", "n_source_ids", "has_first_party_source", "cluster_size",
                 "trust", "trust_reasons", "officialPhone", "officialWebsite", "numberDoctors_num", "capacity_num",
                 "yearEstablished_num", "description"] + _LIST_COLS[:5]
TABLES = ["facilities_clean", "facility_service_summary", "facility_service_evidence",
          "gazetteer_names", "gazetteer_pins", "service_definitions"]


def build_state(raw, pins_raw=None):
    """Run the whole offline pipeline. Returns the state dict used by find_care() and ask()."""
    pins = None
    if pins_raw is not None and len(pins_raw):
        pins = standardize_pins(pins_raw)
    fac = prepare_facilities(raw, pins)
    summary, evidence = build_evidence(fac)
    return {"fac": fac, "summary": summary, "evidence": evidence, "gaz": Gazetteer(fac, pins), "pins": pins}


def state_to_tables(state):
    """The state as six plain tables (lists stored as JSON text), ready to be saved."""
    def dump(v):
        return json.dumps(v.tolist() if isinstance(v, np.ndarray) else list(v), ensure_ascii=False)

    f = state["fac"][FAC_SAVE_COLS].copy()
    for c in _LIST_COLS:
        f[c] = f[c].map(dump)
    s = state["summary"].copy()
    for c in ("supports", "notes"):
        s[c] = s[c].map(dump)
    names, pins = state["gaz"].to_tables()
    defs = pd.DataFrame([{"service": k, "label": v["label"], "generated": bool(v.get("generated")),
                          "definition_yaml": yaml.safe_dump(service_to_yaml_dict(v), sort_keys=False, allow_unicode=True)}
                         for k, v in SERVICES.items()], columns=["service", "label", "generated", "definition_yaml"])
    return {"facilities_clean": f, "facility_service_summary": s, "facility_service_evidence": state["evidence"].copy(),
            "gazetteer_names": names, "gazetteer_pins": pins, "service_definitions": defs}


def state_from_tables(tables):
    """Rebuild the state from the six saved tables (pandas DataFrames keyed by table name).

    This is what an app calls at start-up instead of re-running the pipeline. Model-written
    service definitions found in service_definitions are registered again.
    """
    def load(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return [] if is_missing(v) else json.loads(v)

    f = tables["facilities_clean"].copy()
    for c in _LIST_COLS:
        f[c] = f[c].map(load)
    f["trust_reasons"] = f["trust_reasons"].map(lambda rs: [tuple(r) for r in rs])
    f["description"] = f["description"].map(lambda v: "" if is_missing(v) else str(v))
    s = tables["facility_service_summary"].copy()
    for c in ("supports", "notes"):
        s[c] = s[c].map(load)
    e = tables["facility_service_evidence"].copy()
    e["idx"] = e["idx"].astype(int)
    for r in tables["service_definitions"].itertuples(index=False):
        if bool(r.generated) and r.service not in SERVICES:
            SERVICES[r.service] = compile_service(yaml.safe_load(r.definition_yaml))
    gaz = Gazetteer.from_tables(tables["gazetteer_names"], tables["gazetteer_pins"])
    return {"fac": f.reset_index(drop=True), "summary": s.reset_index(drop=True), "evidence": e.reset_index(drop=True),
            "gaz": gaz, "pins": None}


# ----------------------------------------------------------------------------------
# Optional: a language model for free-text questions and services not predefined
# ----------------------------------------------------------------------------------
# The model never ranks, scores or quotes. It does two things: split a sentence into
# service + location, and write the term lists for a service that is not in services.yaml
# (same format as that file). Matching, grading and citation stay rule-based.
# `llm` is any function str -> str.
_GENERIC_TERMS = {"surgery", "treatment", "care", "hospital", "clinic", "service", "services", "doctor", "doctors",
                  "medical", "health", "patient", "patients", "procedure", "therapy", "consultation", "emergency",
                  "icu", "operation theatre", "operating theatre", "ot", "ward", "bed", "beds", "equipment", "facility"}


def _json_from(text):
    t = re.sub(r"```(?:json)?", "", str(text))
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("The model did not return JSON.")
    return json.loads(t[a:b + 1])


def _clean_terms(v, limit=10):
    out = []
    for x in v if isinstance(v, list) else []:
        x = re.sub(r"\s+", " ", re.sub(r"[^\w\s&/+.-]", " ", str(x))).strip().lower()
        if 3 <= len(x) <= 60 and x not in _GENERIC_TERMS:
            out.append(x)
    return dedupe(out)[:limit]


def define_service(llm, request, specialty_codes=()):
    """Ask the model for the term lists of a service and turn them into a services.yaml-style entry."""
    prompt = f"""You are helping search a directory of healthcare facilities in India. Each facility record has short
English sentences listing its procedures, capabilities and equipment.

Service requested: "{request}"

Return ONLY a JSON object with these keys:
- "label": short name of the service (max 5 words)
- "urgency": "emergency" (minutes to hours matter), "urgent" (needed within days or repeatedly) or "planned"
- "direct_terms": up to 8 lowercase words or phrases that, if they appear in a facility's record, mean the record
  says the facility provides this service. Include common synonyms and abbreviations used in India.
- "equipment_terms": up to 8 lowercase terms for equipment or dedicated units specific to this service.
  Do not include generic items such as ICU, operating theatre, ambulance or laboratory.
- "related_terms": up to 5 lowercase broader terms that suggest the service without stating it
- "specialties": up to 4 items copied exactly from this list, or an empty list: {json.dumps(list(specialty_codes)[:250])}"""
    data = _json_from(llm(prompt))
    direct = _clean_terms(data.get("direct_terms"), 8)
    if not direct:
        raise ValueError("The model returned no usable search terms for this service.")
    equip = _clean_terms(data.get("equipment_terms"), 8)
    related = [t for t in _clean_terms(data.get("related_terms"), 5) if t not in direct]
    allowed = set(specialty_codes)
    specs = [x for x in data.get("specialties", []) if isinstance(x, str) and x in allowed][:4]
    urgency = data.get("urgency") if data.get("urgency") in CONFIG["half_km"] else "urgent"
    label = re.sub(r"\s+", " ", str(data.get("label") or request)).strip()[:60]
    return compile_service({"label": label, "urgency": urgency, "direct": direct, "equipment": equip, "related": related,
                            "specialties": specs, "supports": [], "generated": {"request": str(request)}})


def parse_request(llm, text):
    """Free text in any language -> {'service', 'known_key', 'location', 'location_as_written', 'language'}.

    The model translates: the service comes back in English and the place in Latin letters,
    because the records and the place names in the data are in English.
    """
    known = {k: v["label"] for k, v in SERVICES.items() if not v.get("generated")}
    prompt = f"""Someone is looking for a healthcare facility in India. Their message may be in any language
(English, Hindi, Bengali, Marathi, Tamil, Telugu, Urdu and others, or a mix), in any script, including
an Indian language typed in Latin letters. The message:
"{text}"

Return ONLY a JSON object with these keys:
- "language": the language of the message, named in English (for example "English", "Hindi", "Tamil",
  "Hindi in Latin letters").
- "service": the medical service or care they ask for, translated to English, in a few words. Do not add
  anything they did not say and do not guess a service from symptoms. Use "" if none is stated.
- "known_key": the key from this list that means the same service, or null if none does: {json.dumps(known)}
- "location": only the place name, PIN code or coordinates they mention, without words such as "near" or "in".
  Write a place name in Latin letters, spelled the way it usually is in English, and digits as 0-9
  (for "near Patna" and for "पटना के पास" return "Patna"). Use "" if none.
- "location_as_written": that same place copied exactly as it appears in the message, in its original script."""
    data = _json_from(llm(prompt))
    key = data.get("known_key")
    return {"service": str(data.get("service") or "").strip(), "known_key": key if key in known else None,
            "location": str(data.get("location") or "").strip(),
            "location_as_written": str(data.get("location_as_written") or "").strip(),
            "language": re.sub(r"\s+", " ", str(data.get("language") or "")).strip()[:40]}


def _service_key(name):
    """Key under which a model-defined service is stored. Names that are not plain English letters
    get a short checksum, so that two different services can never end up with the same key."""
    key = "custom_" + _norm(name).replace(" ", "_")[:40]
    if key.isascii() and key != "custom_":
        return key
    slug = re.sub(r"[^a-z0-9]+", "_", "".join(c for c in _fold(name) if c.isascii())).strip("_")[:30]
    return "custom_" + (slug + "_" if slug else "") + format(zlib.crc32(_norm(name).encode("utf-8")), "08x")


def specialty_codes_of(state, limit=250):
    """The specialty codes that actually occur in the data, most common first."""
    return [c for c, _ in Counter(x for xs in state["fac"]["specialties_list"] for x in xs).most_common(limit)]


def ask(state, text, llm, specialty_codes=None, save_dir=None):
    """Answer a free-text question. Returns (result or None, HTML).

    A service that is not predefined gets a model-written definition, which is added to
    SERVICES, graded against every facility (state is updated in place) and written to
    services_generated.yaml so that it can be reviewed and reused.
    """
    try:
        req = parse_request(llm, text)
    except Exception as err:
        return None, render_message(f"Could not interpret the question ({err}). Use the service and location fields instead.")
    place = req["location"] or req["location_as_written"]
    und = {"question": str(text).strip(), "language": req["language"], "service": req["service"], "place": place}
    if not req["service"]:
        known = ", ".join(v["label"] for v in SERVICES.values() if not v.get("generated"))
        return None, render_message("The question does not name the service the patient needs, and this tool does not work one "
                                    f"out from symptoms. Name the service, for example: {known}.", und)
    if not place:
        return None, render_message("The question does not say where. Add a city, district, PIN code or coordinates.", und)
    key = req["known_key"]
    if not key:
        key = _service_key(req["service"])
        if key not in SERVICES:
            try:
                codes = specialty_codes if specialty_codes is not None else specialty_codes_of(state)
                SERVICES[key] = define_service(llm, req["service"], codes)
            except Exception as err:
                known = ", ".join(v["label"] for v in SERVICES.values() if not v.get("generated"))
                return None, render_message(f"Could not build a search for '{req['service']}' ({err}). Try one of: {known}.", und)
            try:
                save_generated_services(save_dir)
            except OSError:
                pass      # read-only folder: the definition still works for this session and is saved with the tables
        if not (state["summary"]["service"] == key).any():
            s_new, e_new = build_evidence(state["fac"], [key])
            state["summary"] = pd.concat([state["summary"], s_new], ignore_index=True)
            state["evidence"] = pd.concat([state["evidence"], e_new], ignore_index=True)
    und["service"] = SERVICES[key]["label"]
    # The place as the model spelled it in Latin letters; if that is not in the data, the place as the user wrote it.
    tries = list(dict.fromkeys(x for x in (req["location"], req["location_as_written"]) if x))
    for i, where in enumerate(tries):
        und["place"] = where
        res, page = find_care(state, key, where, understood=und)
        if res is not None or i == len(tries) - 1:
            break
    if res is None and len(tries) > 1:
        res, page = find_care(state, key, tries[0], understood=dict(und, place=tries[0]))      # report the failure for the first spelling
    if res is not None:
        res["request"] = req
    return res, page


# ----------------------------------------------------------------------------------
# Language check on the data
# ----------------------------------------------------------------------------------
def script_report(fac):
    """How much facility text is written in a script other than Latin.

    The evidence terms in services.yaml are English, so an entry in another script can
    never count as evidence: such a facility is under-credited, not mis-credited. This
    table says how often that can happen. Returns one row per field.
    """
    def other(v):
        return [] if (is_missing(v) or str(v).isascii()) else scripts_in(v)

    rows = []
    for field in ("name", "address_city", "description"):
        vals = [v for v in fac[field] if not is_missing(v) and str(v).strip()]
        hits = Counter(sc for v in vals for sc in other(v)[:1])
        rows.append({"field": field, "unit": "records", "with_text": len(vals), "other_script": sum(hits.values()),
                     "scripts": ", ".join(f"{k} {n}" for k, n in hits.most_common(4))})
    for field in TEXT_FIELDS:
        vals = [x for xs in fac[field + "_list"] for x in xs]
        hits = Counter(sc for v in vals for sc in other(v)[:1])
        rows.append({"field": field, "unit": "entries", "with_text": len(vals), "other_script": sum(hits.values()),
                     "scripts": ", ".join(f"{k} {n}" for k, n in hits.most_common(4))})
    out = pd.DataFrame(rows)
    out["share_pct"] = (100 * out["other_script"] / out["with_text"].clip(lower=1)).round(2)
    return out[["field", "unit", "with_text", "other_script", "share_pct", "scripts"]]


# ----------------------------------------------------------------------------------
# Validation: does the evidence grade mean anything?
# ----------------------------------------------------------------------------------
def validation_sample(fac, summary, per_cell=6, seed=7):
    """A stratified sample (service x tier) to label by hand against the source URLs.

    Fill the 'truly_offers' column with yes / no / unclear, then call tier_precision().
    """
    s = pd.concat([d.sample(min(per_cell, len(d)), random_state=seed)
                   for _, d in summary.groupby(["service", "tier"])]) if len(summary) else summary
    out = s.merge(fac[["unique_id", "name", "address_city", "state", "officialWebsite", "source_url_list"]], on="unique_id")
    out["source_urls"] = out.pop("source_url_list").map(lambda u: " | ".join(u[:4]))
    out["truly_offers"] = ""
    return out[["service", "tier", "name", "address_city", "state", "why", "officialWebsite", "source_urls", "truly_offers", "unique_id"]]


def tier_precision(labeled):
    """Share of 'yes' among yes/no labels, per tier. A useful grade has strong > moderate > weak."""
    d = labeled[labeled["truly_offers"].str.lower().isin(["yes", "no"])].copy()
    d["yes"] = d["truly_offers"].str.lower().eq("yes")
    return d.groupby("tier")["yes"].agg(precision="mean", labeled="size").reindex(["strong", "moderate", "weak"])
