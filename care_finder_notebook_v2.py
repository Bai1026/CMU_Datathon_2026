# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Care Finder: where can this patient get this service, and why trust the answer?
# MAGIC
# MAGIC Two inputs (a service and a location), a ranked answer, and for every recommendation the exact
# MAGIC record entries it rests on.
# MAGIC
# MAGIC **How it decides.** `score = evidence x trust x proximity`, each between 0 and 1.
# MAGIC
# MAGIC | Part | Meaning |
# MAGIC |---|---|
# MAGIC | Evidence | 1.0 the record states the service and a separate entry backs it up; 0.6 it only states it; 0.25 it is only implied |
# MAGIC | Trust | starts at 1.0, loses points for data problems (location conflict, coordinates outside India, possibly merged record, shifted fields, no first-party source) |
# MAGIC | Proximity | `1 / (1 + (km / half_km)^2)`; `half_km` is shorter for emergency services |
# MAGIC
# MAGIC **Run order.** Run all cells top to bottom once (a few minutes). After that, change the two
# MAGIC widgets at the top of the notebook and re-run only section 5.
# MAGIC
# MAGIC **What still needs a human.** The service definitions in `SERVICES` and the weights in `CONFIG`
# MAGIC are first drafts. Have someone with clinical knowledge review the definitions, and use section 6
# MAGIC to check the evidence grades against the source pages.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load

# COMMAND ----------

# 1. Where the data is, and where results go
SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
OUT = "workspace.care_finder"

raw = spark.table(f"{SRC}.facilities").toPandas()
pins_raw = spark.table(f"{SRC}.india_post_pincode_directory").toPandas()
print("facilities:", raw.shape, "| PIN directory:", pins_raw.shape)
print("PIN directory columns:", list(pins_raw.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Core logic
# MAGIC Everything below this heading down to section 3 is plain pandas. Change `CONFIG` and `SERVICES` here; nothing else needs editing to tune the method.

# COMMAND ----------

"""Care Finder core logic: which facility offers a service near a place, and why trust it.

Pure pandas, no Spark, so the same code runs in a Databricks notebook, in a Databricks
App, and on a laptop. Four stages:

  1. prepare_facilities   clean each record and attach trust flags
  2. build_evidence       for every facility x service, collect the exact sentences
                          that support it and grade the evidence
  3. Gazetteer.resolve    turn "Jaipur" / "302001" / "26.9, 75.8" into coordinates
  4. recommend            rank candidates by evidence x trust x proximity, and say why

Every number in CONFIG and every pattern in SERVICES is a starting point to review,
not a validated clinical standard.
"""
import html
import json
import math
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

CONFIG = {
    # lat_min, lat_max, lon_min, lon_max. Loose box around India incl. islands.
    "india_bbox": (6.0, 37.5, 68.0, 97.5),
    # Coordinates further than this from the PIN code's centroid count as a conflict.
    "coord_pin_max_km": 50.0,
    # A PIN region is the set of PIN codes sharing their first 3 digits. Coordinates inside
    # the region's radius (95th percentile of post-office distances, at least region_min_km,
    # plus a margin) are accepted even when they are far from the single PIN's centroid.
    "region_min_km": 50.0,
    "region_margin_km": 25.0,
    # A record whose phone-number count is in the top share of all records (and at least
    # merged_min_phones) may combine several facilities.
    "merged_phone_quantile": 0.95,
    "merged_min_phones": 5,
    # Trust starts at 1.0; each problem found subtracts its penalty.
    "penalties": {
        "location_conflict": 0.15,
        "coords_outside_india": 0.10,
        "possible_merged_record": 0.20,
        "malformed_record": 0.20,
        "no_first_party_source": 0.10,
    },
    "trust_floor": 0.30,
    # Evidence grade -> value used in the score.
    "tier_value": {"strong": 1.0, "moderate": 0.6, "weak": 0.25},
    # Distance at which proximity drops to 0.5, by how urgent the service is.
    "half_km": {"emergency": 25.0, "urgent": 60.0, "planned": 150.0},
    # Beyond this distance a facility is not offered as an answer.
    "max_km": {"emergency": 100.0, "urgent": 200.0, "planned": 500.0},
}

TIER_LABEL = {
    "strong": "Corroborated",
    "moderate": "Claimed, not corroborated",
    "weak": "Implied only",
}
TIER_MEANING = {
    "strong": "The record states the service and a separate entry backs it up.",
    "moderate": "The record states the service, but nothing else in it backs that up.",
    "weak": "The record never states the service. It is only suggested by a specialty or related entry.",
}

TEXT_FIELDS = ["procedure", "capability", "equipment"]


def _rx(pattern):
    return re.compile(pattern, re.I)


# Supporting capabilities: things a facility needs around a service to deliver it.
_T247 = _rx(r"24\s*[/x×]\s*7|24[\s-]*(hours?|hrs?)\b|round[\s-]the[\s-]clock")
_EMERG = _rx(r"emergenc|casualty|trauma")
_NEO = _rx(r"neonat|\bnicu\b|newborn|p(a)?ediatric")
_ICU = _rx(r"\bicu\b|intensive care|critical care")
SUPPORT = {
    "24/7 emergency care": lambda t: bool(_T247.search(t) and _EMERG.search(t)),
    "operating theatre": lambda t: bool(re.search(r"operati(on|ng) (theat(er|re)|room)|\bOT\b", t, re.I)),
    "blood bank": lambda t: bool(re.search(r"blood (bank|storage)", t, re.I)),
    "anaesthesia": lambda t: bool(re.search(r"an(a)?esthe", t, re.I)),
    "adult ICU": lambda t: bool(_ICU.search(t) and not _NEO.search(t)),
    "CT scanner": lambda t: bool(re.search(r"\bCT\b|computed tomography", t, re.I)),
    "ambulance": lambda t: bool(re.search(r"ambulance", t, re.I)),
}

# A mention in one of these contexts is not evidence that the service is offered.
NEGATION = _rx(
    r"\b(no|not|without|lacks?|unavailable|doesn'?t|refer(s|red|ral)?|nearest|nearby|outsourced)\b"
)

# Service definitions.
#   direct      the record says the service itself is offered
#   equipment   equipment that only makes sense if the service is delivered
#   related     nearby concepts that suggest, but do not state, the service
#   specialties specialty codes that suggest the service
#   supports    supporting capabilities that corroborate it (see SUPPORT)
#   min_support how many supports count as corroboration when no equipment is listed
#   needs_247   an emergency service: without a 24/7 statement it cannot be "strong"
SERVICES = {
    "dialysis": {
        "label": "Dialysis",
        "urgency": "urgent",
        "direct": r"dialysis|\bCRRT\b|renal replacement therapy",
        "equipment": r"dialysis (machine|unit|chair|station|bed|equipment|facilit|cent(er|re)|ward|room|suite)|dialy[sz]er|\bRO (water )?plant",
        "related": r"",
        "specialties": r"nephrology",
        "supports": [],
    },
    "emergency_c_section": {
        "label": "Emergency C-section",
        "urgency": "emergency",
        "direct": r"c[\s-]?section|c(a)?esarean|\bLSCS\b|emergency obstetric|obstetric emergenc|\bCEmONC\b",
        "equipment": r"labou?r (room|table|ward)|fetal monitor|\bCTG\b|cardiotocograph|an(a)?esthesia (machine|workstation)",
        "related": r"deliver(y|ies)|maternity|obstetric|high[\s-]risk pregnanc",
        "specialties": r"gynecologyAndObstetrics|obstetric|maternalFetal",
        "supports": ["24/7 emergency care", "operating theatre", "blood bank", "anaesthesia"],
        "min_support": 2,
        "needs_247": True,
    },
    "nicu": {
        "label": "NICU (newborn intensive care)",
        "urgency": "emergency",
        "direct": r"\bNICU\b|neonatal (intensive|icu|critical)|newborn (intensive|icu)|\bSNCU\b|special newborn care",
        "equipment": r"radiant warmer|infant warmer|incubator|phototherapy|bubble cpap|neonatal (ventilator|cpap|monitor|multi|syringe|pulse|central)",
        "related": r"neonat|newborn",
        "specialties": r"neonatology",
        "supports": [],
    },
    "cardiac_surgery": {
        "label": "Cardiac surgery",
        "urgency": "planned",
        "direct": r"\bCABG\b|bypass (surgery|graft)|open[\s-]heart|cardiac surger|heart surger|cardio[\s-]?thoracic|\bCTVS\b|valve (replacement|repair)",
        # Generic supports (ICU, theatre, blood bank) do not count here: nearly every hospital
        # lists them, so they say nothing about cardiac surgery in particular.
        "equipment": r"heart[\s-]lung machine|cardiopulmonary bypass|\bIABP\b|\bECMO\b|cath[\s-]?lab|cardiac (icu|ot\b|operation theat|operating theat)|\bCTVS (icu|ot\b)|\bCICU\b",
        "related": r"angioplast|angiograph|cardiolog",
        "specialties": r"cardi\w*surg|thoracic",
        "supports": [],
    },
    "mri": {
        "label": "MRI scan",
        "urgency": "planned",
        "direct": r"\bMRI\b|magnetic resonance",
        "equipment": r"tesla|MRI (scanner|machine|unit|system)",
        "related": r"",
        "specialties": r"radiology",
        "supports": [],
        "equipment_is_direct": True,
    },
    "ct_scan": {
        "label": "CT scan",
        "urgency": "urgent",
        "direct": r"\bCT\b|computed tomography",
        "equipment": r"CT (scanner|machine|unit|system)|slice CT",
        "related": r"",
        "specialties": r"radiology",
        "supports": [],
        "equipment_is_direct": True,
    },
    "burn_care": {
        "label": "Burn care",
        "urgency": "emergency",
        "direct": r"\bburns? (unit|ward|care|cent(er|re)|icu|treatment|management|surgery|injur)|treat\w* (of )?burns|\bburns\b",
        # As with cardiac surgery, generic supports are not counted as corroboration.
        "equipment": r"skin graft|dermatome|hydrotherapy|burns? (unit|ward|icu|cent(er|re)|bed)",
        "related": r"plastic surg|reconstructive",
        "specialties": r"plasticSurgery|burn",
        "supports": [],
    },
    "trauma": {
        "label": "Trauma care",
        "urgency": "emergency",
        "direct": r"trauma (cent(er|re)|care|surgery|unit|services?|team)|polytrauma|accident (and|&) emergency|road traffic accident",
        "equipment": r"fracture table|c[\s-]arm|defibrillator",
        "related": r"fracture|orthop(a)?edic",
        "specialties": r"emergencyMedicine|orthopedicSurgery|neurosurgery|trauma",
        "supports": ["24/7 emergency care", "operating theatre", "adult ICU", "blood bank", "CT scanner", "ambulance"],
        "min_support": 3,
        "needs_247": True,
    },
}
def compile_service(svc):
    """Add the compiled patterns a service definition needs. Returns the same dict."""
    for k in ("direct", "equipment", "related", "specialties"):
        svc[k + "_re"] = _rx(svc[k]) if svc.get(k) else None
    svc["any_re"] = _rx("|".join(p for p in (svc["direct"], svc.get("equipment"), svc.get("related")) if p))
    svc.setdefault("supports", [])
    return svc


for _s in SERVICES.values():
    compile_service(_s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Small helpers

# COMMAND ----------

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


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


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

# COMMAND ----------

# MAGIC %md
# MAGIC #### PIN code directory

# COMMAND ----------

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
    reg["pref_radius"] = ok.groupby("prefix")["d"].quantile(0.95).clip(lower=CONFIG["region_min_km"]) + CONFIG["region_margin_km"]
    reg.loc[reg["pref_n"] < 5, "pref_radius"] = np.nan
    return reg

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stage 1

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stage 2

# COMMAND ----------

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
        d = bool(svc["direct_re"].search(t))
        e = bool(svc["equipment_re"] and svc["equipment_re"].search(t))
        if svc.get("equipment_is_direct") and fld == "equipment" and (d or e):
            d = True
        if (d or e) and NEGATION.search(t):
            negated.append((fld, i))
            ev.append({"role": "negative or referral context", "field": fld, "idx": i, "text": t})
            continue
        sup = [k for k in svc["supports"] if SUPPORT[k](t)]
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
    spec = [s for s in row["specialties_list"] if svc["specialties_re"] and svc["specialties_re"].search(s)]
    if "anaesthesia" in svc["supports"] and any(re.search(r"an(a)?esthe", s, re.I) for s in row["specialties_list"]):
        supports.add("anaesthesia")
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
    if tier and key not in ("mri", "ct_scan", "dialysis") and pd.notna(row["numberDoctors_num"]) and row["numberDoctors_num"] <= 2:
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
        specs = " ".join(row["specialties_list"])
        for key in (keys or SERVICES):
            svc = SERVICES[key]
            if not (svc["any_re"].search(blob) or (svc["specialties_re"] and svc["specialties_re"].search(specs))):
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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stage 3

# COMMAND ----------

class Gazetteer:
    """Place names -> coordinates, built from the PIN directory and the facilities."""

    def __init__(self, fac, pins=None):
        self.pin = {}
        self.names = defaultdict(list)
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

    def resolve(self, query):
        """Returns {'status': 'ok' | 'ambiguous' | 'not_found', 'candidates': [...], 'note': str}."""
        q = str(query or "").strip()
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*", q)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if not in_india(lat, lon):
                return {"status": "not_found", "candidates": [], "note": f"Coordinates {lat}, {lon} are outside India."}
            return {"status": "ok", "candidates": [{"kind": "coordinates", "label": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon, "state": None, "n": 1}], "note": "Used the coordinates as given."}
        p = extract_pin(q) if re.fullmatch(r"[\d\s]+", q) else None
        if p:
            if p not in self.pin:
                return {"status": "not_found", "candidates": [], "note": f"PIN {p} is not in the PIN directory."}
            lat, lon, dist, st = self.pin[p]
            return {"status": "ok", "candidates": [{"kind": "PIN", "label": f"PIN {p} ({dist}, {st})", "lat": lat, "lon": lon, "state": st, "n": 1}], "note": "Used the centre of the PIN code area."}
        parts = [x.strip() for x in q.split(",") if x.strip()]
        if not parts:
            return {"status": "not_found", "candidates": [], "note": "Enter a city, district, PIN code or coordinates."}
        cands = list(self.names.get(_norm(parts[0]), []))
        if len(parts) > 1:
            hint = _norm(parts[1])
            cands = [c for c in cands if c["state"] and hint in _norm(c["state"])] or cands
        if not cands:
            return {"status": "not_found", "candidates": [], "note": f"Could not find '{q}'. Try a district name, a 6-digit PIN code, or 'lat, lon'."}
        order = {"district": 0, "city": 1, "post office": 2}
        best = min(order.get(c["kind"], 9) for c in cands)
        # A district match beats a same-named post office elsewhere, so only matches of the best kind compete.
        cands = [c for c in cands if order.get(c["kind"], 9) == best]
        cands.sort(key=lambda c: -c["n"])
        # Keep one candidate per state: the same name in two states is a real ambiguity.
        seen, uniq = set(), []
        for c in cands:
            if c["state"] not in seen:
                seen.add(c["state"]); uniq.append(c)
        if len(uniq) > 1:
            return {"status": "ambiguous", "candidates": uniq,
                    "note": f"'{parts[0]}' exists in more than one state. Using {uniq[0]['label']}. Add the state after a comma to choose another."}
        return {"status": "ok", "candidates": uniq, "note": f"Interpreted as {uniq[0]['label']}."}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stage 4

# COMMAND ----------

def recommend(fac, summary, evidence, service, lat, lon, k=3):
    """Rank facilities for a service near a point.

    score = evidence x trust x proximity, each between 0 and 1.
      evidence   1.0 corroborated, 0.6 claimed only, 0.25 implied only
      trust      1.0 minus penalties for data problems in the record
      proximity  1 / (1 + (distance / half_km)^2); half_km depends on urgency
    Distances are straight-line, not travel time.
    """
    svc = SERVICES[service]
    half, max_km = CONFIG["half_km"][svc["urgency"]], CONFIG["max_km"][svc["urgency"]]
    cols = ["unique_id", "name", "address_city", "district", "state", "pin", "lat", "lon", "loc_source", "loc_approximate",
            "trust", "trust_reasons", "officialPhone", "officialWebsite", "source_url_list", "facilityTypeId",
            "numberDoctors_num", "capacity_num"]
    c = summary[summary["service"] == service].merge(fac[cols], on="unique_id")
    n_claiming = len(c)
    n_unlocated = int(c["lat"].isna().sum())
    c = c.dropna(subset=["lat", "lon"]).copy()
    c["distance_km"] = haversine_km(lat, lon, c["lat"].values, c["lon"].values)
    c["proximity"] = 1.0 / (1.0 + (c["distance_km"] / half) ** 2)
    c["score"] = c["evidence_value"] * c["trust"] * c["proximity"]
    within = c[c["distance_km"] <= max_km].sort_values("score", ascending=False)
    # The same facility can appear in several records. Show it once, keeping its best-scoring record.
    within = within.assign(_k=[_norm(n) + "|" + (p if not is_missing(p) else f"{la:.2f},{lo:.2f}")
                               for n, p, la, lo in zip(within["name"], within["pin"], within["lat"], within["lon"])])
    within["n_records"] = within.groupby("_k")["unique_id"].transform("size")
    within = within.drop_duplicates("_k", keep="first").drop(columns="_k")
    ev_by = {uid: g.to_dict("records") for uid, g in evidence[evidence["service"] == service].groupby("unique_id")}

    def pack(r):
        d = r.to_dict()
        d["evidence"] = ev_by.get(d["unique_id"], [])
        d["notes"] = list(d["notes"])
        if d.get("n_records", 1) > 1:
            d["notes"].append(f"The dataset has {int(d['n_records'])} records with this name and PIN; the best-supported one is shown.")
        return d

    stated = within[within["tier"] != "weak"]          # records that actually state the service
    strong = within[within["tier"] == "strong"].sort_values("distance_km")
    implied = within[within["tier"] == "weak"].sort_values("distance_km")
    name = svc["label"]
    top = [pack(r) for _, r in (stated if len(stated) else implied).head(k).iterrows()]
    top_ids = {t["unique_id"] for t in top}
    alts = {}
    if len(strong) and strong.iloc[0]["unique_id"] not in top_ids:
        alts["Closest corroborated option"] = pack(strong.iloc[0])
    if len(stated) and len(stated) < k and len(implied):
        alts["Nearby, but the service is only implied"] = pack(implied.iloc[0])

    def km(d):
        return f"about {max(1, round(d))} km"

    if not top:
        verdict = f"No facility with evidence of {name} was found within {max_km:.0f} km."
        further = c.sort_values("distance_km").head(1)
        if len(further):
            alts["Nearest option beyond the search radius"] = pack(further.iloc[0])
    elif top[0]["tier"] == "strong":
        verdict = f"Best supported option: {top[0]['name']}, {km(top[0]['distance_km'])} away."
    elif top[0]["tier"] == "moderate" and len(strong):
        s0 = strong.iloc[0]
        verdict = (f"Best placed option: {top[0]['name']}, {km(top[0]['distance_km'])} away, but its record only states the service. "
                   f"The nearest corroborated option is {s0['name']}, {km(s0['distance_km'])} away.")
    elif top[0]["tier"] == "moderate":
        n = len(stated)
        verdict = (f"No corroborated option within {max_km:.0f} km. {n} record{'s' if n != 1 else ''} within that distance "
                   f"state{'' if n != 1 else 's'} the service without supporting detail; the best placed is {top[0]['name']}, "
                   f"{km(top[0]['distance_km'])} away. Confirm by phone before sending a patient.")
    else:
        verdict = (f"No record within {max_km:.0f} km states that it offers {name}. The options below only imply it "
                   f"through a specialty or related entry. Confirm by phone before sending a patient.")
    return {"service": service, "label": svc["label"], "urgency": svc["urgency"], "half_km": half, "max_km": max_km,
            "verdict": verdict, "top": top, "alternatives": alts,
            "counts": {"with_any_evidence": n_claiming, "without_usable_location": n_unlocated, "within_radius": int(len(within)),
                       "stating_within_radius": int(len(stated)), "corroborated_within_radius": int(len(strong))}}


_FIELD_NAME = {"procedure": "Procedures", "capability": "Capabilities", "equipment": "Equipment",
               "description": "Description", "specialties": "Specialties"}
_TIER_COLOR = {"strong": "#1a7f4b", "moderate": "#9a6400", "weak": "#b3372f"}


def _card(rec, rank=None, heading=None):
    e = html.escape
    tier = rec["tier"]
    place = ", ".join(dedupe([str(x) for x in (rec.get("address_city"), rec.get("district"), rec.get("state")) if not is_missing(x)]))
    km = "under 1 km" if rec["distance_km"] < 1 else f"about {rec['distance_km']:.0f} km"
    own = domain_of(rec.get("officialWebsite"))
    urls = sorted(rec["source_url_list"], key=lambda u: domain_of(u) != own)   # facility's own pages first
    groups = defaultdict(list)
    for ev in rec["evidence"]:
        if ev["role"] == "related" and tier != "weak":
            continue
        groups[ev["role"]].append(ev)
    ev_html = ""
    for role in sorted(groups, key=lambda r: (not r.startswith("states"), not r.startswith("equip"), r)):
        rows = "".join(f'<li>&ldquo;{e(x["text"])}&rdquo; <span style="color:#667">[{_FIELD_NAME.get(x["field"], x["field"])}, entry {x["idx"] + 1}]</span></li>'
                       for x in groups[role][:4])
        more = f'<li style="color:#667">and {len(groups[role]) - 4} more</li>' if len(groups[role]) > 4 else ""
        ev_html += f'<div style="margin-top:6px"><b>{e(role.capitalize())}</b><ul style="margin:2px 0 0 18px;padding:0">{rows}{more}</ul></div>'
    warn = [t for _, t in rec["trust_reasons"]] + list(rec["notes"])
    if rec["loc_approximate"]:
        warn.append("Location is approximate (centre of the PIN code area), so the distance is too.")
    warn_html = "".join(f"<li>{e(w)}</li>" for w in warn) or "<li>None found in the record.</li>"
    srcs = "".join(f'<li><a href="{e(u)}" style="color:#1f5fbf">{e(u[:90])}</a></li>' for u in urls[:3])
    phone = "" if is_missing(rec.get("officialPhone")) else f' &middot; Phone {e(str(rec["officialPhone"]))}'
    title = heading or (f"{rank}. " if rank else "")
    return f"""
<div style="border:1px solid #cfd6dd;border-radius:10px;padding:14px 16px;margin:10px 0;background:#fff;color:#16222c;font-family:system-ui,sans-serif;font-size:14px;line-height:1.5">
  <div style="font-size:12px;color:#667;text-transform:uppercase;letter-spacing:.06em">{e(heading) if heading else ''}</div>
  <div style="font-size:17px;font-weight:600">{'' if heading else e(title)}{e(str(rec['name']))}</div>
  <div style="color:#445">{e(place)} &middot; {km}{phone}</div>
  <div style="margin-top:8px"><span style="background:{_TIER_COLOR[tier]};color:#fff;border-radius:999px;padding:2px 10px;font-size:12.5px;font-weight:600">{TIER_LABEL[tier]}</span>
    <span style="color:#445"> {e(rec['why'])}</span></div>
  <div style="margin-top:8px;font-variant-numeric:tabular-nums">Score <b>{rec['score']:.2f}</b> = evidence {rec['evidence_value']:.2f} &times; trust {rec['trust']:.2f} &times; proximity {rec['proximity']:.2f}</div>
  <div style="margin-top:8px"><b>Evidence from the record</b>{ev_html}</div>
  <div style="margin-top:8px"><b>Check before relying on this</b><ul style="margin:2px 0 0 18px;padding:0">{warn_html}</ul></div>
  <div style="margin-top:8px"><b>Sources</b><ul style="margin:2px 0 0 18px;padding:0">{srcs or '<li>No source URL in the record.</li>'}</ul></div>
</div>"""


def _definition_html(key):
    g = SERVICES[key].get("generated")
    if not g:
        return ""
    e = html.escape
    rows = "".join(f"<div><b>{name}:</b> {e(', '.join(g[k])) or 'none'}</div>" for name, k in (
        ("Counts as stating the service", "direct_terms"), ("Counts as backing it up", "equipment_terms"),
        ("Only suggests it", "related_terms"), ("Specialty tags", "specialties")))
    return (f'<div style="margin:10px 0;padding:10px 14px;border:1px dashed #8a97a5;border-radius:10px;background:#fff;font-size:13px;line-height:1.5">'
            f'<b>Search definition written by an AI model for &ldquo;{e(g["request"])}&rdquo;.</b> '
            f'It has not been clinically reviewed. The matching and the quotes below are done by fixed rules, not by the model.{rows}</div>')


def render_html(result, place_note=""):
    """A self-contained HTML answer: verdict, ranked cards, alternatives, method."""
    e = html.escape
    cards = "".join(_card(r, i + 1) for i, r in enumerate(result["top"]))
    alts = "".join(_card(r, heading=h) for h, r in result["alternatives"].items())
    c = result["counts"]
    return f"""
<div style="font-family:system-ui,sans-serif;color:#16222c;background:#f5f7f9;padding:16px;border-radius:12px;max-width:860px">
  <div style="font-size:13px;color:#556">{e(result['label'])} &middot; {e(place_note)}</div>
  <div style="font-size:19px;font-weight:600;margin-top:4px">{e(result['verdict'])}</div>
  {_definition_html(result['service'])}
  {cards}
  {('<div style="margin-top:14px;font-weight:600">Other options worth knowing</div>' + alts) if alts else ''}
  <div style="margin-top:14px;font-size:13px;color:#445">
    <b>How this was decided.</b> {c['with_any_evidence']} records mention or imply this service. Within {result['max_km']:.0f} km,
    {c['stating_within_radius']} state it and {c['corroborated_within_radius']} of those are corroborated
    ({c['without_usable_location']} records were left out because their location could not be established).
    Records that only imply the service are shown only when none states it.
    Each is scored as evidence &times; trust &times; proximity. Evidence is 1.0 when the record states the service and a separate entry backs it up,
    0.6 when it only states it, 0.25 when it is only implied. Trust starts at 1.0 and loses points for data problems listed on each card.
    Proximity halves at {result['half_km']:.0f} km because this is treated as {'an' if result['urgency'][0] in 'aeiou' else 'a'} {result['urgency']} service.
    Distances are straight-line, not travel time. Records are web-extracted claims, not verified facts: call the facility before sending a patient.
  </div>
</div>"""


def find_care(fac, summary, evidence, gaz, service, location, k=3):
    """Service + free-text location -> (result dict or None, HTML)."""
    loc = gaz.resolve(location)
    if not loc["candidates"]:
        return None, f'<div style="font-family:system-ui;padding:12px;background:#fbe9e7;color:#16222c;border-radius:8px">{html.escape(loc["note"])}</div>'
    c = loc["candidates"][0]
    res = recommend(fac, summary, evidence, service, c["lat"], c["lon"], k)
    res["location"] = loc
    return res, render_html(res, loc["note"])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Optional

# COMMAND ----------

# The model never ranks, scores or quotes. It does two things: split a sentence into
# service + location, and write the search terms for a service that is not in SERVICES.
# Matching, grading and citation stay rule-based. `llm` is any function str -> str.
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
        x = re.sub(r"\s+", " ", str(x)).strip().lower()
        if 3 <= len(x) <= 60 and x not in _GENERIC_TERMS:
            out.append(x)
    return dedupe(out)[:limit]


def _term_regex(terms):
    return "|".join(r"\b" + r"[\s-]+".join(re.escape(w) for w in t.split()) for t in terms)


def define_service(llm, request, specialty_codes=()):
    """Ask the model for a search definition of a service and turn it into a SERVICES entry."""
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
    equip = [t for t in _clean_terms(data.get("equipment_terms"), 8)]
    related = [t for t in _clean_terms(data.get("related_terms"), 5) if t not in direct]
    allowed = set(specialty_codes)
    specs = [x for x in data.get("specialties", []) if isinstance(x, str) and x in allowed][:4]
    urgency = data.get("urgency") if data.get("urgency") in CONFIG["half_km"] else "urgent"
    label = re.sub(r"\s+", " ", str(data.get("label") or request)).strip()[:60]
    return compile_service({
        "label": label, "urgency": urgency, "direct": _term_regex(direct), "equipment": _term_regex(equip),
        "related": _term_regex(related), "specialties": "|".join(r"\b" + re.escape(x) + r"\b" for x in specs),
        "supports": [],
        "generated": {"request": str(request), "direct_terms": direct, "equipment_terms": equip,
                      "related_terms": related, "specialties": specs}})


def parse_request(llm, text):
    """Free text -> {'service': str, 'known_key': key or None, 'location': str}."""
    known = {k: v["label"] for k, v in SERVICES.items() if not v.get("generated")}
    prompt = f"""Someone is looking for a healthcare facility in India. Their message:
"{text}"

Return ONLY a JSON object with these keys:
- "service": the medical service or care they need, in a few words. Use "" if none is stated.
- "known_key": the key from this list that means the same service, or null if none does: {json.dumps(known)}
- "location": the place they mention (city, district, PIN code or coordinates), copied from the message. Use "" if none."""
    data = _json_from(llm(prompt))
    key = data.get("known_key")
    return {"service": str(data.get("service") or "").strip(), "known_key": key if key in known else None,
            "location": str(data.get("location") or "").strip()}


def ask(state, text, llm, specialty_codes=()):
    """Answer a free-text question. state holds fac, summary, evidence, gaz and is updated in place
    when a new service definition is created. Returns (result or None, HTML)."""
    def box(msg):
        return None, f'<div style="font-family:system-ui;padding:12px;background:#fbe9e7;color:#16222c;border-radius:8px">{html.escape(msg)}</div>'
    try:
        req = parse_request(llm, text)
    except Exception as err:
        return box(f"Could not interpret the question ({err}). Use the service and location fields instead.")
    if not req["service"]:
        return box("The question does not say which service is needed.")
    if not req["location"]:
        return box("The question does not say where. Add a city, district, PIN code or coordinates.")
    key = req["known_key"]
    if not key:
        key = "custom_" + _norm(req["service"]).replace(" ", "_")[:40]
        if key not in SERVICES:
            try:
                SERVICES[key] = define_service(llm, req["service"], specialty_codes)
            except Exception as err:
                return box(f"Could not build a search for '{req['service']}' ({err}). Try one of: {', '.join(v['label'] for v in SERVICES.values() if not v.get('generated'))}.")
            s_new, e_new = build_evidence(state["fac"], [key])
            state["summary"] = pd.concat([state["summary"][state["summary"]["service"] != key], s_new], ignore_index=True)
            state["evidence"] = pd.concat([state["evidence"][state["evidence"]["service"] != key], e_new], ignore_index=True)
    res, page = find_care(state["fac"], state["summary"], state["evidence"], state["gaz"], key, req["location"])
    if res is not None:
        res["request"] = req
    return res, page

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validation

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run

# COMMAND ----------

# 3. Run the pipeline (once)
try:
    pins = standardize_pins(pins_raw)
    print(f"PIN directory: {pins['pincode'].nunique():,} PIN codes, "
          f"{pins['lat'].notna().mean():.0%} of rows have usable coordinates")
    if pins["lat"].notna().sum() == 0:
        print("WARNING: the PIN directory has no usable coordinates. PIN cross-checks are off; "
              "locations rely on each record's own coordinates.")
except ValueError as err:
    pins = None
    print("WARNING: PIN directory not usable, continuing without it:", err)

fac = prepare_facilities(raw, pins)
summary, evidence = build_evidence(fac)
gaz = Gazetteer(fac, pins)
print(f"{len(fac):,} facilities | {len(summary):,} facility-service pairs with evidence | {len(evidence):,} cited entries")
print("duplicates:", PREP_STATS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality and evidence overview

# COMMAND ----------

# 4. What the data looks like (use these numbers in the write-up)
flags = pd.DataFrame({
    "facilities": [len(fac)],
    "no usable location": [(fac["loc_source"] == "none").sum()],
    "coordinates outside India": [fac["coords_outside_india"].sum()],
    "coordinates vs PIN conflict": [fac["location_conflict"].sum()],
    "located by PIN centroid": [fac["loc_approximate"].sum()],
    "possibly merged record": [fac["possible_merged"].sum()],
    "fields look shifted": [fac["malformed_record"].sum()],
    "no first-party source": [(~fac["has_first_party_source"]).sum()],
}).T.rename(columns={0: "count"})
flags["share"] = (flags["count"] / len(fac)).map("{:.1%}".format)
display(flags.reset_index().rename(columns={"index": "check"}))

tiers = summary.pivot_table(index="service", columns="tier", values="unique_id", aggfunc="count", fill_value=0)
tiers = tiers.reindex(columns=["strong", "moderate", "weak"], fill_value=0)
tiers["corroborated share of claims"] = (tiers["strong"] / (tiers["strong"] + tiers["moderate"]).clip(lower=1)).map("{:.0%}".format)
# The same share among records that list any equipment at all. A record with an empty
# equipment list cannot be corroborated by equipment, so this is the fairer number.
has_eq = dict(zip(fac["unique_id"], fac["equipment_list"].map(len) > 0))
t2 = summary[summary["unique_id"].map(has_eq).astype(bool)].pivot_table(index="service", columns="tier", values="unique_id", aggfunc="count", fill_value=0)
t2 = t2.reindex(columns=["strong", "moderate", "weak"], fill_value=0)
tiers["same, records with an equipment list"] = (t2["strong"] / (t2["strong"] + t2["moderate"]).clip(lower=1)).map("{:.0%}".format)
display(tiers.reset_index())

# COMMAND ----------

# 4b. Diagnostics. Paste this cell's output when asking for help tuning the method.
def q(s, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    s = pd.Series(s).dropna().astype(float)
    return {f"p{int(x * 100)}": round(float(s.quantile(x)), 1) for x in qs} if len(s) else {}

cut = max(CONFIG["merged_min_phones"], float(fac["n_phones"].quantile(CONFIG["merged_phone_quantile"])))
print("phones per record:", q(fac["n_phones"]), "| flagged as possibly merged at >=", cut)
print("record has a PIN:", f"{fac['pin'].notna().mean():.1%}", "| PIN centroid known:", f"{fac['pin_known'].mean():.1%}",
      "| PIN region known:", f"{fac['pref_radius'].notna().mean():.1%}")
print("km, coordinates to PIN centroid:", q(fac["coord_pin_km"]))
print("km, coordinates to PIN-region centre:", q(fac["coord_region_km"]), "| region radius km:", q(fac["pref_radius"]))
print("\nhow each location was chosen:"); print(fac["loc_source"].value_counts().to_string())
if pins is not None:
    _ps, _reg = summarize_pins(pins), prefix_regions(pins)
    _chk = _ps.join(_reg, on=_ps.index.str[:3])
    _far = haversine_km(_chk["pin_lat"], _chk["pin_lon"], _chk["pref_lat"], _chk["pref_lon"]) > _chk["pref_radius"]
    print(f"\nPIN directory: {len(_ps):,} PINs, {int(_ps['pin_lat'].isna().sum()):,} without coordinates, "
          f"{int(_far.sum()):,} with a centroid outside their own region (ignored)")
print("\nrecords sharing exact coordinates with another record:", int((fac["cluster_size"] > 1).sum()))
_key = fac["name"].map(_norm) + "|" + fac["pin"].fillna("")
print("records sharing name + PIN with another record:", int(_key.duplicated(keep=False).sum()))
print("duplicate rows and ids:", PREP_STATS)
print("records whose fields look shifted:", int(fac["malformed_record"].sum()))
print("records with an equipment list:", f"{fac['equipment_list'].map(len).gt(0).mean():.1%}")
print("\nfacility types:"); print(fac["facilityTypeId"].value_counts(dropna=False).head(8).to_string())
_dom = Counter(domain_of(u) for urls in fac["source_url_list"] for u in urls)
print("\ntop source domains:", [(d, n) for d, n in _dom.most_common(25) if d])

# COMMAND ----------

# Save the results as tables so a dashboard, Genie space or app can read them
from pyspark.sql import types as T

def to_spark(pdf):
    """pandas -> Spark with an explicit schema. Lists and dicts are stored as JSON text."""
    pdf = pdf.copy()
    fields = []
    for c in pdf.columns:
        col = pdf[c]
        if pd.api.types.is_bool_dtype(col):
            fields.append(T.StructField(c, T.BooleanType()))
        elif pd.api.types.is_integer_dtype(col):
            fields.append(T.StructField(c, T.LongType()))
        elif pd.api.types.is_float_dtype(col):
            fields.append(T.StructField(c, T.DoubleType()))
        else:
            vals = []
            for v in col:
                if isinstance(v, (list, tuple, dict, np.ndarray)):
                    vals.append(json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, ensure_ascii=False))
                else:
                    vals.append(None if is_missing(v) else str(v))
            pdf[c] = pd.Series(vals, index=pdf.index, dtype=object)
            fields.append(T.StructField(c, T.StringType()))
    return spark.createDataFrame(pdf, schema=T.StructType(fields))

FAC_COLS = ["unique_id", "source_unique_id", "id_repeated", "name", "facilityTypeId", "operatorTypeId", "address_city", "district", "state", "pin",
            "lat", "lon", "loc_source", "loc_approximate", "location_conflict", "coords_outside_india", "coord_pin_km",
            "coord_region_km", "possible_merged", "malformed_record", "n_phones", "n_source_ids", "has_first_party_source", "cluster_size",
            "trust", "trust_reasons", "officialPhone", "officialWebsite",
            "numberDoctors_num", "capacity_num", "yearEstablished_num", "source_url_list"]

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT}")
for name, pdf in [("facilities_clean", fac[FAC_COLS]), ("facility_service_summary", summary), ("facility_service_evidence", evidence)]:
    to_spark(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
    print("saved", f"{OUT}.{name}", len(pdf))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Find care

# COMMAND ----------

# 5. Ask a question: set the two widgets at the top, then run this cell
dbutils.widgets.dropdown("service", "dialysis", sorted(SERVICES))
dbutils.widgets.text("location", "Jaipur")

result, page = find_care(fac, summary, evidence, gaz, dbutils.widgets.get("service"), dbutils.widgets.get("location"))
displayHTML(page)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validation

# COMMAND ----------

# 6. Check the evidence grades by hand
# Open each row's source URLs, decide whether the facility really offers the service,
# and fill truly_offers with yes / no / unclear. Then run tier_precision(labeled).
# A grade that means something shows strong > moderate > weak.
sample = validation_sample(fac, summary, per_cell=6)
display(sample)
# labeled = <the sample with truly_offers filled in, as a pandas DataFrame>
# display(tier_precision(labeled).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optional: plain-language questions and any service
# MAGIC A Databricks-hosted model does two small jobs here: it splits a sentence into service and location, and for a service that is not predefined it writes the search terms. It never ranks, scores or quotes. The terms it wrote are shown on the answer, and the matching and citations are still done by the fixed rules above.

# COMMAND ----------

# 7a. Connect to a Databricks-hosted model (no API key needed)
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
_llm_route = None

def _text(content):
    if isinstance(content, list):   # some models return a list of parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)

def _via_sdk(prompt):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    r = WorkspaceClient().serving_endpoints.query(
        name=LLM_MODEL, messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)], temperature=0.0, max_tokens=900)
    return _text(r.choices[0].message.content)

def _via_rest(prompt):
    import requests
    from databricks.sdk import WorkspaceClient
    cfg = WorkspaceClient().config
    r = requests.post(f"{cfg.host.rstrip('/')}/serving-endpoints/{LLM_MODEL}/invocations",
                      headers={**cfg.authenticate(), "Content-Type": "application/json"},
                      json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 900}, timeout=90)
    r.raise_for_status()
    return _text(r.json()["choices"][0]["message"]["content"])

def _via_sql(prompt):
    return spark.sql(f"SELECT ai_query('{LLM_MODEL}', :p) AS a", args={"p": prompt}).collect()[0]["a"]

def llm(prompt):
    """Send one prompt to the model. Tries three ways to reach it and remembers the one that works."""
    global _llm_route
    errors = []
    for route in ([_llm_route] if _llm_route else [_via_sdk, _via_rest, _via_sql]):
        try:
            out = route(prompt)
            if out and str(out).strip():
                _llm_route = route
                return str(out)
        except Exception as err:
            errors.append(f"{route.__name__}: {type(err).__name__}: {str(err)[:300]}")
    raise RuntimeError("No route to the model worked. " + " | ".join(errors))

try:
    print("model replied:", llm("Reply with the single word OK").strip()[:40], "| route:", _llm_route.__name__)
    LLM_OK = True
except Exception as err:
    LLM_OK = False
    print("MODEL NOT REACHABLE. Sections 1 to 6 still work. Details:", err)

# COMMAND ----------

# 7b. Ask in plain words, for any service
state = {"fac": fac, "summary": summary, "evidence": evidence, "gaz": gaz}
specialty_codes = [c for c, _ in Counter(x for xs in fac["specialties_list"] for x in xs).most_common(250)]
dbutils.widgets.text("question", "My patient needs cataract surgery near Patna")

if LLM_OK:
    result, page = ask(state, dbutils.widgets.get("question"), llm, specialty_codes)
    displayHTML(page)
    if result is not None:
        print("understood as:", result["request"])
else:
    print("Model not reachable: use section 5 instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you present this
# MAGIC
# MAGIC - **Clinical review.** The patterns in `SERVICES` say what counts as stating a service and what
# MAGIC   counts as backing it up. They were written without clinical review.
# MAGIC - **Weights.** Penalties, grade values and `half_km` in `CONFIG` are judgment calls. Be ready to say why.
# MAGIC - **Corroboration is specific.** For cardiac surgery and burn care, a generic ICU, theatre or blood bank
# MAGIC   does not count as backing; only service-specific entries do. For emergency C-section and trauma those
# MAGIC   supports do count, because they are what the service needs.
# MAGIC - **Possibly merged** means only that a record lists unusually many phone numbers (top 5%). It is a weak signal.
# MAGIC - **Model-written definitions** (section 7) are not clinically reviewed and can be too broad or too narrow.
# MAGIC   They are shown on the answer so a reader can judge them. The eight predefined services do not use the model.
# MAGIC - **Known limits.** Straight-line distance, not travel time. Keyword rules miss unusual wording and
# MAGIC   can misread a sentence. Records are web-extracted claims; nothing here verifies them against the world.
# MAGIC - **Entry numbers** on the cards refer to the position in the de-duplicated field, not the raw array.