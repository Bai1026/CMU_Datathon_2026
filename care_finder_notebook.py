# Databricks notebook source
# MAGIC %md
# MAGIC # Care Finder: where can this patient get this service, and why trust the answer?
# MAGIC
# MAGIC Two inputs (a service and a location), a ranked answer, and for every recommendation the exact
# MAGIC record entries it rests on.
# MAGIC
# MAGIC **This notebook holds no logic.** It loads data, calls `carefinder.py`, saves tables and shows results.
# MAGIC Everything that is a judgment call is in three YAML files in the same folder:
# MAGIC
# MAGIC | File | What it decides |
# MAGIC |---|---|
# MAGIC | `scoring.yaml` | distances, penalties, grade values, travel-time assumptions |
# MAGIC | `services.yaml` | what counts as evidence for each service (plain term lists) |
# MAGIC | `location_aliases.yaml` | other names for places |
# MAGIC
# MAGIC **How it decides.** `score = evidence x trust x proximity`, each between 0 and 1.
# MAGIC
# MAGIC **Run order.** Run all once (about a minute). After that, change the widgets at the top and re-run
# MAGIC only section 5 or 7. After editing a YAML file or `carefinder.py`, re-run from section 0.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Load the code and the settings

# COMMAND ----------

import importlib, json, os, sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.getcwd()                 # the folder this notebook is in
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import carefinder as cf
importlib.reload(cf)               # pick up edits to carefinder.py and to the YAML files
print(cf.load_settings())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the data

# COMMAND ----------

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
OUT = "workspace.care_finder"

raw = spark.table(f"{SRC}.facilities").toPandas()
pins_raw = spark.table(f"{SRC}.india_post_pincode_directory").toPandas()
print("facilities:", raw.shape, "| PIN directory:", pins_raw.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run the pipeline (once)
# MAGIC Clean the records, resolve each location, grade the evidence for every facility and service,
# MAGIC and build the place-name index.

# COMMAND ----------

try:
    state = cf.build_state(raw, pins_raw)
except ValueError as err:          # PIN directory not usable: continue without it
    print("WARNING: continuing without the PIN directory:", err)
    state = cf.build_state(raw, None)

fac, summary, evidence, gaz, pins = (state[k] for k in ("fac", "summary", "evidence", "gaz", "pins"))
if pins is not None:
    print(f"PIN directory: {pins['pincode'].nunique():,} PIN codes, {pins['lat'].notna().mean():.0%} of rows have usable coordinates")
print(f"{len(fac):,} facilities | {len(summary):,} facility-service pairs with evidence | {len(evidence):,} cited entries")
print("duplicates:", cf.PREP_STATS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data quality and evidence overview
# MAGIC Use these numbers in the write-up.

# COMMAND ----------

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

# 3b. Diagnostics. Paste this cell's output when asking for help tuning the method.
def q(s, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    s = pd.Series(s).dropna().astype(float)
    return {f"p{int(x * 100)}": round(float(s.quantile(x)), 1) for x in qs} if len(s) else {}

cut = max(cf.CONFIG["merged_min_phones"], float(fac["n_phones"].quantile(cf.CONFIG["merged_phone_quantile"])))
print("phones per record:", q(fac["n_phones"]), "| flagged as possibly merged at >=", cut)
print("record has a PIN:", f"{fac['pin'].notna().mean():.1%}", "| PIN centroid known:", f"{fac['pin_known'].mean():.1%}",
      "| PIN region known:", f"{fac['pref_radius'].notna().mean():.1%}")
print("km, coordinates to PIN centroid:", q(fac["coord_pin_km"]))
print("km, coordinates to PIN-region centre:", q(fac["coord_region_km"]), "| region radius km:", q(fac["pref_radius"]))
print("\nhow each location was chosen:"); print(fac["loc_source"].value_counts().to_string())
print("\nrecords sharing name + PIN with another record:", int((fac["name"].map(cf._norm) + "|" + fac["pin"].fillna("")).duplicated(keep=False).sum()))
print("records with an equipment list:", f"{fac['equipment_list'].map(len).gt(0).mean():.1%}")
print("\nfacility types:"); print(fac["facilityTypeId"].value_counts(dropna=False).head(8).to_string())
_dom = Counter(cf.domain_of(u) for urls in fac["source_url_list"] for u in urls)
print("\ntop source domains:", [(d, n) for d, n in _dom.most_common(25) if d])
# The evidence terms are English. Text in another script can never count as evidence, so this says how much there is.
print("\ntext that is not in Latin letters:"); print(cf.script_report(fac).to_string(index=False))

# COMMAND ----------

# 3c. Place names: how much of location_aliases.yaml matches this data, and what it may be missing.
rep = gaz.alias_report()
print(f"alias groups: {rep['groups']} | matched to a name in the data: {rep['matched']} | matched nothing: {len(rep['unmatched'])}")
print("groups that matched nothing (check the spelling, or the place is simply not in the data):")
print("  " + ", ".join(rep["unmatched"]))
print("\nPairs of names in the data that look like two spellings of one place and are not yet in the alias file.")
print("Check each one; if it is right, add it to location_aliases.yaml as   - names: [A, B]")
display(pd.DataFrame(gaz.suggest_aliases(limit=100)))

# COMMAND ----------

# 3d. Place names that are not typed in English. No model is used here.
#     Listed names come from other_scripts in location_aliases.yaml; the rest are read letter by letter.
print(f"places with names in other scripts: {rep['other_script_places']} ({rep['other_script_names']} names)",
      "| English name not found in the data:", rep["other_script_unmatched"] or "none")
_places = ["पटना", "गया", "मुजफ्फरपुर", "सीतामढ़ी", "मधुबनी", "औरंगाबाद", "औरंगाबाद, बिहार", "गया के पास", "८००००१",
           "দার্জিলিং", "மதுரை", "వరంగల్", "ಮೈಸೂರು", "കൊല്ലം", "સુરત", "ਬਠਿੰਡਾ", "କଟକ", "لکھنؤ", "Purnéa", "बिहार"]
display(pd.DataFrame([{"typed": t, "status": r["status"], "used": r["candidates"][0]["label"] if r["candidates"] else "",
                       "note shown to the user": r["note"]} for t in _places for r in [gaz.resolve(t)]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save the state as tables, and load it back
# MAGIC Six tables hold everything a query needs. `load_state()` is what an app calls at start-up
# MAGIC instead of re-running the pipeline.

# COMMAND ----------

from pyspark.sql import types as T

def to_spark(pdf):
    """pandas -> Spark with an explicit schema. Anything that is not a number or a boolean is stored as text."""
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
                    vals.append(None if cf.is_missing(v) else str(v))
            pdf[c] = pd.Series(vals, index=pdf.index, dtype=object)
            fields.append(T.StructField(c, T.StringType()))
    return spark.createDataFrame(pdf, schema=T.StructType(fields))


def save_state(state):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT}")
    for name, pdf in cf.state_to_tables(state).items():
        to_spark(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
        print("saved", f"{OUT}.{name}", len(pdf))


def load_state():
    """Rebuild the query state from the saved tables. No pipeline run, no source data needed."""
    return cf.state_from_tables({name: spark.table(f"{OUT}.{name}").toPandas() for name in cf.TABLES})


save_state(state)
loaded = load_state()
a, _ = cf.find_care(state, "dialysis", "Jaipur")
b, _ = cf.find_care(loaded, "dialysis", "Jaipur")
same = (a is None and b is None) or (a is not None and b is not None and a["verdict"] == b["verdict"]
                                     and [x["unique_id"] for x in a["top"]] == [x["unique_id"] for x in b["top"]])
print("answer from the saved tables matches the answer from memory:", same)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Find care
# MAGIC Set the two widgets at the top of the notebook, then run this cell.

# COMMAND ----------

dbutils.widgets.dropdown("service", "dialysis", sorted(cf.SERVICES))
dbutils.widgets.text("location", "Jaipur")

result, page = cf.find_care(state, dbutils.widgets.get("service"), dbutils.widgets.get("location"))
displayHTML(page)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validation: check the evidence grades by hand
# MAGIC Open each row's source URLs, decide whether the facility really offers the service, and fill
# MAGIC `truly_offers` with yes / no / unclear. Then run `cf.tier_precision(labeled)`.
# MAGIC A grade that means something shows strong > moderate > weak.

# COMMAND ----------

sample = cf.validation_sample(fac, summary, per_cell=6)
display(sample)
# labeled = <the sample with truly_offers filled in, as a pandas DataFrame>
# display(cf.tier_precision(labeled).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optional: plain-language questions and any service
# MAGIC A Databricks-hosted model does two small jobs: it splits a sentence into service and location, and for
# MAGIC a service that is not in `services.yaml` it writes the term lists, in that file's format. It never
# MAGIC ranks, scores or quotes. What it wrote is shown on the answer and saved to `services_generated.yaml`
# MAGIC for review; move an entry into `services.yaml` once someone has checked it.
# MAGIC
# MAGIC The question can be in any language. The model returns the service in English and the place in Latin
# MAGIC letters, and the answer shows that reading on top so that a wrong translation is visible. The answer
# MAGIC itself stays in English, and quotes from the records are never translated.

# COMMAND ----------

# 7a. Connect to a model
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# Optional, for testing only: your own OpenAI-compatible endpoint. Never paste a key into this notebook,
# it lives in a Git folder. Store the key as a Databricks secret and read it here, for example:
#   EXTERNAL = {"base_url": "https://api.openai.com/v1", "model": "<model name>",
#               "api_key": dbutils.secrets.get("care_finder", "llm_key")}
# The submitted version should leave this as None so that anyone can re-run the notebook.
EXTERNAL = None
_llm_route = None

def _text(content):
    if isinstance(content, list):   # some models return a list of parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)

def _via_external(prompt):
    import requests
    r = requests.post(f"{EXTERNAL['base_url'].rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {EXTERNAL['api_key']}", "Content-Type": "application/json"},
                      json={"model": EXTERNAL["model"], "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}, timeout=90)
    r.raise_for_status()
    return _text(r.json()["choices"][0]["message"]["content"])

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
    """Send one prompt to the model. Tries the available routes and remembers the one that works."""
    global _llm_route
    errors = []
    for route in ([_llm_route] if _llm_route else ([_via_external] if EXTERNAL else []) + [_via_sdk, _via_rest, _via_sql]):
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
    print("MODEL NOT REACHABLE. Sections 0 to 6 still work. Details:", err)

# COMMAND ----------

# 7b. Ask in plain words, for any service
dbutils.widgets.text("question", "My patient needs cataract surgery near Patna")

if LLM_OK:
    result, page = cf.ask(state, dbutils.widgets.get("question"), llm)
    displayHTML(page)
    if result is not None:
        print("understood as:", result["request"])
    generated = [k for k, v in cf.SERVICES.items() if v.get("generated")]
    if generated:
        print("model-written definitions this session:", generated, "| saved for review in", cf.GENERATED_FILE)
        save_state(state)          # keep the new service's evidence and definition in the tables too
else:
    print("Model not reachable: use section 5 instead.")

# COMMAND ----------

# 7c. Language check: does the model read questions in other languages correctly?
#     Only the reading is tested (one model call per question); nothing is ranked and nothing is saved.
#     expected service "" = the question names no service (symptoms only), and the model must not guess one.
LANGUAGE_TESTS = [
    ("My patient needs dialysis near Gaya",                          "dialysis",            "Gaya"),
    ("गया के पास डायलिसिस कहाँ होता है?",                              "dialysis",            "Gaya"),
    ("Patna ke paas dialysis kahan hota hai",                        "dialysis",            "Patna"),
    ("मुजफ्फरपुर में नवजात शिशु के लिए आईसीयू चाहिए",                    "nicu",                "Muzaffarpur"),
    ("नाशिक जवळ सिझेरियन प्रसूतीसाठी रुग्णालय हवे आहे",                 "emergency_c_section", "Nashik"),
    ("শিলিগুড়ির কাছে এমআরআই কোথায় করা যায়?",                          "mri",                 "Siliguri"),
    ("மதுரை அருகில் சிடி ஸ்கேன் எங்கே செய்யலாம்?",                    "ct_scan",             "Madurai"),
    ("వరంగల్ దగ్గర డయాలసిస్ ఎక్కడ చేస్తారు?",                          "dialysis",            "Warangal"),
    ("لکھنؤ کے قریب دل کی سرجری کہاں ہوتی ہے؟",                        "cardiac_surgery",     "Lucknow"),
    ("800001 के पास बर्न यूनिट",                                      "burn_care",           "800001"),
    ("巴特那附近哪裡可以洗腎？",                                         "dialysis",            "Patna"),
    ("डायलिसिस कहाँ होता है?",                                        "dialysis",            ""),
    ("मेरी माँ के पैर सूज गए हैं, पटना में कहाँ ले जाऊँ?",                  "",                    "Patna"),
]
if LLM_OK:
    _rows = []
    for _q, _svc, _place in LANGUAGE_TESTS:
        try:
            _r = cf.parse_request(llm, _q)
        except Exception as err:
            _rows.append({"question": _q, "ok": False, "problem": f"model reply not usable: {str(err)[:80]}"}); continue
        _where = _r["location"] or _r["location_as_written"]
        _loc = gaz.resolve(_where) if _where else {"candidates": [], "note": ""}
        _used = _loc["candidates"][0]["label"] if _loc["candidates"] else ""
        _svc_ok = (_r["known_key"] == _svc) if _svc else (not _r["service"])
        _place_ok = (_place.lower() in _used.lower()) if _place else (not _where)
        _rows.append({"question": _q, "ok": _svc_ok and _place_ok, "language": _r["language"], "service": _r["service"],
                      "key": _r["known_key"], "expected key": _svc, "place (model)": _r["location"],
                      "place as written": _r["location_as_written"], "place used": _used,
                      "problem": ", ".join(x for x, good in (("service", _svc_ok), ("place", _place_ok)) if not good)})
    _t = pd.DataFrame(_rows)
    print(f"read correctly: {int(_t['ok'].sum())} of {len(_t)} | model: {EXTERNAL['model'] if EXTERNAL else LLM_MODEL}")
    display(_t)
else:
    print("Model not reachable: section 3d shows what works without it.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before you present this
# MAGIC
# MAGIC - **Clinical review.** `services.yaml` says what counts as stating a service and what counts as backing it up.
# MAGIC   It was written without a clinician. So were the `target_minutes` values, which are drafts.
# MAGIC - **Weights.** Every number in `scoring.yaml` is a judgment call. Be ready to say why.
# MAGIC - **Travel time** is a range estimated from straight-line distance and assumed speeds, not a route calculation.
# MAGIC - **Place aliases** are a first draft from general knowledge. Section 3c shows what did not match.
# MAGIC - **Languages.** Place names in nine Indian scripts work without a model (a list of names, then a
# MAGIC   letter-by-letter reading that the answer marks as automatic). Whole questions in other languages go
# MAGIC   through the model; section 7c measures how well. Answers are in English only. The names in Indian
# MAGIC   scripts in `location_aliases.yaml` were not checked by native speakers.
# MAGIC - **Model-written definitions** (section 7) are not reviewed and can be too broad or too narrow.
# MAGIC - **Known limits.** Keyword rules miss unusual wording and can misread a sentence. Records are
# MAGIC   web-extracted claims; nothing here verifies them against the world. Entry numbers on the cards refer
# MAGIC   to the position in the de-duplicated field, not the raw array.
