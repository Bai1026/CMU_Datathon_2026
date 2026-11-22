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

# 3b. Diagnostics: the distributions the thresholds in scoring.yaml were chosen from.
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
# Who runs the listed facilities matters for section 9: a web-collected list can under-count public facilities.
print("\noperator types:"); print(fac["operatorTypeId"].map(lambda v: "not stated" if cf.is_missing(v) else str(v)).value_counts().head(6).to_string())
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
        _svc_ok = (_r["known_key"] == _svc) if _svc else (not _r["service"] or _r["asks_for"] == "symptoms")
        _place_ok = (_place.lower() in _used.lower()) if _place else (not _where)
        _rows.append({"question": _q, "ok": _svc_ok and _place_ok, "language": _r["language"], "asks for": _r["asks_for"], "service": _r["service"],
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
# MAGIC ## 8. Optional: the web app
# MAGIC The same answers behind a simple screen, for people who will never open a notebook. The app holds no
# MAGIC logic: it loads the state saved below and calls the same `find_care` and `ask` as sections 5 and 7.
# MAGIC
# MAGIC This cell writes a folder that Databricks Apps can deploy: `app.py`, `app.yaml`, `requirements.txt`,
# MAGIC the library and settings, and the current state as data files. To put it online:
# MAGIC
# MAGIC 1. Compute > Apps > Create app > Create a custom app. Give it a name and create it.
# MAGIC 2. Deploy > From workspace > choose the folder printed below.
# MAGIC 3. For the question box: on the app, Edit > add a resource > Serving endpoint > the model named in
# MAGIC    `app.yaml`, permission "Can query". Without it the service and place fields still work.
# MAGIC
# MAGIC Re-run this cell and deploy again after changing the method or the data.

# COMMAND ----------

# 8. Write the app folder
try:
    APP_DIR = f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/care_finder_app"
    _info = cf.export_app(state, APP_DIR)
    print("app folder:", _info["app_dir"])
    print("code files:", ", ".join(_info["code_files"]))
    print(f"data: {_info['data_mb']} MB in {sum(_info['data_parts'].values())} files, largest {_info['largest_file_mb']} MB (the limit is 10 MB per file)")
except Exception as err:
    print("The app folder could not be written. Sections 0 to 7 are not affected. Details:", type(err).__name__, str(err)[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. District view: where is care out of reach, and where does it matter most?
# MAGIC Sections 5 and 7 answer one question for one patient. This section answers the planner's question
# MAGIC for the whole country, with the same evidence grades.
# MAGIC
# MAGIC **Supply.** For every PIN area (about 19,000), the distance from its centre to the nearest facility
# MAGIC that states each service, and to the nearest one whose claim is corroborated. A district is described
# MAGIC by its PIN areas, so one wrong coordinate cannot move it.
# MAGIC
# MAGIC **Need.** The NFHS-5 district survey, a second dataset: for each service, a few indicators chosen in
# MAGIC `scoring.yaml` (for emergency C-section: births in a facility, C-section rate, antenatal visits).
# MAGIC
# MAGIC **Priority = need x gap.** Gap is the share of a district's PIN areas with no facility stating the
# MAGIC service within reach.
# MAGIC
# MAGIC **Not finding a facility is not the same as there being none.** The dataset covers rural areas thinly.
# MAGIC So each district is put in one of four classes, and the last one says "we cannot tell":
# MAGIC
# MAGIC | Class | Meaning |
# MAGIC |---|---|
# MAGIC | Corroborated within reach | Most PIN areas are within reach of a facility whose claim is backed up |
# MAGIC | Claimed within reach | Within reach of a facility that states the service, without corroboration |
# MAGIC | No listing within reach | Not within reach, although many facilities are listed nearby: probably a real gap |
# MAGIC | Too few listings to tell | Not within reach, and few facilities of any kind are listed there: possibly a gap in the data |

# COMMAND ----------

# 9a. Access for every PIN area and district, joined with the NFHS-5 district survey
_nfhs = [r.tableName for r in spark.sql(f"SHOW TABLES IN {SRC}").collect() if "nfhs" in r.tableName.lower()]
survey_raw = spark.table(f"{SRC}.{_nfhs[0]}").toPandas() if _nfhs else None
print("survey table:", _nfhs[0] if _nfhs else "not found (the supply side below still works)", "| rows:", 0 if survey_raw is None else len(survey_raw))

dv = cf.district_table(state, survey_raw)
pins_access, districts = dv["pins"], dv["districts"]
for k, v in dv["report"].items():
    print(f"{k}: {v if not isinstance(v, list) else str(len(v)) + ' ' + str(v)}")
to_spark(districts).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.district_access")
print("saved", f"{OUT}.district_access", len(districts))

def show(fig):
    """Display a chart once (and close it, so the notebook does not draw it a second time)."""
    import matplotlib.pyplot as plt
    display(fig); plt.close(fig)

# COMMAND ----------

# 9b. How many districts are in each class, per service
display(cf.district_summary(districts))
show(cf.plot_district_status(districts))

# COMMAND ----------

# 9c. Maps: every PIN area, coloured by distance to the nearest facility. Dark = far.
_map_services = ["emergency_c_section", "nicu", "dialysis", "trauma"]
show(cf.plot_access_maps(pins_access, _map_services, tier="stated"))
show(cf.plot_access_maps(pins_access, _map_services, tier="corroborated"))

# COMMAND ----------

# 9d. Where need is high and the service is out of reach, ranked by need x gap.
#     First the candidate gaps (many facilities listed nearby, none states the service), then the
#     districts where the dataset is too thin to tell. The two lists call for different actions.
for _svc in cf.CONFIG["district_view"]["indicators"]:
    if districts.loc[districts["service"] == _svc, "priority"].notna().any():
        for _class, _n in ((cf.DISTRICT_STATUS[2], 10), (cf.DISTRICT_STATUS[3], 10)):
            _top = cf.district_priority(districts, _svc, top=_n, status=_class)
            if len(_top):
                print("\n" + cf.SERVICES[_svc]["label"] + " | " + _class)
                display(_top.drop(columns=["status"]))

# COMMAND ----------

# 9e. Is distance related to outcomes? Change in the outcome for each doubling of distance, alone and
#     with literacy, clean cooking fuel and electricity held constant (remote districts are also poorer).
_pairs = [("emergency_c_section", "institutional_birth_5y_pct", "Births in a health facility (%)"),
          ("emergency_c_section", "births_delivered_by_csection_5y_pct", "Births by C-section (%)"),
          ("emergency_c_section", "mothers_who_had_at_least_4_anc_visits_lb5y_pct", "Mothers with 4+ antenatal visits (%)"),
          ("nicu", "children_who_received_pnc_from_a_doctor_nurse_lhv_anm_midwi_pct", "Newborns checked by a health worker (%)")]
_pairs = [p for p in _pairs if p[1] in districts.columns]
if _pairs:
    _rel = [cf.access_relation(districts, s, o) for s, o, _ in _pairs]
    display(pd.DataFrame(_rel))
    for _r, (_, _, _label) in zip(_rel, _pairs):
        print(cf.relation_sentence(_r, _label))
    show(cf.plot_relation(districts, _pairs[0][0], _pairs[0][1], _pairs[0][2]))
else:
    print("No survey columns: this step needs the NFHS table.")

# COMMAND ----------

# MAGIC %md
# MAGIC **How to read 9e.** Two numbers per outcome: the raw relationship, and the relationship once
# MAGIC literacy, clean cooking fuel and electricity are held constant.
# MAGIC
# MAGIC In our run the raw relationships are negative (farther from a listed facility, fewer facility births,
# MAGIC fewer C-sections, fewer newborn checks) and they shrink to about zero with the controls. So distance
# MAGIC to a *listed* facility is mostly standing in for how developed a district is. Two consequences:
# MAGIC
# MAGIC - This dataset alone does not show that a closer facility would improve these outcomes.
# MAGIC - Where facilities are listed follows wealth and web presence, which is why "Too few listings to
# MAGIC   tell" is kept apart from a real gap.
# MAGIC
# MAGIC We report the result that does not support the obvious story, because the adjusted numbers are the
# MAGIC ones a planner should act on.

# COMMAND ----------

# 9f. The findings in sentences. Every number is counted from the tables above.
for _block in cf.district_findings(districts):
    print(_block + "\n")
_no_source = 1 - fac["has_first_party_source"].mean()
print(f"Data quality: {_no_source:.0%} of records have no source from the facility itself; "
      f"{fac['location_conflict'].mean():.1%} have coordinates that disagree with their PIN code; "
      f"{1 - fac['equipment_list'].map(len).gt(0).mean():.0%} list no equipment, so their claims cannot be corroborated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this suggests
# MAGIC The numbers and district names are in 9b, 9d and 9f. The four classes call for four different actions:
# MAGIC
# MAGIC 1. **Verify before building** ("Claimed within reach"). A facility there already says it offers the
# MAGIC    service. A call or a visit to confirm equipment and staffing is the cheapest way to add capacity
# MAGIC    that a referrer can rely on. Start with the districts where need is highest.
# MAGIC 2. **Check the candidate gaps on the ground** ("No listing within reach"). Many facilities are listed
# MAGIC    and none states the service, so these are the strongest candidates for a real gap. They are a short
# MAGIC    list for a field check, and after that for new capacity or referral transport. They are not proof:
# MAGIC    the dataset can miss a facility that exists.
# MAGIC 3. **Count facilities where the data is thin** ("Too few listings to tell"). This is the largest class
# MAGIC    for every emergency service, and it contains the districts with the worst survey indicators. Ranking
# MAGIC    them as gaps would send money to the places we know least about. They need a facility census first.
# MAGIC 4. **Fix the registry at the source.** Ask facilities for three things the method depends on: an
# MAGIC    equipment list, a PIN code that agrees with the coordinates, and a link to their own page. 9f shows
# MAGIC    how many records lack each.
# MAGIC 5. **Do not read distance in this dataset as cause.** 9e shows that it mostly reflects development.
# MAGIC
# MAGIC **Limits of this view.** Distances are straight-line. "Within reach" uses the distances set in
# MAGIC `scoring.yaml` (25 km for emergency services) and the classes move if those are changed. Need indicators
# MAGIC are proxies chosen without a public-health specialist. NFHS-5 was collected in 2019 to 2021 and some
# MAGIC districts have been split or renamed since (9a lists the ones that did not match). The facility list
# MAGIC comes from the web, so public facilities without a web presence may be missing (3b shows who runs the
# MAGIC listed ones).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Known limits
# MAGIC
# MAGIC - **Clinical review.** `services.yaml` says what counts as stating a service and what counts as backing it up.
# MAGIC   It was written without a clinician. So were the `target_minutes` values, which are drafts.
# MAGIC - **Weights.** Every number in `scoring.yaml` is a judgment call, not a fitted value.
# MAGIC - **Travel time** is a range estimated from straight-line distance and assumed speeds, not a route calculation.
# MAGIC - **Place aliases** are a first draft from general knowledge. Section 3c shows what did not match.
# MAGIC - **Languages.** Place names in nine Indian scripts work without a model (a list of names, then a
# MAGIC   letter-by-letter reading that the answer marks as automatic). Whole questions in other languages go
# MAGIC   through the model; section 7c measures how well. Answers are in English only. The names in Indian
# MAGIC   scripts in `location_aliases.yaml` were not checked by native speakers.
# MAGIC - **Model-written definitions** (section 7) are not reviewed and can be too broad or too narrow.
# MAGIC - **The app** cannot be made public: only people with an account in this workspace can open it,
# MAGIC   and on Free Edition it stops 24 hours after it was started.
# MAGIC - **Known limits.** Keyword rules miss unusual wording and can misread a sentence. Records are
# MAGIC   web-extracted claims; nothing here verifies them against the world. Entry numbers on the cards refer
# MAGIC   to the position in the de-duplicated field, not the raw array.
