import json, os, sys
import pandas as pd, numpy as np
import carefinder as cf

COLS = """unique_id source_types source_ids source_content_id name organization_type content_table_id phone_numbers officialPhone email websites officialWebsite yearEstablished acceptsVolunteers facebookLink address_line1 address_line2 address_line3 address_city address_stateOrRegion address_zipOrPostcode address_country address_countryCode countries facilityTypeId operatorTypeId affiliationTypeIds description area numberDoctors capacity specialties procedure equipment capability recency_of_page_update distinct_social_media_presence_count affiliated_staff_presence custom_logo_presence number_of_facts_about_the_organization post_metrics_most_recent_social_media_post_date post_metrics_post_count engagement_metrics_n_followers engagement_metrics_n_likes engagement_metrics_n_engagements source coordinates latitude longitude cluster_id source_urls""".split()
assert len(COLS) == 51
# Usage: python test_carefinder.py [sample_rows.tsv] [answer.html]
# sample_rows.tsv holds five real rows of the facilities table, tab-separated, no header.
_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "sample_rows.tsv")
rows = [l.rstrip("\n").split("\t") for l in open(SAMPLE, encoding="utf-8") if l.strip()]
raw = pd.DataFrame(rows, columns=COLS).astype(object)
raw = raw.where(raw != "null", None)            # the table has real NULLs
raw["latitude"] = raw["latitude"].astype(float); raw["longitude"] = raw["longitude"].astype(float)

def J(x): return json.dumps(x)
def synth(uid, name, city, state, pin, lat, lon, **kw):
    r = {c: None for c in COLS}
    r.update(unique_id=uid, name=name, address_city=city, address_stateOrRegion=state, address_zipOrPostcode=pin,
             latitude=lat, longitude=lon, facilityTypeId="hospital", description="", phone_numbers=J(["+911111111111"]),
             source_ids=J(["a"]), specialties=J([]), procedure=J([]), equipment=J([]), capability=J([]),
             source_urls=J(["https://example-hospital.in/about"]), officialWebsite="example-hospital.in", cluster_id=uid)
    r.update(kw); return r
extra = pd.DataFrame([
    synth("s1", "Negation General", "Ahmedabad", "Gujarat", "380007", 23.02, 72.57,
          capability=J(["No MRI on site; patients are referred to a nearby centre", "Dialysis is not available"])),
    synth("s2", "Imaging Plus", "Ahmedabad", "Gujarat", "380007", 23.03, 72.55,
          procedure=J(["MRI scans of brain and spine"]), equipment=J(["1.5 Tesla MRI scanner"])),
    synth("s3", "Kidney Care", "Ahmedabad", "Gujarat", "380007", 23.00, 72.60,
          procedure=J(["Offers hemodialysis"]), equipment=J(["12 dialysis machines", "RO water plant"])),
    synth("s4", "City Maternity", "Aurangabad", "Bihar", "824101", 24.75, 84.37,
          procedure=J(["Performs caesarean section (LSCS)"]),
          capability=J(["24/7 emergency care", "Modular operation theatre", "Blood bank on site"]),
          specialties=J(["gynecologyAndObstetrics", "anesthesia"])),
    synth("s5", "Day Maternity", "Aurangabad", "Maharashtra", "431001", 19.88, 75.34,
          procedure=J(["Performs C-section deliveries"]), capability=J(["Operation theatre", "Blood storage unit"])),
    synth("s6", "Heartburn Clinic", "Pune", "Maharashtra", "411045", 18.56, 73.78,
          capability=J(["Treats heartburn and acidity"]), specialties=J(["gastroenterology"])),
    synth("s8", "Conflict City Matches PIN", "Ahmedabad", "Gujarat", "380007", 28.61, 77.21),
    synth("s9", "Conflict City Differs", "New Delhi", "Delhi", "380007", 28.61, 77.21),
    synth("s10", "Shifted Columns Hospital", "Ahmedabad", "Gujarat", "380007", 23.01, 72.56,
          officialWebsite='["Cathlab in HeartCare Center","Fully-equipped NICU","Neonatal incubators"]',
          source_urls=J(['["not a url"]', "https://ok.example.in/x", "[::bad"])),
    synth("s11", "Claim Only Dialysis", "Mehsana", "Gujarat", None, 23.90, 72.60, capability=J(["Dialysis available"])),
    synth("s12", "Nephro Clinic", "Ahmedabad", "Gujarat", "380007", 23.01, 72.57, specialties=J(["nephrology"])),
    synth("s13", "Generic Burns", "Ahmedabad", "Gujarat", "380007", 23.02, 72.58,
          capability=J(["Treats burns", "ICU available", "Modular operation theatre", "Blood bank"])),
    synth("s14", "Heart Centre", "Ahmedabad", "Gujarat", "380007", 23.02, 72.56,
          procedure=J(["CABG and valve replacement"]), equipment=J(["Cath lab", "Heart-lung machine"])),
    synth("s15", "Region OK", "Hyderabad", "Telangana", "500001", 17.95, 78.50),
    synth("s16", "Region Dropped PIN", "Hyderabad", "Telangana", "500099", 17.41, 78.49),
    synth("s17", "Region Conflict", "Hyderabad", "Telangana", "500001", 28.60, 77.20),
    synth("s19", "Tiny NICU Clinic", "Bhuj", "Gujarat", "370001", 23.25, 69.67, facilityTypeId="clinic", capability=J(["NICU available"])),
    synth("s7", "No Location Hospital", "Nowhere", None, None, None, None, procedure=J(["Dialysis"])),
])
raw = pd.concat([raw, extra], ignore_index=True)
# unique_id is not unique in the real table: add an exact copy, and a different record reusing an id
copy_row = raw[raw["name"] == "Shaurya Hospital"]
reuse = pd.DataFrame([synth("s3", "Kidney Care Annex", "Ahmedabad", "Gujarat", "380007", 23.00, 72.60, procedure=J(["Peritoneal dialysis"]))])
same_place = pd.DataFrame([synth("s18", "Kidney Care", "Ahmedabad", "Gujarat", "380007", 23.001, 72.601, capability=J(["Dialysis unit"]))])
raw = pd.concat([raw, copy_row, reuse, same_place], ignore_index=True)

pins_raw = pd.DataFrame([
    # deliberately messy headers and values, like the public directory
    ("Bathinda H.O", "151001", "Bathinda", "PUNJAB", "30.2110", "74.9455"),
    ("Chengannur S.O", "690509", "Alappuzha", "KERALA", "9.3180", "76.6150"),
    ("Kollakadavu B.O", "690509", "Alappuzha", "KERALA", "9.2700", "76.5900"),
    ("Paldi S.O", "380007", "Ahmedabad", "GUJARAT", "23.0120", "72.5630"),
    ("Bhuj H.O", "370001", "Kachchh", "GUJARAT", "23.2500", "69.6700"),
    ("Parvati S.O", "411009", "Pune", "MAHARASHTRA", "18.4850", "73.8560"),
    ("Baner S.O", "411045", "Pune", "MAHARASHTRA", "18.5600", "73.7800"),
    ("Aurangabad H.O", "824101", "Aurangabad", "BIHAR", "24.7500", "84.3700"),
    ("Aurangabad H.O", "431001", "Aurangabad", "MAHARASHTRA", "19.8800", "75.3400"),
    ("Broken B.O", "431001", "Aurangabad", "MAHARASHTRA", "NA", "NA"),
    ("Ahmedabad B.O", "781001", "Kamrup", "ASSAM", "26.1800", "91.7500"),          # same name, other state, post office only
    ("Wrong Coords S.O", "500099", "Hyderabad", "TELANGANA", "20.0000", "78.5000"),  # directory error, 290 km off
] + [(f"Hyd Office {i} S.O", "500001" if i < 3 else f"5000{10 + i}", "Hyderabad", "TELANGANA",
      f"{17.40 + 0.01 * (i % 7 - 3):.4f}", f"{78.50 + 0.01 * (i % 5 - 2):.4f}") for i in range(40)],
    columns=["OfficeName", "Pincode", "District", "StateName", "Latitude", "Longitude"])

pins = cf.standardize_pins(pins_raw)
fac = cf.prepare_facilities(raw, pins)
summary, evidence = cf.build_evidence(fac)
gaz = cf.Gazetteer(fac, pins)
state0 = {"fac": fac, "summary": summary, "evidence": evidence, "gaz": gaz, "pins": pins}
F = fac.drop_duplicates("name").set_index("name")
_rank = {"weak": 0, "moderate": 1, "strong": 2}
T = {}
for n, sv, tr in summary.merge(fac[["unique_id", "name"]]).loc[:, ["name", "service", "tier"]].itertuples(index=False):
    if (n, sv) not in T or _rank[tr] > _rank[T[(n, sv)]]:      # two records can share a name: keep the best grade
        T[(n, sv)] = tr

# --- duplicate rows and ids
assert fac["unique_id"].is_unique and cf.PREP_STATS["exact_duplicate_rows_dropped"] == 1, cf.PREP_STATS
assert cf.PREP_STATS["rows_sharing_an_id_with_different_content"] == 2 and (fac["name"] == "Shaurya Hospital").sum() == 1
assert set(fac.loc[fac["source_unique_id"] == "s3", "unique_id"]) == {"s3", "s3#2"}
assert fac["name"].value_counts()["Kidney Care"] == 2

# --- trust flags on the real sample rows
sj = F.loc["Sanjivani Multi Speciality Hospital"]
assert sj.coords_outside_india and sj.loc_source.startswith("PIN centroid") and abs(sj.lat - 9.29) < 0.1, sj.loc_source
assert sj.state == "Kerala" and sj.district == "Alappuzha", (sj.state, sj.district)
assert sj.possible_merged and sj.n_phones == 8 and sj.n_source_ids == 6, (sj.n_phones, sj.n_source_ids)
assert sj.has_first_party_source
ps = F.loc["Puresmiles Dental Clinic and Implant Center"]
assert not ps.location_conflict, ps.loc_source   # 11 km from its PIN centroid: inside the 50 km tolerance
c1 = F.loc["Conflict City Matches PIN"]; assert c1.location_conflict and c1.loc_source.startswith("PIN centroid") and c1.trust == 0.85, (c1.loc_source, c1.trust)
c2 = F.loc["Conflict City Differs"]; assert c2.location_conflict and c2.loc_source.startswith("coordinates (PIN disagrees"), c2.loc_source
sh = F.loc["Shaurya Hospital"]
assert not sh.location_conflict and sh.loc_source == "coordinates, consistent with PIN" and sh.trust == 1.0, (sh.loc_source, sh.trust, sh.trust_reasons)
assert F.loc["No Location Hospital"].loc_source == "none"

assert cf.domain_of('["Cathlab in HeartCare Center","ICU available"]') is None
assert cf.domain_of("[::bad") is None and cf.domain_of("smsh.in") == "smsh.in" and cf.domain_of("https://www.Smsh.in/dept/x") == "smsh.in"
assert cf.domain_of("http://example.com:8080/a") == "example.com" and cf.domain_of("just some words") is None
sc = F.loc["Shifted Columns Hospital"]
assert not sc.has_first_party_source and sc.malformed_record and sc.trust == 0.70, (sc.trust, sc.trust_reasons)
assert not F.loc["Shaurya Hospital"].malformed_record and fac["malformed_record"].sum() == 1

# --- PIN region logic
ro = F.loc["Region OK"]; assert ro.coord_pin_km > 50 and not ro.location_conflict and ro.loc_source == "coordinates, consistent with PIN region", (ro.coord_pin_km, ro.loc_source, ro.pref_radius)
rd = F.loc["Region Dropped PIN"]; assert not rd.pin_known and rd.loc_source == "coordinates, consistent with PIN region" and rd.district == "Hyderabad", (rd.pin_known, rd.loc_source)
rc = F.loc["Region Conflict"]; assert rc.location_conflict and rc.loc_source.startswith("PIN centroid") and abs(rc.lat - 17.4) < 0.1, rc.loc_source
assert "PIN 500001" in rc.trust_reasons[0][1]

# --- region radius is capped, and a clinic claiming a hospital-level service gets a note
noisy = pd.DataFrame({"pincode": ["700%03d" % i for i in range(40)], "lat": [22.5] * 30 + [30.0] * 10, "lon": [88.3] * 40,
                      "district": "X", "state": "Y", "office": "Z"})
# a quarter of this region's post offices have wrong coordinates 830 km away: the median ignores them
assert cf.prefix_regions(noisy)["pref_radius"].iloc[0] == cf.CONFIG["region_min_km"] + cf.CONFIG["region_margin_km"]
sparse_region = pd.DataFrame({"pincode": ["194%03d" % i for i in range(40)], "lat": np.linspace(32.0, 35.5, 40), "lon": [77.5] * 40,
                              "district": "X", "state": "Y", "office": "Z"})
assert cf.prefix_regions(sparse_region)["pref_radius"].iloc[0] > 200      # a genuinely spread-out region gets a wide radius
tiny = summary[(summary["unique_id"] == "s19") & (summary["service"] == "nicu")].iloc[0]
assert any("facility type is 'clinic'" in n for n in tiny["notes"]), tiny["notes"]

# --- evidence tiers
assert T[("Generic Burns", "burn_care")] == "moderate"          # ICU and theatre no longer corroborate burn care
assert T[("Heart Centre", "cardiac_surgery")] == "strong"
assert T[("Nephro Clinic", "dialysis")] == "weak" and T[("Claim Only Dialysis", "dialysis")] == "moderate"
assert T[("Shaurya Hospital", "nicu")] == "strong"
assert T[("Sanjivani Multi Speciality Hospital", "nicu")] == "moderate"
assert T[("Sanjivani Multi Speciality Hospital", "ct_scan")] == "strong"
assert T[("Sanjivani Multi Speciality Hospital", "dialysis")] == "moderate"
assert T[("Shaurya Hospital", "trauma")] in ("strong", "moderate"), T[("Shaurya Hospital", "trauma")]
assert ("Negation General", "mri") not in T and ("Negation General", "dialysis") not in T, {k: v for k, v in T.items() if k[0] == "Negation General"}
assert T[("Imaging Plus", "mri")] == "strong"
assert T[("Kidney Care", "dialysis")] == "strong"
assert T[("City Maternity", "emergency_c_section")] == "strong"
assert T[("Day Maternity", "emergency_c_section")] == "moderate"      # no 24/7 statement
assert ("Heartburn Clinic", "burn_care") not in T
assert not any(k[0].startswith("Dr Satish") and k[1] in ("nicu", "dialysis", "mri") for k in T)
assert not any(k[0].startswith("Puresmiles") and k[1] != "x" and T[k] == "strong" for k in T), [k for k in T if k[0].startswith("Pure")]

# --- every cited text is an exact entry of the record
items = {r["unique_id"]: {(f, i): t for f, i, t in cf.facility_items(r)} for r in fac.to_dict("records")}
specs = dict(zip(fac.unique_id, fac.specialties_list))
for e in evidence.itertuples():
    src = specs[e.unique_id][e.idx] if e.field == "specialties" else items[e.unique_id][(e.field, e.idx)]
    assert src == e.text

# --- location resolution
assert gaz.resolve("380007")["candidates"][0]["state"] == "Gujarat"
assert gaz.resolve("23.01, 72.56")["status"] == "ok"
assert gaz.resolve("59.9, -38.2")["status"] == "not_found"
a = gaz.resolve("Aurangabad"); assert a["status"] == "ambiguous" and len(a["candidates"]) == 2, a
b = gaz.resolve("Aurangabad, Bihar"); assert b["status"] == "ok" and b["candidates"][0]["state"] == "Bihar", b
assert gaz.resolve("Atlantis")["status"] == "not_found"
assert gaz.resolve("999999")["status"] == "not_found"
assert gaz.resolve("Ahmedabad")["candidates"][0]["kind"] == "district"
for q in ("near Ahmedabad", "in Ahmedabad district", "around the city of Ahmedabad", "somewhere near Ahmedabad", "near 380007"):
    r_ = gaz.resolve(q); assert r_["status"] == "ok" and r_["candidates"][0]["state"] == "Gujarat", (q, r_)
assert gaz.resolve("near Aurangabad, Bihar")["candidates"][0]["state"] == "Bihar"
assert gaz.resolve("near the district")["status"] == "not_found"

# --- ranking
res, page = cf.find_care(state0, "nicu", "Ahmedabad")
assert res["top"][0]["name"] == "Shaurya Hospital" and res["top"][0]["tier"] == "strong"
assert abs(res["top"][0]["score"] - res["top"][0]["evidence_value"] * res["top"][0]["trust"] * res["top"][0]["proximity"]) < 1e-9
res2, _ = cf.find_care(state0, "dialysis", "690509")
assert res2["top"][0]["name"].startswith("Sanjivani") and "No corroborated option" in res2["verdict"], res2["verdict"]
res3, _ = cf.find_care(state0, "ct_scan", "Bhuj")
assert not res3["top"] and "No facility with evidence" in res3["verdict"], res3["verdict"]
assert "Nearest option beyond the search radius" in res3["alternatives"]
assert res2["counts"]["without_usable_location"] == 1
none, msg = cf.find_care(state0, "mri", "Atlantis"); assert none is None and "Could not find" in msg
none, msg = cf.find_care(state0, "teleportation", "Ahmedabad"); assert none is None and "Unknown service" in msg
res4, _ = cf.find_care(state0, "emergency_c_section", "Aurangabad, Maharashtra")
assert res4["top"][0]["name"] == "Day Maternity"

r5, _ = cf.find_care(state0, "dialysis", "23.90, 72.60")
assert r5["top"][0]["name"] == "Claim Only Dialysis" and "nearest corroborated option is Kidney Care" in r5["verdict"], r5["verdict"]
r6, _ = cf.find_care(state0, "dialysis", "Ahmedabad")
names6 = [t["name"] for t in r6["top"]]
assert names6[0] == "Kidney Care" and "Nephro Clinic" not in names6 and names6.count("Kidney Care") == 1, names6
assert any("2 records with this name and PIN" in n for n in r6["top"][0]["notes"]) and r6["top"][0]["tier"] == "strong", r6["top"][0]["notes"]
assert "Kidney Care Annex" in names6
assert "Nearby, but the service is only implied" not in r6["alternatives"]      # three records state it, so no implied-only filler
r6b, _ = cf.find_care(state0, "dialysis", "Ahmedabad", k=5)
assert r6b["alternatives"]["Nearby, but the service is only implied"]["name"] == "Nephro Clinic"
assert gaz.resolve("Ahmedabad")["status"] == "ok"            # a same-named post office in Assam is not an ambiguity
r7, _ = cf.find_care(state0, "emergency_c_section", "690509")
assert "only imply" in r7["verdict"] and r7["top"][0]["tier"] == "weak", r7["verdict"]

# --- language model helpers, with a scripted fake model
codes = ["ophthalmology", "nephrology", "gynecologyAndObstetrics"]
calls = []
def fake(prompt):
    calls.append(prompt)
    if "Their message" in prompt:
        prompt = prompt.split('The message:\n"')[1].split('"\n')[0]      # look only at the user's sentence
        if "मोतियाबिंद" in prompt:      # Hindi question; this model ignores the "answer in English" rule
            return json.dumps({"language": "Hindi", "service": "मोतियाबिंद ऑपरेशन", "known_key": None, "location": "Ahmedabad", "location_as_written": "अहमदाबाद"})
        if "घुटने" in prompt:
            return json.dumps({"language": "Hindi", "service": "घुटने का ऑपरेशन", "known_key": None, "location": "Ahmedabad", "location_as_written": "अहमदाबाद"})
        if "डायलिसिस" in prompt:        # Hindi question, the model's English spelling of the place is not in the data
            return json.dumps({"language": "Hindi", "service": "dialysis", "known_key": "dialysis", "location": "Karnavati Nagar", "location_as_written": "अहमदाबाद"})
        if "எம்ஆர்ஐ" in prompt:          # Tamil question about a place that is nowhere in the data
            return json.dumps({"language": "Tamil", "service": "MRI scan", "known_key": "mri", "location": "Atlantis", "location_as_written": "அட்லாண்டிஸ்"})
        if "cataract" in prompt:
            return 'Sure! ```json\n{"service": "cataract surgery", "known_key": null, "location": "Ahmedabad"}\n```'
        if "dialysis" in prompt:
            return '{"service": "dialysis", "known_key": "dialysis", "location": "Ahmedabad"}'
        if "nowhere" in prompt:
            return '{"service": "MRI", "known_key": "mri", "location": ""}'
        return "I cannot help with that."
    return json.dumps({"label": "Cataract surgery", "urgency": "planned",
                       "direct_terms": ["Cataract Surgery", "phaco", "surgery", "care", "IOL implantation"],
                       "equipment_terms": ["phaco machine", "operating microscope", "ICU"],
                       "related_terms": ["eye surgery", "phaco"], "specialties": ["ophthalmology", "madeUpCode"]})
eye = pd.DataFrame([synth("e1", "Eye Centre", "Ahmedabad", "Gujarat", "380007", 23.02, 72.57,
                          procedure=J(["Phacoemulsification cataract surgery"]), equipment=J(["Zeiss operating microscope"]),
                          specialties=J(["ophthalmology"])),
                    synth("e2", "Eye Clinic", "Ahmedabad", "Gujarat", "380007", 23.03, 72.57, specialties=J(["ophthalmology"]))])
fac2 = cf.prepare_facilities(pd.concat([raw, eye], ignore_index=True), pins)
s2, e2 = cf.build_evidence(fac2)
state = {"fac": fac2, "summary": s2, "evidence": e2, "gaz": cf.Gazetteer(fac2, pins), "pins": pins}
import os, tempfile
TMP = tempfile.mkdtemp()
n_before = len(cf.SERVICES)
ra, pa = cf.ask(state, "My patient needs cataract surgery near Ahmedabad", fake, codes, save_dir=TMP)
g = cf.SERVICES["custom_cataract_surgery"]
assert g["direct"] == ["cataract surgery", "phaco", "iol implantation"], g["direct"]       # generic words dropped
assert g["equipment"] == ["phaco machine", "operating microscope"] and g["specialties"] == ["ophthalmology"], g
assert g["related"] == ["eye surgery"] and g["generated"]["request"] == "cataract surgery", g
import tempfile, yaml as _yaml
saved = _yaml.safe_load(open(os.path.join(TMP, cf.GENERATED_FILE)))["services"]["custom_cataract_surgery"]
assert saved["direct"] == g["direct"] and "direct_re" not in saved, saved
assert ra["top"][0]["name"] == "Eye Centre" and ra["top"][0]["tier"] == "strong", (ra["top"][0]["name"], ra["top"][0]["tier"])
assert "Search definition written by an AI model" in pa and "Eye Clinic" in pa      # implied-only alternative shown
assert ra["urgency"] == "planned"
n_calls = len(calls); cf.ask(state, "cataract op near Ahmedabad please", fake, codes, save_dir=TMP); assert len(calls) == n_calls + 1   # definition reused
rb, pb = cf.ask(state, "dialysis around Ahmedabad", fake, codes, save_dir=TMP); assert rb["service"] == "dialysis" and "Search definition" not in pb
rc_, pc = cf.ask(state, "MRI nowhere", fake, codes, save_dir=TMP); assert rc_ is None and "does not say where" in pc
rd_, pd_ = cf.ask(state, "hello", fake, codes, save_dir=TMP); assert rd_ is None and "Could not interpret" in pd_
def bad(prompt): raise RuntimeError("endpoint down")
re_, pe = cf.ask(state, "anything", bad, codes, save_dir=TMP); assert re_ is None and "endpoint down" in pe
assert len(cf.SERVICES) == n_before + 1
del cf.SERVICES["custom_cataract_surgery"]

# --- alias file, misspellings, states
r_ = gaz.resolve("Amdavad"); assert r_["status"] == "ok" and r_["candidates"][0]["state"] == "Gujarat" and "another name for Ahmedabad" in r_["note"], r_
assert gaz.resolve("Alleppey")["candidates"][0]["label"].startswith("Alappuzha")
assert gaz.resolve("near Alleppey district")["status"] == "ok"
r_ = gaz.resolve("Ahmedabd"); assert r_["status"] == "ok" and "closest known name" in r_["note"], r_
r_ = gaz.resolve("Gujarat"); assert r_["status"] == "not_found" and "is a state" in r_["note"], r_
assert gaz.resolve("Aurangabad (Bihar)")["candidates"][0]["state"] == "Bihar"
r_ = gaz.resolve("Chhatrapati Sambhajinagar"); assert r_["status"] == "ok" and r_["candidates"][0]["state"] == "Maharashtra", r_
cf.ALIASES["places"].append({"names": ["Testville"], "within": "Ahmedabad"}); gaz._index()
r_ = gaz.resolve("Testville"); assert r_["status"] == "ok" and "district that contains it" in r_["note"], r_
rep_ = gaz.alias_report(); assert rep_["matched"] >= 4 and rep_["groups"] == len(cf.ALIASES["places"]), rep_["matched"]
assert isinstance(gaz.suggest_aliases(), list)
assert gaz.resolve("Baroda")["status"] == "not_found"          # neither spelling is in this small test data

# --- widening the search, sparse areas, travel time
rw, _ = cf.find_care(state0, "dialysis", "23.25, 69.67")
assert rw["widened"] and rw["max_km"] == 400 and rw["verdict"].startswith("Nothing within 200 km states this service"), rw["verdict"]
assert rw["sparse_note"].startswith("Only ")
assert rw["top"][0]["time_status"] is None and "by road" in rw["top"][0]["travel_text"]
rt, page_t = cf.find_care(state0, "trauma", "Ahmedabad")
assert rt["top"][0]["name"] == "Shaurya Hospital" and rt["top"][0]["time_status"] == "within" and rt["target_minutes"] == 60, rt["top"][0]["time_status"]
assert "60-minute target" in page_t and "by road" in page_t
assert cf.time_check(25, "trauma")["time_status"] == "uncertain" and cf.time_check(80, "trauma")["time_status"] == "beyond"
assert cf.time_check(2, "trauma")["travel_text"] == "roughly 5 min by road"

# --- the three ranking steps can be used separately
cands, counts = cf.score_candidates(state0, "dialysis", 23.02, 72.57)
picked = cf.pick(cands, "dialysis", k=2)
assert len(picked["top"]) == 2 and picked["radius_km"] == 200 and not picked["widened"] and "Kidney Care" in cf.verdict(picked, "dialysis")
assert (cands["score"].diff().dropna() <= 1e-12).all()

# --- save and reload the state: same answer from the tables as from memory
tables = cf.state_to_tables(state0)
assert set(tables) == set(cf.TABLES) and all(isinstance(v, str) for v in tables["facilities_clean"]["procedure_list"])
state1 = cf.state_from_tables({k: v.copy() for k, v in tables.items()})
for svc_, loc_ in (("dialysis", "Ahmedabad"), ("nicu", "380007"), ("emergency_c_section", "Aurangabad, Maharashtra")):
    ra_, pa_ = cf.find_care(state0, svc_, loc_); rb_, pb_ = cf.find_care(state1, svc_, loc_)
    assert ra_["verdict"] == rb_["verdict"] and [x["unique_id"] for x in ra_["top"]] == [x["unique_id"] for x in rb_["top"]] and pa_ == pb_, svc_

# --- exclusions, and settings validation
ex = cf.prepare_facilities(pd.DataFrame([synth("x1", "Obesity Clinic", "Ahmedabad", "Gujarat", "380007", 23.0, 72.5,
                                               procedure=J(["Gastric bypass surgery for weight loss"]))]), pins)
assert cf.assess(ex.iloc[0].to_dict(), "cardiac_surgery")["tier"] is None

# --- validation helpers
vs = cf.validation_sample(fac, summary, per_cell=2)
assert {"service", "tier", "truly_offers"} <= set(vs.columns) and len(vs)
vs["truly_offers"] = np.where(vs["tier"] == "strong", "yes", "no")
print(cf.tier_precision(vs))

# --- input that is not English ---------------------------------------------------------------
# names and digits
assert cf._norm("Puné") == "pune" and cf._norm("  Kochi – (Cochin) ") == "kochi cochin"
assert cf._norm("३८०००७") == "380007" and cf._norm("near अहमदाबाद") == "near अहमदाबाद"
assert cf._norm("क़") == cf._norm("क़")                    # one letter, two ways of typing it
assert cf.scripts_in("Patna पटना حیدرآباد") in (["Arabic", "Devanagari"], ["Devanagari", "Arabic"]) and cf.scripts_in("Patna 800001") == []
for nat_, eng_ in {"पटना": "patna", "गया": "gaya", "दरभंगा": "darbhanga", "कानपुर": "kanpur", "मुंबई": "mumbai", "किशनगंज": "kishanganj",
                   "কলকাতা": "kalkata", "மதுரை": "madurai", "చిత్తూరు": "chitturu", "ಮೈಸೂರು": "maisuru", "കൊല്ലം": "kollam",
                   "સુરત": "surat", "ਪਟਿਆਲਾ": "patiala", "ପୁରୀ": "puri", "Patna": "patna"}.items():
    assert cf.romanize(nat_) == eng_, (nat_, cf.romanize(nat_))
for a_, b_ in (("mujaphpharpur", "Muzaffarpur"), ("jaypur", "Jaipur"), ("haidarabad", "Hyderabad"), ("koyambuttur", "Coimbatore"), ("kataka", "Cuttack")):
    assert cf.sound_key(a_) == cf.sound_key(b_), (a_, cf.sound_key(a_), cf.sound_key(b_))
assert cf.sound_key("Patna") != cf.sound_key("Pune")

# places typed in another script, without any model
def place(q): return gaz.resolve(q)
for q in ("अहमदाबाद", "અમદાવાદ", "अहमदाबाद के पास", "near अहमदाबाद", "अहमदाबाद जिला", "Ahmédabad"):
    r_ = place(q); assert r_["status"] == "ok" and r_["candidates"][0]["label"].startswith("Ahmedabad district"), (q, r_)
assert "was read as Ahmedabad" in place("अहमदाबाद")["note"]
r_ = place("पुणे"); assert r_["status"] == "ok" and r_["candidates"][0]["state"].lower() == "maharashtra", r_
r_ = place("३८०००७"); assert r_["status"] == "ok" and r_["candidates"][0]["kind"] == "PIN", r_
r_ = place("२३.०१, ७२.५६"); assert r_["status"] == "ok" and r_["candidates"][0]["kind"] == "coordinates", r_
r_ = place("औरंगाबाद"); assert r_["status"] == "ambiguous" and len(r_["candidates"]) == 2, r_
r_ = place("औरंगाबाद, बिहार"); assert r_["status"] == "ok" and r_["candidates"][0]["state"].lower() == "bihar", r_
r_ = place("औरंगाबाद (महाराष्ट्र)"); assert r_["status"] == "ok" and r_["candidates"][0]["state"].lower() == "maharashtra", r_
r_ = place("बिहार"); assert r_["status"] == "not_found" and "Bihar is a state" in r_["note"], r_
r_ = place("ગુજરાત"); assert r_["status"] == "not_found" and "Gujarat is a state" in r_["note"], r_
# not in the alias file: read letter by letter, and the note says so
r_ = place("भुज"); assert r_["status"] == "ok" and "letter by letter as 'bhuj'" in r_["note"] and "automatic reading" in r_["note"], r_
r_ = place("મહેસાણા"); assert r_["status"] == "ok" and r_["candidates"][0]["label"].startswith("Mehsana"), r_
r_ = place("हैद्राबाद"); assert r_["status"] == "ok" and "sounds like Hyderabad" in r_["note"], r_      # Marathi spelling, not the listed one
assert place("हैदरपुर")["status"] == "not_found"                       # same first consonants, different place: no guess
r_ = place("जमुई"); assert r_["status"] == "not_found" and "is Jamui" in r_["note"], r_      # listed, but not in this data: never a sound-alike
r_ = place("बठिण्डा"); assert r_["status"] == "ok" and r_["candidates"][0]["label"].startswith("Bathinda"), r_
r_ = place("अटलांटिस"); assert r_["status"] == "not_found" and "Devanagari script" in r_["note"] and "PIN" in r_["note"], r_
# scripts that cannot be read by rule: listed names work, others get a clear message
r_ = place("حیدرآباد"); assert r_["status"] == "ok" and "was read as Hyderabad" in r_["note"], r_
r_ = place("پٹنہ"); assert r_["status"] == "not_found" and "Arabic script" in r_["note"], r_
r_ = place("艾哈默達巴德"); assert r_["status"] == "not_found" and "English letters" in r_["note"], r_
rep_ = gaz.alias_report(); assert rep_["other_script_places"] > 100 and "Ahmedabad" not in rep_["other_script_unmatched"] and "Patna" in rep_["other_script_unmatched"]
rn_, pn_ = cf.find_care(state0, "dialysis", "अहमदाबाद"); assert rn_["top"][0]["name"] == "Kidney Care" and "was read as Ahmedabad" in pn_
assert cf.find_care(state1, "dialysis", "अहमदाबाद")[1] == pn_                # same after saving and reloading
assert "Check the place" not in pn_                                          # a listed name is not a guess
for q in ("भुज", "Ahmedabd", "Aurangabad"):                                   # read by rule, misspelled, in two states
    assert "Check the place" in cf.find_care(state0, "dialysis", q)[1], q
assert place("भुज")["guessed"] and place("Ahmedabd")["guessed"] and not place("Ahmedabad")["guessed"] and not place("380007").get("guessed")

# questions in another language, with the scripted model
n_sv = len(cf.SERVICES)
rh, ph = cf.ask(state, "अहमदाबाद के पास डायलिसिस कहाँ होता है?", fake, codes, save_dir=TMP)
assert rh["service"] == "dialysis" and rh["top"][0]["name"] == "Kidney Care", rh
assert "Your question was read as" in ph and "language Hindi" in ph and "डायलिसिस" in ph and "was read as Ahmedabad" in ph, ph[:600]
assert "Your question was read as" not in cf.find_care(state, "dialysis", "Ahmedabad")[1]      # only for free-text questions
rt, pt = cf.ask(state, "அட்லாண்டிஸ் அருகில் எம்ஆர்ஐ", fake, codes, save_dir=TMP)
assert rt is None and "Could not find" in pt and "Atlantis" in pt and "language Tamil" in pt and "MRI" in pt, pt
cf.ask(state, "अहमदाबाद में मोतियाबिंद ऑपरेशन", fake, codes, save_dir=TMP)
cf.ask(state, "अहमदाबाद में घुटने का ऑपरेशन", fake, codes, save_dir=TMP)
new_ = [k for k in cf.SERVICES if k.startswith("custom_") and k != "custom_cataract_surgery"]
assert len(new_) == 2 and all(k.isascii() and len(k) > len("custom_") for k in new_), new_      # two services, two keys
assert cf._service_key("cataract surgery") == "custom_cataract_surgery" and cf._service_key("घुटने का ऑपरेशन") in new_
for k in new_:
    del cf.SERVICES[k]
assert len(cf.SERVICES) == n_sv

# entries that are not in English are kept (they used to be dropped) and counted
fl = cf.prepare_facilities(pd.DataFrame([
    synth("l1", "जीवन अस्पताल", "Ahmedabad", "Gujarat", "380007", 23.02, 72.57, description="यहाँ डायलिसिस की सुविधा उपलब्ध है।",
          procedure=J(["डायलिसिस", "Dialysis unit"]), capability=J(["24 घंटे आपातकालीन सेवा"])),
    synth("l2", "City Hospital", "Ahmedabad", "Gujarat", "380007", 23.02, 72.57, procedure=J(["Café-style waiting area"]))]), pins)
assert fl.iloc[0]["procedure_list"] == ["डायलिसिस", "Dialysis unit"] and fl.iloc[0]["capability_list"] == ["24 घंटे आपातकालीन सेवा"]
sr = cf.script_report(fl).set_index("field")
assert sr.loc["name", "other_script"] == 1 and sr.loc["procedure", "other_script"] == 1 and sr.loc["procedure", "with_text"] == 3, sr
assert sr.loc["capability", "share_pct"] == 100 and "Devanagari" in sr.loc["description", "scripts"], sr
print(sr.to_string())

print(summary.merge(fac[["unique_id", "name"]])[["name", "service", "tier"]].sort_values(["name", "service"]).to_string(index=False))
print("trust:", dict(zip(fac.name.str[:22], fac.trust)))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write("<html><body style='background:#fff'>" + page + cf.find_care(state0, "dialysis", "690509")[1] + "</body></html>")
print("ALL TESTS PASSED")
