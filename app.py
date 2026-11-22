"""Care Finder web app, for Databricks Apps.

A thin screen over carefinder.py. It holds no ranking logic of its own: it loads the state
that the notebook saved next to it (data/*.parquet), takes a service and a place (or one
sentence in any language), asks carefinder for the result, and lays that result out for
someone who is not technical: one headline, then for each facility three questions
(do they offer it, is the information reliable, how far), the record's own words, and a
button to call. English and Hindi.

Run locally:  python app.py   then open http://localhost:8000
"""
import logging
import os
import re
import threading

from flask import Flask, render_template_string, request

import carefinder as cf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CARE_FINDER_DATA", os.path.join(HERE, "data"))
MODEL = os.environ.get("CARE_FINDER_MODEL", "databricks-meta-llama-3-3-70b-instruct")

log = logging.getLogger("care_finder")
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# The state is loaded once, at start-up. A question that needs a new service definition
# changes it, so those run one at a time.
_ask_lock = threading.Lock()
try:
    STATE, LOAD_ERROR = cf.load_state_files(DATA_DIR), ""
    log.info("loaded %d facilities, %d services", len(STATE["fac"]), len(cf.SERVICES))
except Exception as err:      # show the reason on the page instead of failing to start
    STATE, LOAD_ERROR = None, f"The data could not be loaded ({type(err).__name__}: {err})"
    log.exception("could not load the state")


def llm(prompt):
    """One prompt to the Databricks-hosted model, as the app's own identity."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        r = WorkspaceClient().serving_endpoints.query(
            name=MODEL, messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)], temperature=0.0, max_tokens=900)
        content = r.choices[0].message.content
    except Exception as err:
        log.warning("model call failed: %s: %s", type(err).__name__, err)
        raise RuntimeError("the language model is not reachable from this app") from err
    if isinstance(content, list):      # some models return a list of parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ----------------------------------------------------------------------------------
# Words. English and Hindi. The Hindi has not been checked by a native speaker.
# Quotes from records are never translated.
# ----------------------------------------------------------------------------------
T = {
    "en": {
        "title": "Care Finder",
        "lead": "Which hospital near you offers this care, and can you trust it?",
        "quickTitle": "Not sure where to start? Tap an example",
        "quickSub": "It runs the search for you.",
        "trySay": "Try asking:",
        "near": "near {place}",
        "step1": "What care is needed?",
        "step2": "Where is the patient?",
        "placeHolder": "Gaya  ·  गया  ·  800001",
        "placeHint": "Town, district or PIN code. Hindi and other Indian scripts work too.",
        "find": "Find care",
        "or": "or ask in your own words",
        "askHolder": "My patient needs dialysis near Gaya",
        "ask": "Ask",
        "askHint": "Any language. AI only reads the care and the place. It does not choose the hospital.",
        "back": "← New search",
        "understood": "We read your question as:",
        "understoodNear": "near",
        "understoodFix": "Not right? Start a new search and use the boxes.",
        "changePlace": "Change place",
        "qOffer": "Do they offer it?",
        "qTrust": "Is the information reliable?",
        "qFar": "How far?",
        "recordSays": "Their record says:",
        "backedBy": "Backed up by:",
        "detail": "Related detail, not enough to confirm:",
        "hintedBy": "Only hinted by:",
        "more": "and {n} more",
        "entry": "entry",
        "call": "Call",
        "noPhone": "No phone number in the record",
        "sources": "See where this comes from",
        "noSources": "The record has no source link.",
        "others": "Other options worth knowing",
        "whyTitle": "Why this order? (for planners)",
        "whyLead": "Each hospital gets a score: evidence × trust × distance. Higher is better.",
        "whyNote": ("Evidence: {strong} confirmed, {moderate} claimed only, {weak} hinted only. Trust starts at 1.0 and drops for "
                    "each data problem. The distance score halves at {half} km for this kind of care. Travel times are rough "
                    "estimates from straight-line distance, not route calculations."),
        "th": ["#", "Hospital", "Evidence", "Trust", "Distance", "Score"],
        "footer": "This information comes from websites and is not verified. Always call the hospital before sending a patient.",
        "status": {"g": "Confirmed", "y": "They say so, not confirmed", "n": "Only hinted"},
        "trustOk": "No problems found",
        "trustCheck": "{n} thing to check|{n} things to check",
        "needBoth": "Choose the care needed and enter a place.",
        "needQuestion": "Type a question, for example: My patient needs dialysis near Gaya.",
        "aiDefined": "This care was not on our list. An AI wrote the search words below, and nobody has reviewed them. The hospital's own words are still quoted exactly.",
        "aiWords": "Search words:",
        "checkPlace": "Check the place.",
        "km": "{km} km",
        "under1": "under 1 km",
        "byRoad": "{dist} · about {time} by road",
        "range": "{a} to {b}",
        "min": "{n} minutes", "hour": "{n} hour|{n} hours",
        "vBest": "Best confirmed option: {name}, about {km} km away.",
        "vClaimed": "Hospitals near you say they offer {svc}, but none is confirmed. Call before you go.",
        "vClaimedFar": "The nearest hospitals only say they offer {svc}. The closest confirmed one is {name}, about {km} km away.",
        "vHinted": "No hospital within {km} km says it offers {svc}. The options below only hint at it. Call before you go.",
        "vNone": "No hospital with any sign of {svc} was found within {km} km.",
        "sCount": "{n} hospital within {km} km says it offers this.|{n} hospitals within {km} km say they offer this.",
        "sWidened": "Nothing closer than {km} km offers it, so we looked farther.",
        "sTooFar": "At this distance the trip may take too long for an emergency.",
        "sSparse": "Only {n} hospital of any kind is listed near this place, so this answer may be incomplete.|Only {n} hospitals of any kind are listed near this place, so this answer may be incomplete.",
        "tWithin": "Inside the {n}-minute limit for this care, even on a slow trip.",
        "tUncertain": "May not make the {n}-minute limit for this care.",
        "tBeyond": "Too far for the {n}-minute limit for this care, even on a fast trip.",
        "warn": {"no_first_party_source": "Not from the hospital's own website"},
        "approx": "The location is approximate (centre of the PIN code area), so the distance is too.",
        "alt": {"Closest corroborated option": "Closest confirmed option",
                "Nearby, but the service is only implied": "Nearby, but only hinted",
                "Nearest option beyond the search radius": "Nearest, but beyond the search distance"},
    },
    "hi": {
        "title": "केयर फ़ाइंडर",
        "lead": "आपके पास कौन-सा अस्पताल यह इलाज देता है, और क्या उस पर भरोसा किया जा सकता है?",
        "quickTitle": "समझ नहीं आ रहा कहाँ से शुरू करें? कोई उदाहरण दबाएँ",
        "quickSub": "यह आपके लिए खोज चला देगा।",
        "trySay": "ऐसे पूछकर देखें:",
        "near": "{place} के पास",
        "step1": "कौन-सा इलाज चाहिए?",
        "step2": "मरीज़ कहाँ है?",
        "placeHolder": "गया  ·  Gaya  ·  800001",
        "placeHint": "शहर, ज़िला या पिन कोड। हिन्दी या किसी भी भारतीय लिपि में लिख सकते हैं।",
        "find": "अस्पताल खोजें",
        "or": "या अपने शब्दों में पूछें",
        "askHolder": "गया के पास डायलिसिस कहाँ होता है?",
        "ask": "पूछें",
        "askHint": "किसी भी भाषा में। AI सिर्फ़ इलाज और जगह पढ़ता है, अस्पताल नहीं चुनता।",
        "back": "← नई खोज",
        "understood": "हमने आपका सवाल ऐसे समझा:",
        "understoodNear": "· जगह:",
        "understoodFix": "सही नहीं है? नई खोज शुरू करें और बॉक्स इस्तेमाल करें।",
        "changePlace": "जगह बदलें",
        "qOffer": "क्या यह इलाज मिलता है?",
        "qTrust": "क्या जानकारी भरोसेमंद है?",
        "qFar": "कितनी दूर?",
        "recordSays": "रिकॉर्ड में लिखा है:",
        "backedBy": "इसका सबूत:",
        "detail": "संबंधित जानकारी, पुष्टि के लिए काफ़ी नहीं:",
        "hintedBy": "सिर्फ़ इससे संकेत मिलता है:",
        "more": "और {n}",
        "entry": "प्रविष्टि",
        "call": "फ़ोन करें",
        "noPhone": "रिकॉर्ड में फ़ोन नंबर नहीं है",
        "sources": "यह जानकारी कहाँ से आई",
        "noSources": "रिकॉर्ड में कोई स्रोत लिंक नहीं है।",
        "others": "जानने लायक़ अन्य विकल्प",
        "whyTitle": "यह क्रम क्यों? (योजनाकारों के लिए)",
        "whyLead": "हर अस्पताल का स्कोर: सबूत × भरोसा × दूरी। ज़्यादा स्कोर बेहतर है।",
        "whyNote": ("सबूत: पुष्टि {strong}, सिर्फ़ दावा {moderate}, सिर्फ़ संकेत {weak}। भरोसा 1.0 से शुरू होता है और हर डेटा समस्या पर "
                    "घटता है। इस तरह के इलाज के लिए दूरी का स्कोर {half} किमी पर आधा हो जाता है। सफ़र का समय सीधी दूरी से लगाया गया "
                    "अनुमान है, असली रास्ते की गणना नहीं।"),
        "th": ["#", "अस्पताल", "सबूत", "भरोसा", "दूरी", "स्कोर"],
        "footer": "यह जानकारी वेबसाइटों से ली गई है और सत्यापित नहीं है। मरीज़ भेजने से पहले हमेशा अस्पताल को फ़ोन करें।",
        "status": {"g": "पुष्टि हुई", "y": "दावा है, पुष्टि नहीं", "n": "सिर्फ़ संकेत"},
        "trustOk": "कोई समस्या नहीं मिली",
        "trustCheck": "{n} बात जाँचें|{n} बातें जाँचें",
        "needBoth": "इलाज चुनें और जगह लिखें।",
        "needQuestion": "सवाल लिखें, जैसे: गया के पास डायलिसिस कहाँ होता है?",
        "aiDefined": "यह इलाज हमारी सूची में नहीं था। नीचे के खोज-शब्द AI ने लिखे हैं और किसी ने इनकी जाँच नहीं की है। अस्पताल के अपने शब्द फिर भी ज्यों के त्यों दिखाए गए हैं।",
        "aiWords": "खोज-शब्द:",
        "checkPlace": "जगह जाँच लें।",
        "km": "{km} किमी",
        "under1": "1 किमी से कम",
        "byRoad": "{dist} · सड़क से लगभग {time}",
        "range": "{a} से {b}",
        "min": "{n} मिनट", "hour": "{n} घंटा|{n} घंटे",
        "vBest": "सबसे भरोसेमंद विकल्प: {name}, लगभग {km} किमी दूर।",
        "vClaimed": "पास के अस्पताल {svc} देने का दावा करते हैं, पर किसी की पुष्टि नहीं हुई। जाने से पहले फ़ोन करें।",
        "vClaimedFar": "सबसे पास के अस्पताल {svc} का सिर्फ़ दावा करते हैं। सबसे पास का पुष्टि वाला अस्पताल {name} है, लगभग {km} किमी दूर।",
        "vHinted": "{km} किमी के अंदर कोई अस्पताल {svc} देने का दावा नहीं करता। नीचे के विकल्प सिर्फ़ संकेत देते हैं। जाने से पहले फ़ोन करें।",
        "vNone": "{km} किमी के अंदर {svc} का कोई संकेत देने वाला अस्पताल नहीं मिला।",
        "sCount": "{km} किमी के अंदर {n} अस्पताल यह इलाज देने का दावा करता है।|{km} किमी के अंदर {n} अस्पताल यह इलाज देने का दावा करते हैं।",
        "sWidened": "{km} किमी के अंदर कुछ नहीं मिला, इसलिए खोज आगे बढ़ाई गई।",
        "sTooFar": "इतनी दूरी पर आपातकाल में पहुँचने में बहुत देर हो सकती है।",
        "sSparse": "इस जगह के पास किसी भी तरह का सिर्फ़ {n} अस्पताल दर्ज है, इसलिए यह जवाब अधूरा हो सकता है।|इस जगह के पास किसी भी तरह के सिर्फ़ {n} अस्पताल दर्ज हैं, इसलिए यह जवाब अधूरा हो सकता है।",
        "tWithin": "धीमे सफ़र में भी इस इलाज की {n} मिनट की समय-सीमा के अंदर।",
        "tUncertain": "इस इलाज की {n} मिनट की समय-सीमा में पहुँचना मुश्किल हो सकता है।",
        "tBeyond": "तेज़ सफ़र में भी इस इलाज की {n} मिनट की समय-सीमा से ज़्यादा समय लगेगा।",
        "warn": {"no_first_party_source": "जानकारी अस्पताल की अपनी वेबसाइट से नहीं है",
                 "location_conflict": "रिकॉर्ड का पता उसके पिन कोड से मेल नहीं खाता",
                 "coords_outside_india": "दर्ज निर्देशांक भारत के बाहर हैं",
                 "possible_merged_record": "इस रिकॉर्ड में बहुत ज़्यादा फ़ोन नंबर हैं; शायद इसमें कई अस्पताल मिल गए हैं",
                 "malformed_record": "इस रिकॉर्ड के खाने गड़बड़ लगते हैं, इसलिए जानकारी ग़लत खाने में हो सकती है"},
        "approx": "जगह अंदाज़न है (पिन कोड क्षेत्र का केंद्र), इसलिए दूरी भी अंदाज़न है।",
        "alt": {"Closest corroborated option": "सबसे पास का पुष्टि वाला विकल्प",
                "Nearby, but the service is only implied": "पास में, पर सिर्फ़ संकेत",
                "Nearest option beyond the search radius": "सबसे पास, पर खोज की दूरी से बाहर"},
    },
}

# The eight predefined services: icon and plain names. A service the model defined gets a default icon.
TILES = [
    ("dialysis", "🩸", "Dialysis", "डायलिसिस"),
    ("emergency_c_section", "🤰", "Emergency C-section", "आपातकालीन सिज़ेरियन"),
    ("nicu", "👶", "Newborn ICU", "नवजात आईसीयू"),
    ("trauma", "🚑", "Accident / trauma", "दुर्घटना / ट्रॉमा"),
    ("cardiac_surgery", "❤️", "Heart surgery", "हृदय सर्जरी"),
    ("mri", "🧲", "MRI scan", "एमआरआई"),
    ("ct_scan", "🩻", "CT scan", "सीटी स्कैन"),
    ("burn_care", "🔥", "Burn care", "जलने का इलाज"),
]
QUICK = [("emergency_c_section", "Nashik", "नासिक"), ("dialysis", "Gaya", "गया"), ("cardiac_surgery", "194101", "194101")]
SAY = ["My patient needs dialysis near Gaya", "गया के पास डायलिसिस कहाँ होता है?", "Pregnant woman needs emergency C-section near Nashik"]
ICON = {"g": "✓", "y": "!", "r": "✕", "n": "?"}
TIER_STATUS = {"strong": "g", "moderate": "y", "weak": "n"}
TIME_STATUS = {"within": "g", "uncertain": "y", "beyond": "r"}
FIELD = {"procedure": "Procedures", "capability": "Capabilities", "equipment": "Equipment", "description": "Description", "specialties": "Specialties"}


def plural(text, n, **more):
    """'one form|other form' -> the right one for n."""
    one, _, many = text.partition("|")
    return (one if n == 1 or not many else many).format(n=n, **more)


def tiles(lang):
    out = [{"key": k, "icon": ic, "label": hi if lang == "hi" else en} for k, ic, en, hi in TILES if k in cf.SERVICES]
    known = {t["key"] for t in out}
    out += [{"key": k, "icon": "🏥", "label": v["label"]} for k, v in cf.SERVICES.items() if k not in known]
    return out


def service_name(key, lang):
    return next((t["label"] for t in tiles(lang) if t["key"] == key), key)


def trip_time(lo, hi, t):
    """A travel-time range in plain words: '5 to 15 minutes', '1 to 2 hours', '45 minutes to 1.5 hours'."""
    def part(minutes):
        minutes = max(5, int(round(minutes / 5.0)) * 5)
        if minutes < 90:
            return minutes, "min"
        hours = round(minutes / 30.0) / 2        # to the nearest half hour
        return (int(hours) if float(hours).is_integer() else hours), "hour"

    (a, ua), (b, ub) = part(lo), part(hi)
    if (a, ua) == (b, ub):
        return plural(t[ua], a)
    if ua == ub:                                  # same unit: say it once
        return t["range"].format(a=a, b=plural(t[ub], b))
    return t["range"].format(a=plural(t[ua], a), b=plural(t[ub], b))


def card_view(rec, lang, service):
    """One facility as the page shows it: three questions, the record's own words, a phone number."""
    t = T[lang]
    offer = TIER_STATUS[rec["tier"]]
    ev = rec["evidence"]

    def quotes(test, limit=2):
        hits = [e for e in ev if test(e["role"])]
        return {"items": [{"text": e["text"], "where": f"{FIELD.get(e['field'], e['field'])}, {t['entry']} {int(e['idx']) + 1}"} for e in hits[:limit]],
                "more": t["more"].format(n=len(hits) - limit) if len(hits) > limit else ""}

    says = quotes(lambda r: r == "states the service")
    backed = quotes(lambda r: r == "equipment" or r.startswith("supporting"))
    hints = quotes(lambda r: r in ("related", "specialty tag")) if offer == "n" else {"items": [], "more": ""}

    warn = [t["warn"].get(code, text) for code, text in rec["trust_reasons"]] + [str(n) for n in rec["notes"]]
    if rec.get("loc_approximate"):
        warn.append(t["approx"])

    km = rec["distance_km"]
    dist = t["under1"] if km < 1 else t["km"].format(km=f"{km:.0f}")
    lo, hi = rec["travel_min"]
    far_status = TIME_STATUS.get(rec.get("time_status")) or ("g" if km <= cf.CONFIG["half_km"][cf.SERVICES[service]["urgency"]] else "y")
    target = cf.SERVICES[service].get("target_minutes")
    far_note = {"within": t["tWithin"], "uncertain": t["tUncertain"], "beyond": t["tBeyond"]}.get(rec.get("time_status"), "").format(n=target)

    phone = "" if cf.is_missing(rec.get("officialPhone")) else str(rec["officialPhone"]).strip()
    own = cf.domain_of(rec.get("officialWebsite"))
    urls = [u for u in sorted(rec["source_url_list"], key=lambda u: cf.domain_of(u) != own) if re.match(r"https?://", str(u))][:3]
    return {
        "name": str(rec["name"]),
        "town": ", ".join(cf.dedupe([str(x) for x in (rec.get("address_city"), rec.get("district"), rec.get("state")) if not cf.is_missing(x)])),
        "offer": offer, "offer_text": t["status"][offer], "says": says, "backed": backed, "hints": hints,
        "trust": "y" if warn else "g", "trust_text": plural(t["trustCheck"], len(warn)) if warn else t["trustOk"], "warn": warn,
        "far": far_status, "far_text": t["byRoad"].format(dist=dist, time=trip_time(lo, hi, t)), "far_note": far_note,
        "phone": phone, "tel": re.sub(r"[^0-9+]", "", phone),
        "sources": [{"url": u, "text": re.sub(r"^https?://(www\.)?", "", u)[:80]} for u in urls],
        "score": [f"{rec['evidence_value']:.2f}", f"{rec['trust']:.2f}", f"{rec['proximity']:.2f}", f"{rec['score']:.2f}"],
    }


def result_view(res, lang):
    """The whole answer as the page shows it. Every number and name comes from carefinder's result."""
    t, key = T[lang], res["service"]
    svc = service_name(key, lang)
    top, alts = res["top"], res["alternatives"]
    radius = f"{res['max_km']:.0f}"
    confirmed = alts.get("Closest corroborated option")
    if not top:
        status, big = "n", t["vNone"].format(svc=svc, km=radius)
    elif top[0]["tier"] == "strong":
        status, big = "g", t["vBest"].format(name=top[0]["name"], km=f"{max(1, round(top[0]['distance_km']))}")
    elif top[0]["tier"] == "moderate" and confirmed is not None:
        status, big = "y", t["vClaimedFar"].format(svc=svc, name=confirmed["name"], km=f"{max(1, round(confirmed['distance_km']))}")
    elif top[0]["tier"] == "moderate":
        status, big = "y", t["vClaimed"].format(svc=svc)
    else:
        status, big = "n", t["vHinted"].format(svc=svc, km=radius)
    sub = []
    if res["widened"]:
        sub.append(t["sWidened"].format(km=f"{res['base_km']:.0f}"))
        if res["urgency"] == "emergency":
            sub.append(t["sTooFar"])
    n = res["counts"]["stating_within_radius"]
    if n:
        sub.append(plural(t["sCount"], n, km=radius))
    if res["sparse_note"]:
        sub.append(plural(t["sSparse"], res["facilities_nearby"]))
    loc = res["location"]
    cards = [dict(card_view(r, lang, key), rank=i + 1, heading="") for i, r in enumerate(top)]
    others = [dict(card_view(r, lang, key), rank="", heading=t["alt"].get(h, h)) for h, r in alts.items()]
    definition = cf.SERVICES[key].get("generated") and {"words": ", ".join(cf.SERVICES[key]["direct"])}
    tv = cf.CONFIG["tier_value"]
    return {"status": status, "big": big, "sub": " ".join(sub), "place": loc["candidates"][0]["label"],
            "place_note": loc["note"] if (loc.get("guessed") or loc["status"] == "ambiguous") else "",
            "cards": cards, "others": others, "definition": definition, "service": svc,
            "why_note": t["whyNote"].format(strong=tv["strong"], moderate=tv["moderate"], weak=tv["weak"], half=f"{res['half_km']:.0f}")}


PAGE = """<!doctype html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ t.title }}</title>
<style>
  :root {
    --ink:#111a22; --soft:#3d4a56; --line:#c3ccd5; --bg:#eef1f4; --card:#fff; --blue:#1a56b0;
    --g:#1e7a3c; --g-bg:#e3f4e8; --y:#8a5a00; --y-bg:#fdf1d6; --r:#b3261e; --r-bg:#fbe6e4; --n:#4a5560; --n-bg:#e9ecef;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI","Noto Sans Devanagari",sans-serif; font-size:20px; line-height:1.55; }
  main { max-width:920px; margin:0 auto; padding:16px 16px 64px; }
  button { font:inherit; cursor:pointer; }
  a { color:var(--blue); }
  header { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin:12px 0 18px; flex-wrap:wrap; }
  h1 { font-size:36px; margin:0; letter-spacing:-.01em; line-height:1.2; }
  .lead { color:var(--soft); margin:6px 0 0; font-size:21px; }
  .lang { display:flex; border:2px solid var(--blue); border-radius:12px; overflow:hidden; flex-shrink:0; }
  .lang a { background:#fff; color:var(--blue); padding:10px 18px; font-size:20px; font-weight:700; min-height:52px; text-decoration:none; display:inline-flex; align-items:center; }
  .lang a.on { background:var(--blue); color:#fff; }

  .panel { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; }
  .step { font-size:24px; font-weight:700; margin:0 0 12px; display:flex; align-items:center; gap:10px; }
  .num { display:inline-grid; place-items:center; width:38px; height:38px; border-radius:50%; background:var(--blue); color:#fff; font-size:20px; flex-shrink:0; }
  .services { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; }
  .services input { position:absolute; opacity:0; width:1px; height:1px; }
  .svc { background:#fff; border:2px solid var(--line); border-radius:14px; padding:14px 8px; min-height:112px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; color:var(--ink); font-size:19px; font-weight:600; text-align:center; line-height:1.25; cursor:pointer; }
  .svc .ic { font-size:38px; line-height:1; }
  .svc:hover { border-color:var(--blue); }
  .services input:checked + .svc { border-color:var(--blue); background:#e7effb; box-shadow:inset 0 0 0 2px var(--blue); }
  .services input:focus-visible + .svc { outline:3px solid var(--blue); outline-offset:2px; }
  .where { display:grid; grid-template-columns:1fr auto; gap:12px; }
  input[type=text] { width:100%; min-height:64px; padding:12px 16px; font:inherit; font-size:22px; color:inherit; background:#fff; border:2px solid #8f9ba7; border-radius:12px; }
  input[type=text]:focus { outline:3px solid var(--blue); outline-offset:1px; border-color:var(--blue); }
  .primary { min-height:64px; padding:0 30px; font-size:22px; font-weight:700; color:#fff; background:var(--blue); border:0; border-radius:12px; white-space:nowrap; }
  .secondary { min-height:64px; padding:0 26px; font-size:22px; font-weight:700; color:var(--blue); background:#fff; border:2px solid var(--blue); border-radius:12px; white-space:nowrap; }
  button:disabled { opacity:.6; cursor:progress; }
  .hint { color:var(--soft); font-size:18px; margin:8px 0 0; }
  .or { display:flex; align-items:center; gap:12px; color:var(--soft); font-size:18px; margin:26px 0 14px; }
  .or::before, .or::after { content:""; flex:1; border-top:2px solid var(--line); }

  .quick-wrap { background:#fff; border:2px dashed #9fb6d8; border-radius:16px; padding:18px 20px; margin-bottom:16px; }
  .quick-title { font-size:23px; font-weight:700; margin:0; }
  .quick-sub { color:var(--soft); font-size:18px; margin:2px 0 12px; }
  .quick { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
  .qk { display:flex; align-items:center; gap:14px; text-align:left; background:#f3f7fd; color:var(--ink); border:2px solid #b9cbe6; border-radius:14px; padding:14px 16px; min-height:84px; text-decoration:none; }
  .qk:hover, .qk:focus-visible { border-color:var(--blue); background:#e7effb; }
  .qk .ic { font-size:38px; line-height:1; flex-shrink:0; }
  .qk b { display:block; font-size:21px; line-height:1.25; }
  .qk .pl { display:block; font-size:18px; color:var(--soft); }
  .say { display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; align-items:center; }
  .say-label { font-size:18px; color:var(--soft); font-weight:600; }
  .sayc { background:#fff; color:var(--ink); border:2px solid #b9cbe6; border-radius:999px; padding:8px 18px; font-size:19px; min-height:52px; text-decoration:none; display:inline-flex; align-items:center; }
  .sayc:hover { border-color:var(--blue); background:#e7effb; }
  .notice { background:var(--r-bg); border:2px solid var(--r); border-radius:14px; padding:14px 18px; margin-bottom:16px; }

  .back { display:inline-block; color:var(--blue); font-weight:700; font-size:20px; padding:8px 0; margin-bottom:8px; text-decoration:none; }
  .understood { background:#e7effb; border-left:6px solid var(--blue); border-radius:10px; padding:12px 16px; margin-bottom:14px; font-size:19px; }
  .aidef { background:#fff; border:2px dashed #8f9ba7; border-radius:14px; padding:12px 16px; margin-bottom:14px; font-size:18px; }
  .verdict { border-radius:16px; padding:20px 22px; display:flex; gap:16px; align-items:flex-start; border:2px solid var(--c); background:var(--cb); }
  .verdict .big { font-size:27px; font-weight:700; line-height:1.35; }
  .verdict .sub { margin-top:6px; font-size:20px; }
  .verdict .place { margin-top:10px; font-size:19px; color:var(--soft); }
  .verdict .place a { font-weight:700; margin-left:8px; }
  .verdict .check { margin-top:8px; font-size:18px; background:#fff; border-radius:10px; padding:8px 12px; }
  .light { flex-shrink:0; display:inline-grid; place-items:center; border-radius:50%; color:#fff; font-weight:800; background:var(--c); }
  .verdict .light { width:52px; height:52px; font-size:30px; }
  .s-g { --c:var(--g); --cb:var(--g-bg); } .s-y { --c:var(--y); --cb:var(--y-bg); } .s-r { --c:var(--r); --cb:var(--r-bg); } .s-n { --c:var(--n); --cb:var(--n-bg); }

  .others { font-size:24px; font-weight:700; margin:28px 0 0; }
  .card { background:var(--card); border:2px solid var(--line); border-radius:16px; padding:22px; margin-top:16px; }
  .card .kind { font-size:17px; font-weight:700; color:var(--soft); text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px; }
  .card h2 { margin:0 0 4px; font-size:28px; display:flex; gap:12px; align-items:center; line-height:1.25; }
  .rank { display:inline-grid; place-items:center; min-width:44px; height:44px; border-radius:12px; background:var(--ink); color:#fff; font-size:24px; flex-shrink:0; }
  .card .town { color:var(--soft); font-size:19px; }
  .card .town.in { margin-left:56px; }
  .q { border-top:1px solid var(--line); margin-top:16px; padding-top:14px; }
  .q-head { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  .q-title { font-weight:700; font-size:20px; color:var(--soft); }
  .pill { display:inline-flex; align-items:center; gap:10px; background:var(--cb); color:var(--c); border:2px solid var(--c); border-radius:999px; padding:4px 16px 4px 6px; font-weight:800; font-size:20px; }
  .pill .light { width:32px; height:32px; font-size:19px; }
  .q-body { margin-top:8px; font-size:20px; }
  .quote { background:#f5f7f9; border-left:5px solid var(--c, var(--line)); border-radius:8px; padding:10px 14px; margin-top:6px; font-size:21px; }
  .quote small { display:block; color:var(--soft); font-size:16px; margin-bottom:2px; }
  .quote .where { color:var(--soft); font-size:16px; }
  .q-body ul { margin:4px 0 0; padding-left:26px; }
  .actions { display:flex; gap:12px; align-items:center; margin-top:20px; flex-wrap:wrap; }
  .call { display:inline-flex; align-items:center; gap:10px; min-height:64px; padding:0 26px; font-size:23px; font-weight:800; color:#fff; background:var(--g); border-radius:14px; text-decoration:none; }
  .nophone { color:var(--soft); font-size:19px; }
  details { margin-top:4px; }
  summary { cursor:pointer; color:var(--blue); font-weight:700; font-size:20px; padding:10px 4px; }
  details ul { margin:4px 0 0; padding-left:26px; font-size:17px; word-break:break-all; }
  .why { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:6px 22px 18px; margin-top:22px; }
  .why table { border-collapse:collapse; width:100%; font-size:17px; font-variant-numeric:tabular-nums; margin-top:8px; }
  .why th, .why td { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }
  .why p { font-size:17px; color:var(--soft); }
  footer { margin-top:24px; font-size:18px; color:var(--soft); }

  @media (max-width:700px) {
    body { font-size:19px; }
    .services { grid-template-columns:repeat(2, 1fr); }
    .quick { grid-template-columns:1fr; }
    .where { grid-template-columns:1fr; }
    .primary, .secondary, .call { width:100%; justify-content:center; }
    .card .town.in { margin-left:0; }
    h1 { font-size:30px; }
  }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{{ t.title }}</h1>
      <p class="lead">{{ t.lead }}</p>
    </div>
    <div class="lang" role="group" aria-label="Language">
      <a href="{{ url_en }}" {% if lang == "en" %}class="on" aria-current="true"{% endif %}>English</a>
      <a href="{{ url_hi }}" {% if lang == "hi" %}class="on" aria-current="true"{% endif %}>हिन्दी</a>
    </div>
  </header>

{% macro quotes(label, group) %}
  {% for e in group["items"] %}
  <div class="quote"><small>{{ label }}</small>&ldquo;{{ e.text }}&rdquo; <span class="where">[{{ e.where }}]</span></div>
  {% endfor %}
  {% if group.more %}<div class="quote"><span class="where">{{ group.more }}</span></div>{% endif %}
{% endmacro %}

{% macro card(c) %}
  <article class="card">
    {% if c.heading %}<div class="kind">{{ c.heading }}</div>{% endif %}
    <h2>{% if c.rank %}<span class="rank">{{ c.rank }}</span>{% endif %}{{ c.name }}</h2>
    <div class="town{% if c.rank %} in{% endif %}">{{ c.town }}</div>

    <div class="q">
      <div class="q-head"><span class="q-title">🩺 {{ t.qOffer }}</span><span class="pill s-{{ c.offer }}"><span class="light">{{ icon[c.offer] }}</span>{{ c.offer_text }}</span></div>
      <div class="q-body s-{{ c.offer }}">
        {{ quotes(t.recordSays, c.says) }}
        {{ quotes(t.backedBy if c.offer == "g" else t.detail, c.backed) }}
        {{ quotes(t.hintedBy, c.hints) }}
      </div>
    </div>

    <div class="q">
      <div class="q-head"><span class="q-title">🔍 {{ t.qTrust }}</span><span class="pill s-{{ c.trust }}"><span class="light">{{ icon[c.trust] }}</span>{{ c.trust_text }}</span></div>
      {% if c.warn %}<div class="q-body"><ul>{% for w in c.warn %}<li>{{ w }}</li>{% endfor %}</ul></div>{% endif %}
    </div>

    <div class="q">
      <div class="q-head"><span class="q-title">📍 {{ t.qFar }}</span><span class="pill s-{{ c.far }}"><span class="light">{{ icon[c.far] }}</span>{{ c.far_text }}</span></div>
      {% if c.far_note %}<div class="q-body">{{ c.far_note }}</div>{% endif %}
    </div>

    <div class="actions">
      {% if c.tel %}<a class="call" href="tel:{{ c.tel }}">📞 {{ t.call }} {{ c.phone }}</a>{% else %}<span class="nophone">{{ t.noPhone }}</span>{% endif %}
    </div>
    <details><summary>{{ t.sources }}</summary>
      {% if c.sources %}<ul>{% for s in c.sources %}<li><a href="{{ s.url }}" target="_blank" rel="noopener noreferrer">{{ s.text }}</a></li>{% endfor %}</ul>
      {% else %}<p class="hint">{{ t.noSources }}</p>{% endif %}
    </details>
  </article>
{% endmacro %}

{% if v %}
  <section id="result">
    <a class="back" href="{{ url_search }}">{{ t.back }}</a>
    {% if understood %}
    <div class="understood">{{ t.understood }} <b>{{ v.service }}</b> {{ t.understoodNear }} <b>{{ understood.place }}</b>. {{ t.understoodFix }}
      <div class="hint" dir="auto">&ldquo;{{ understood.question }}&rdquo;</div></div>
    {% endif %}
    {% if v.definition %}<div class="aidef">{{ t.aiDefined }}<br><b>{{ t.aiWords }}</b> {{ v.definition.words }}</div>{% endif %}
    <div class="verdict s-{{ v.status }}">
      <span class="light">{{ icon[v.status] }}</span>
      <div>
        <div class="big">{{ v.big }}</div>
        {% if v.sub %}<div class="sub">{{ v.sub }}</div>{% endif %}
        <div class="place">📍 {{ v.place }}<a href="{{ url_search }}">{{ t.changePlace }}</a></div>
        {% if v.place_note %}<div class="check"><b>{{ t.checkPlace }}</b> {{ v.place_note }}</div>{% endif %}
      </div>
    </div>
    {% for c in v.cards %}{{ card(c) }}{% endfor %}
    {% if v.others %}<p class="others">{{ t.others }}</p>{% for c in v.others %}{{ card(c) }}{% endfor %}{% endif %}
    {% if v.cards or v.others %}
    <details class="why">
      <summary>{{ t.whyTitle }}</summary>
      <p>{{ t.whyLead }}</p>
      <table>
        <tr>{% for h in t.th %}<th>{{ h }}</th>{% endfor %}</tr>
        {% for c in v.cards + v.others %}<tr><td>{{ c.rank or "·" }}</td><td>{{ c.name }}</td><td>{{ c.score[0] }}</td><td>{{ c.score[1] }}</td><td>{{ c.score[2] }}</td><td><b>{{ c.score[3] }}</b></td></tr>{% endfor %}
      </table>
      <p>{{ v.why_note }}</p>
    </details>
    {% endif %}
  </section>
{% else %}
  <section id="search">
    {% if notice %}<div class="notice">{{ notice }}</div>{% endif %}
    <div class="quick-wrap">
      <p class="quick-title">{{ t.quickTitle }}</p>
      <p class="quick-sub">{{ t.quickSub }}</p>
      <div class="quick">
        {% for q in quick %}<a class="qk" href="{{ q.url }}"><span class="ic">{{ q.icon }}</span><span><b>{{ q.label }}</b><span class="pl">{{ q.near }}</span></span></a>{% endfor %}
      </div>
    </div>
    <form class="panel" method="get" action="" onsubmit="const f = this; setTimeout(function () { for (const b of f.querySelectorAll('button')) { b.disabled = true; } }, 0);">
      <input type="hidden" name="lang" value="{{ lang }}">
      <div class="step"><span class="num">1</span><span>{{ t.step1 }}</span></div>
      <div class="services">
        {% for s in tiles %}<input type="radio" name="service" id="svc-{{ s.key }}" value="{{ s.key }}"{% if s.key == service %} checked{% endif %}><label class="svc" for="svc-{{ s.key }}"><span class="ic">{{ s.icon }}</span>{{ s.label }}</label>{% endfor %}
      </div>

      <div class="step" style="margin-top:28px"><span class="num">2</span><span>{{ t.step2 }}</span></div>
      <div class="where">
        <input type="text" id="location" name="location" value="{{ location }}" placeholder="{{ t.placeHolder }}" autocomplete="off" aria-label="{{ t.step2 }}">
        <button class="primary" type="submit" name="go" value="fields">{{ t.find }}</button>
      </div>
      <p class="hint">{{ t.placeHint }}</p>

      <div class="or">{{ t.or }}</div>
      <div class="where">
        <input type="text" id="q" name="q" value="{{ q }}" placeholder="{{ t.askHolder }}" autocomplete="off" aria-label="{{ t.or }}"
               onkeydown="if (event.key === 'Enter') { event.preventDefault(); document.getElementById('ask').click(); }">
        <button class="secondary" type="submit" id="ask" name="go" value="ask">{{ t.ask }}</button>
      </div>
      <p class="hint">{{ t.askHint }}</p>
      <div class="say"><span class="say-label">{{ t.trySay }}</span>
        {% for s in say %}<a class="sayc" href="{{ s.url }}" dir="auto">&ldquo;{{ s.text }}&rdquo;</a>{% endfor %}
      </div>
    </form>
  </section>
{% endif %}

  <footer>{{ t.footer }}</footer>
</main>
</body>
</html>"""


def link(**params):
    from urllib.parse import urlencode
    return "?" + urlencode({k: v for k, v in params.items() if v})


@app.route("/")
def home():
    lang = "hi" if request.args.get("lang") == "hi" else "en"
    t = T[lang]
    q = (request.args.get("q") or "").strip()[:500]
    service = (request.args.get("service") or "").strip()
    location = (request.args.get("location") or "").strip()[:200]
    go = request.args.get("go")
    if go == "fields" and q and not (service and location):      # Enter pressed in the question box
        go = "ask"
    view, notice, understood = None, "", None
    if STATE is None:
        notice = LOAD_ERROR
    elif go == "ask":
        if not q:
            notice = t["needQuestion"]
        else:
            with _ask_lock:
                out = cf.ask_lookup(STATE, q, llm)
            understood = out["understood"]
            if out["result"] is None:
                notice = out["message"]
            else:      # carry the reading into the boxes, where it can be corrected
                service = out["result"]["service"]
                location = understood["place"]
                view = result_view(out["result"], lang)
    elif go == "fields":
        if not (service and location):
            notice = t["needBoth"]
        else:
            res, message = cf.lookup(STATE, service, location)
            notice, q = message, ""
            view = result_view(res, lang) if res is not None else None
    # Switching language shows the same answer again without asking the model a second time.
    here = {"service": service, "location": location, "go": "fields"} if view else {"service": service, "location": location, "q": q}
    return render_template_string(
        PAGE, lang=lang, t=t, icon=ICON, v=view, notice=notice, understood=understood if view else None,
        tiles=tiles(lang), service=service, location=location, q=q,
        url_en=link(**here), url_hi=link(lang="hi", **here),
        url_search=link(lang=lang if lang == "hi" else "", service=service, location=location),
        quick=[{"url": link(lang=lang if lang == "hi" else "", service=k, location=en, go="fields"),
                "icon": next(ic for kk, ic, _, _ in TILES if kk == k), "label": service_name(k, lang),
                "near": t["near"].format(place=("PIN " + en) if en.isdigit() else (hi if lang == "hi" else en))} for k, en, hi in QUICK if k in cf.SERVICES],
        say=[{"url": link(lang=lang if lang == "hi" else "", q=s, go="ask"), "text": s} for s in SAY])


@app.route("/health")
def health():
    return ({"status": "ok", "facilities": len(STATE["fac"])}, 200) if STATE else ({"status": "no data", "detail": LOAD_ERROR}, 503)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")), threaded=True)
