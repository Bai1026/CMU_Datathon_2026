# Care Finder

Given a service and a location in India, return the best facility options from the Virtue
Foundation dataset, with the exact record entries each recommendation rests on.

`score = evidence x trust x proximity`

## Files

| File | What it is |
|---|---|
| `care_finder_notebook.py` | Databricks notebook. No logic: loads data, calls the library, saves tables, shows results. |
| `carefinder.py` | The library. Pure pandas, no Spark. |
| `scoring.yaml` | Every number: distance limits, penalties, grade values, travel-time assumptions. |
| `services.yaml` | What counts as evidence for each service, as plain term lists. |
| `location_aliases.yaml` | Other names for places (old names, spellings, abbreviations, names in Indian scripts), and filler words. |
| `answer.html.j2` | The answer's HTML template. |
| `services_generated.yaml` | Created at run time: definitions a language model wrote for services not in `services.yaml`. |
| `test_carefinder.py` | Tests. Run after any change. |
| `sample_rows.tsv` | Five real rows used by the tests. Git-ignored: do not publish dataset rows. |

All of these must sit in the same folder.

## Run on Databricks

1. Put the whole folder in the workspace. Easiest with the Git folder: commit these files to the
   repository, then Pull in Databricks.
2. Open `care_finder_notebook` and Run all.
3. Change the widgets at the top and re-run section 5 (fields) or section 7 (plain-language question).

After editing a YAML file or `carefinder.py`, re-run from section 0.

## Change the method

- A service's terms: edit `services.yaml`. The header of that file explains the matching rules.
- A new service: add an entry to `services.yaml`, or ask for it in section 7 and review what the model
  wrote in `services_generated.yaml`, then move it into `services.yaml`.
- Weights and distances: edit `scoring.yaml`.
- A place name that is not recognised: add a group to `location_aliases.yaml`. Section 3c of the notebook
  lists groups that matched nothing and suggests pairs found in the data.

## Languages

The records and the place names in the data are English, so everything is matched in English.
Input in other languages is handled in three layers:

| Input | How | Model needed |
|---|---|---|
| Digits in any script (`८००००१`), accents (`Puné`) | Normalised before anything else. | No |
| A place name in an Indian script that is listed (`पटना`, `கோவை`, `لکھنؤ`) | `other_scripts` in `location_aliases.yaml`. | No |
| A place name in an Indian script that is not listed (`सीतामढ़ी`) | Read letter by letter (`sitamrhi`) and matched to the place whose consonants are the same. The answer says this was an automatic reading. Nine scripts: Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam. | No |
| A whole question in any language | The model returns the service in English and the place in Latin letters. The answer shows that reading on top. | Yes |

Not handled: place names in Arabic script (Urdu) or other scripts that are not listed, without a model
(the answer asks for English letters or a PIN code); answers in a language other than English; records
whose own text is not English (section 3b counts them; they can never count as evidence).

To add a name: `other_scripts` for a name in another script, `filler_words` for words such as "near".

## Use from code

```python
import carefinder as cf
state = cf.build_state(raw_facilities_df, pin_directory_df)     # once
result, html = cf.find_care(state, "dialysis", "Jaipur")
result, html = cf.ask(state, "My patient needs dialysis near Gaya", llm)   # llm: any function str -> str

tables = cf.state_to_tables(state)        # six pandas DataFrames to save
state = cf.state_from_tables(tables)      # rebuild without re-running the pipeline
```

Ranking is three separate steps: `score_candidates` -> `pick` -> `verdict`.

## Test

```
python test_carefinder.py
```

## Not yet done, and limits

- Service definitions, time targets and place aliases have not been reviewed by a clinician or checked
  against the full data.
- Travel time is a range estimated from straight-line distance, not a route calculation.
- The evidence grades have not been validated by hand (notebook section 6).
- Matching is by keyword rules. It misses unusual wording and can misread a sentence.
