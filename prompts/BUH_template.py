findings_template = '''
### 1. Section Type  
**Findings** – the anatomic‑descriptive portion of a chest CT pulmonary angiography (CTPA) report.

### 2. Section‑Level Reporting Template (Abstract)  
The *Findings* section is written as a **structured, anatomy‑driven list**, with 7 regions.   
Each anatomic sub‑heading appears on its own line (often preceded by a colon) and is followed by a single short sentence that either describes a positive abnormality or states “Normal” / “No acute abnormality”.  When several abnormalities affect the same structure they are grouped in one sentence, separated by commas or conjunctions.  
The overall tone is **objective, declarative and concise**; uncertainty is expressed with hedging verbs such as “possible”, “suggestive of”, “may represent”.  Time‑related comments (“stable”, “unchanged”, “new”) are inserted directly into the sentence describing the finding.  

### 3. Canonical Content Order  
The typical ordering of headings (most frequent → least frequent) is:
1. **Pulmonary arteries** – emboli, filling defects, thrombus burden, size of main PA, reflux, etc.  
2. **Lungs and Airways** – parenchymal opacities, air‑space disease, atelectasis, emphysema, nodules, masses, ground‑glass, scarring, infection, etc.  
3. **Pleura** – effusion, pneumothorax, thickening, loculated collections, normal.  
4. **Heart** (sometimes combined with mediastinum) – size, cardiomegaly, coronary calcifications, pericardial effusion, septal bowing, right‑heart strain, postoperative changes.  
5. **Mediastinum and Hila** – adenopathy, lymph nodes, aortic/vascular calcification, normal.  
6. **Chest Wall and Lower Neck** – devices (PICC, ports, endotracheal tube), surgical wires, prostheses, hiatal hernia, thyroid nodules, normal.  
7. **Bones** – acute fracture, degenerative change, healed fractures, sclerotic lesions, normal.  

*If a heading is not applicable it is omitted; the report never includes headings that contain no comment.*

### 4. Template Skeleton (with 7 regions)
FINDINGS:
Pulmonary arteries: {Embolus/filling defect description – e.g., "There are bilateral segmental emboli extending into the right lower lobe, with a saddle thrombus in the main PA."}
Lungs and Airways: {Parenchymal description – e.g., "Well‑expanded lungs with bibasilar ground‑glass opacities; 6 mm nodule in left lower lobe (stable)." }
Pleura: {Finding or "Normal" – e.g., "Small right pleural effusion."}
Heart: {Size/strain description – e.g., "Cardiomegaly with mild coronary calcification; no RV/LV bowing."}
Mediastinum and Hila: {Lymph node or vascular comment – e.g., "No significant mediastinal adenopathy."}
Chest Wall and Lower Neck: {Device/surgical note or "Normal" – e.g., "Right PICC line tip in SVC; median sternotomy wires intact."}
Bones: {Acute/degenerative comment – e.g., "No acute abnormality. Multilevel degenerative changes of the thoracic spine."}

*Notes:*  
- Any heading that does not apply to a particular study is simply omitted.  
- When multiple abnormalities belong to the same heading they are concatenated with commas or “and”.  
- Temporal comparison (e.g., “stable”, “new”) and uncertainty language are inserted directly in the placeholder text.  

### 5. Language & Style Guidelines  

| Aspect | Guideline |
|--------|-----------|
| **Tone** | Objective, factual, assertive. Use neutral adjectives (“well‑expanded”, “clear”) and avoid subjective judgments. |
| **Sentence length** | One‑sentence per heading; sentences are short (10–20 words). Compound sentences only when grouping several related findings of the same structure. |
| **Verb tense / voice** | Present simple (“are normal”, “shows”), occasional past (“was noted”) for comparison with prior studies. Predominantly active voice. |
| **Positive findings** | *Structure*: **{Anatomic area}: {Finding description}**.<br>Examples: “Pulmonary arteries: There are bilateral segmental emboli extending into the right lower lobe.” |
| **Negative findings** | *Structure*: **{Anatomic area}: Normal.** or **No acute abnormality.** |
| **Uncertainty / limitation** | Use hedging verbs/adverbs: “possible”, “suggestive of”, “may represent”, “limited by motion artifact”. |
| **Temporal qualifiers** | Insert directly after the finding: “stable”, “new”, “unchanged from prior”, “improved compared with…”. |
| **Abbreviations** | Use only widely accepted CT‑PA abbreviations (e.g., PA, RV/LV ratio, IVC). No institution‑specific acronyms are introduced. |
| **Punctuation** | Colon after heading; period at end of each sentence. Commas separate multiple findings within the same sentence. |

**Result:** 
This abstracted template captures the chest‑CTPA *Findings* style—anatomically ordered, concise bullet‑like sentences with a consistent set of headings, objective phrasing, and limited use of hedging. 
It can be reused for automated report generation or as a guide for style‑consistent manual reporting.
'''

impression_template = '''
**Impression-Section Writing Template (BUH CTPA – Findings → Impression)**

### 0) Output format (mandatory)
- Output a single block that starts with **IMPRESSION:**.
- Use a **numbered list** (1., 2., 3., ...).
- **Item #1 must always state the pulmonary embolism conclusion** (positive / negative / limited / residual / chronic).
- Omit non-applicable items and **renumber sequentially** without gaps.
- Write in short, declarative radiology sentences (usually **one sentence per item**, max 1–2).
- Consolidate bilateral/multifocal disease into one item when possible; avoid duplication across items.
- BUH style commonly mixes sentence-case and ALL CAPS; unless explicitly instructed, use sentence-case.

### 1) What must be retained from the Findings (include only if present or management-relevant)

| Category | Findings elements that must be reflected in the Impression |
|----------|------------------------------------------------------------|
| **Pulmonary embolism (always address first)** | Presence/absence; distribution (main/lobar/segmental/subsegmental; side/lobe); clot burden qualifiers (small/moderate/large; “extensive”; “massive”; “saddle”); occlusive vs nonocclusive if stated; acuity (acute/chronic/subacute/residual) and interval change (new/worsened/decreased/resolved); explicit statements of clinical significance/uncertainty for tiny SSPE (“clinical significance uncertain”). |
| **Right-heart strain / RV dysfunction (very common in BUH positives)** | Present vs not evident vs equivocal; supporting CT signs used in BUH sample: RV/LV ratio thresholding (“> 1”), septal straightening/flattening/bowing, RV dilation, contrast reflux to hepatic veins; “submassive” wording sometimes used with saddle + strain. |
| **Pulmonary infarct / PE-related parenchymal change (often paired with PE)** | “Probable pulmonary infarct” and its location (lingula/RLL etc.); infarct vs pneumonia phrasing when nonenhancing consolidation; “cannot be excluded” language is used. |
| **Dominant lung/airway process (if not purely infarct)** | Aspiration/pneumonia/atelectasis; groundglass or airspace disease; tree-in-bud or airway disease; BUH tends to write **one consolidated statement** with a short differential (pneumonia vs aspiration vs atelectasis). |
| **Pleura** | Effusions (side/size; tiny/trace/small/moderate; unilateral/bilateral); pneumothorax (size; tension physiology if stated). |
| **Mass / nodules / malignancy / lymph nodes (frequent BUH add-ons)** | Pulmonary nodules/mass (size; interval progression); “suspicious for metastatic disease”; mediastinal/hilar adenopathy (progression; reactive vs neoplastic wording); lymphangitic carcinomatosis if stated. |
| **Major vessels / cardiac / other thoracic** | Pulmonary artery enlargement suggesting pulmonary HTN (with measurement); cardiomegaly/CAD; aortic flap/dissection mention when present; pericardial effusion if stated. |
| **Devices / critical communication / recommendations** | Explicit “STAT results communicated…” blocks may appear; keep placeholders/time stamps if present. Recommendations are relatively common in BUH positives (pulmonary consult, PET/CT/biopsy, follow-up CT at 6 weeks/3 months/12 months), and should be preserved only when present/required. |

*All other normal/unchanged details may be omitted.*

### 2) Style & tone rules (BUH-specific)
- **PE is the organizing axis**: even when many findings exist, BUH impressions lead with PE (or limitation/negative statement) first.
- BUH frequently uses **burden adjectives** (small/moderate/large; extensive/massive) and **severity framing** (e.g., “submassive” when saddle + strain is described).
- **Right-heart strain language is explicit** and often appears as a clause in item #1 or as item #2.
- **Compact differentials** are acceptable (“pneumonia vs aspiration”, “infarct vs pneumonia”) but do not over-elaborate.
- **Follow-up** is common for nodules/masslike opacity and for pneumonia/aspiration (“follow-up CT in 6 weeks/3 months”) when explicitly stated.

### 3) Canonical ordering (observed in BUH positive-PE examples)
1. **Pulmonary embolism conclusion** (include extent/burden; sometimes includes strain clause).
2. **Right-heart strain / RV dysfunction** (if present or explicitly negated; may be merged into #1 if brief).
3. **Pulmonary infarct / dominant PE-related parenchymal finding** (if present).
4. **Dominant non-infarct lung/airway process** (pneumonia/aspiration/atelectasis/groundglass; if present).
5. **Pleura** (effusions/pneumothorax).
6. **Malignancy / nodules / lymph nodes** (mass, nodules, metastatic progression, adenopathy).
7. **Other major findings** (pulmonary HTN cue, CAD/cardiomegaly, aorta, esophagus).
8. **Recommendation / communication** (final item(s), only if present/required).

### 4) Required item #1: Pulmonary embolism phrasing patterns (BUH)
Choose exactly ONE pattern for item #1:

A) Definitive negative:
- `1. No evidence of pulmonary embolism.`
- Variants: `No evidence of pulmonary embolus.` / `No evidence of PE.`

B) Limited study with partial exclusion:
- `1. {Limited/poor} opacification and/or motion; no evidence of large central pulmonary embolism.` 
- `1. Exam limited; findings suspicious for {segmental/subsegmental} pulmonary emboli in the {location}.`

C) Very small / uncertain SSPE:
- `1. Possible very small subsegmental embolus in the {location}; clinical significance is uncertain given its diminutive size.`

D) Chronic / residual / subacute / interval change:
- `1. Chronic pulmonary embolism in the {location}.`
- `1. Subacute pulmonary embolism in the {location}.`
- `1. Interval {improvement/decrease/resolution} in pulmonary embolus in the {location}; no new pulmonary emboli identified.`

E) Positive PE (with burden and distribution; BUH commonly explicit):
- `1. {Acute} pulmonary emboli involving the {right/left/bilateral} {main/lobar/segmental/subsegmental} pulmonary arteries {extent}, with {small/moderate/large} clot burden.`
- Saddle / massive:
  - `1. Saddle pulmonary embolus extending into the main and segmental pulmonary arteries bilaterally.`
  - `1. Massive/extensive bilateral pulmonary emboli with {saddle thrombus if stated}.`
- With RV dysfunction clause (optional if explicitly stated):
  - `1. {Acute} {extensive} bilateral pulmonary emboli … with CT evidence of right heart strain.`

### 5) Optional item patterns (BUH phrase bank; use only if supported)

**Right-heart strain / RV dysfunction**
- `#. No evidence of right heart strain.`
- `#. CT evidence of right heart strain ({RV/LV ratio > 1 / septal straightening/flattening/bowing / RV dilatation / reflux into hepatic veins}).`
- `#. Borderline elevation of the RV/LV ratio.`

**Pulmonary infarct (common in BUH positives)**
- `#. Findings most compatible with {right/left} lower lobe/lingular pulmonary infarct.`
- `#. Nonenhancing consolidated lung in the {lobe} could represent infarct versus pneumonia.`
- `#. Pulmonary infarction cannot be excluded.`

**Dominant lung/airway process (choose ONE; BUH often uses short differential)**
- `#. Bibasilar/lingular airspace disease, which may represent atelectasis, aspiration, pneumonia, or a combination.`
- `#. Patchy bilateral airspace disease with interlobular septal thickening, likely multifocal pneumonia or aspiration with superimposed edema.`
- `#. Groundglass opacities … may be infectious/inflammatory; mild pulmonary edema is also considered.`
- `#. Bibasilar discoid atelectasis.`

**Pleura**
- `#. {Tiny/trace/small/moderate} {right/left/bilateral} pleural effusion(s) with adjacent atelectasis.`
- `#. {Small/moderate} pneumothorax.`

**Malignancy / nodules / lymph nodes**
- Nodules/mass:
  - `#. Multiple pulmonary nodules/mass, {indeterminate/suspicious for metastatic disease}; {interval progression if stated}.`
- Adenopathy:
  - `#. Mediastinal/hilar adenopathy, {reactive vs neoplastic}; follow-up imaging should be considered if stated.`
- Lymphangitic:
  - `#. Irregular septal thickening suggestive of lymphangitic carcinomatosis.`

**Other major findings**
- Pulmonary hypertension cue:
  - `#. Enlarged main pulmonary artery measuring {value} cm, which can be seen with pulmonary hypertension.`
- CAD/cardiomegaly:
  - `#. Severe coronary artery disease / cardiomegaly.` 
- Aorta:
  - `#. Focal intimal flap in the descending thoracic aorta concerning for dissection of uncertain chronicity.`

**Recommendation / communication (final item only; only if present/required)**
- Follow-up CT:
  - `#. Recommend follow-up chest CT in {6 weeks/3 months/12 months} {after treatment/for stability} as stated.`
- PET/biopsy/consult:
  - `#. Recommend pulmonary consultation and consideration of PET/CT or biopsy.`
- STAT communication:
  - `#. STAT results were communicated to {service/provider} at <TIME> on <DATE>.`
'''
