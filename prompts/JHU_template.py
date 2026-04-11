findings_template = '''
### 1. Section Type
**Findings** – the anatomy-based descriptive portion of a chest CT pulmonary angiography (CTPA) report.

### 2. Section-Level Reporting Template (Abstract)
The *Findings* section is written as a **structured, anatomy‑driven list**, with 10 regions. 
  
For each anatomic group, the radiologist either:
* inserts a **concise positive description** when an abnormality is present, or
* supplies a **standardized negative statement (Using the paradigm from 4. Template Skeleton)** when the structure is normal.

Blocks are separated by **line breaks**, not bullets.
The tone is **objective, factual, and non-speculative**. Temporal qualifiers (“stable”, “unchanged”, “new”) are embedded directly in the sentence describing the finding.

### 3. Canonical Content Order
The typical ordering of anatomic blocks in CTPA *Findings* is:
1. **Technical adequacy / contrast timing** – bolus quality and level of opacification
2. **Pulmonary vasculature** – main, lobar, segmental and subsegmental pulmonary arteries
3. **Systemic thoracic vasculature** – thoracic aorta and major vessels
4. **Heart & pericardium** – cardiac size, chambers, pericardial effusion, calcifications
5. **Mediastinum & hila** – lymph nodes and central structures
6. **Airways** – trachea and bronchial tree
7. **Lung parenchyma** – consolidations, ground-glass, nodules, emphysema, fibrosis, atelectasis
8. **Pleura** – effusion, pneumothorax, pleural disease
9. **Thyroid / esophagus / chest wall soft tissues**
10. **Musculoskeletal / skeletal** – bones and surgical hardware

*If an anatomic system is entirely normal, a standardized negative sentence is still included to preserve checklist completeness.*

### 4. Template Skeleton (with 10 regions)
FINDINGS:
Technical adequacy: {e.g., “Adequate bolus timing with opacification of the pulmonary arteries to the subsegmental level.”}
Pulmonary arteries: {e.g., “No filling defects identified in the main, lobar, or segmental pulmonary arteries. Main pulmonary artery is normal in caliber.”}
Systemic thoracic vessels: {e.g., “Thoracic aorta normal in caliber without aneurysm or dissection.”}
Heart and pericardium: {e.g., “Heart size within normal limits; no pericardial effusion.”}
Mediastinum and hila: {e.g., “No mediastinal or hilar lymphadenopathy.”}
Airways: {e.g., “Trachea and mainstem bronchi are patent.”}
Lung parenchyma: {e.g., “No focal consolidation; no suspicious pulmonary nodules.”}
Pleura: {e.g., “No pleural effusion or pneumothorax.”}
Thyroid, esophagus, and chest wall soft tissues: {e.g., “Visualized thyroid gland is unremarkable.”}
Musculoskeletal: {e.g., “No acute osseous abnormality; no suspicious osseous lesions.”}

*Notes:*
* Each anatomic block is always presented to preserve the checklist structure.
* When multiple findings occur in one system, they are combined into a single sentence using commas or conjunctions.
* Size, laterality, chronicity, and comparison with prior imaging are embedded directly into the finding sentence.
* If the input contains no abnormality (or no content) for that block, use the default normal sentence inside \{\}.

### 5. Language & Style Guidelines

| Aspect                   | Guideline                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Tone**                 | Objective, factual, assertive; no clinical speculation beyond image evidence.                                               |
| **Sentence length**      | Short (≈12–18 words). One sentence per anatomic block; two only when size or chronicity must be specified.                  |
| **Verb tense / voice**   | Present simple (“are patent”, “shows”). Negative findings often use impersonal passive (“is not visualized”, “are absent”). |
| **Positive findings**    | *Structure*: **{Anatomic system}: {Finding with location + descriptor + size if applicable}.**                              |
| **Negative findings**    | Standardized phrases (e.g., “No pleural effusion or pneumothorax.”).                                                        |
| **Uncertainty language** | Explicit hedging when required: “possible”, “suggestive of”, “cannot be excluded”.                                          |
| **Temporal qualifiers**  | Embedded directly after the descriptor: “stable”, “unchanged from prior”, “new”.                                            |
| **Abbreviations**        | Limited to common radiology usage (CTPA, PE, RLL, LUL, RV/LV ratio). No institution-specific shorthand.                     |
| **Punctuation**          | Period at the end of every sentence; commas for grouping related findings.                                                  |

**Result:**
This rewritten template preserves the **anatomy-first, checklist-driven CTPA Findings style**, while matching the **format, hierarchy, and clarity** of the BUH example.
It is directly suitable for **LLM prompting, report-template mining, or center-aware automated report generation**.
'''


impression_template = '''
**Impression-Section Writing Template (CTPA – Findings → Impression)**

### 0) Output format (mandatory)
- Output a single block that starts with **IMPRESSION:**.
- Use a **numbered list** (1., 2., 3., ...).
- **Item #1 must always state the pulmonary embolism conclusion** (positive / negative / limited / residual).
- Omit non-applicable items and **renumber sequentially** without gaps.
- Use short, declarative radiology sentences (usually **one sentence per item**, max 1–2).
- Do not repeat the same finding in multiple items; consolidate bilateral/multifocal disease.

### 1) What must be retained from the Findings (include only if present or management-relevant)

| Category | Findings elements that must be reflected in the Impression |
|----------|------------------------------------------------------------|
| **Pulmonary embolism (always address first)** | Presence/absence; distribution (main/lobar/segmental/subsegmental, side, lobe); occlusive vs non-occlusive (if stated); acuity (acute/chronic/residual) and interval change (new/unchanged/decreased/increased/resolved); study limitations that affect confidence. |
| **Right-heart strain (only if mentioned in Findings)** | Present vs not evident; supporting signs if provided (RV/LV ratio, septal flattening/bowing, IVC/hepatic reflux). |
| **Pulmonary parenchyma (only the dominant clinically important pattern)** | Consolidation / ground-glass opacities / tree-in-bud / mucus impaction / infarct-type change; distribution (lobes) and likely pattern **only when supported** (infection/aspiration/infarct/organizing pneumonia/COVID-like). |
| **Pleura (only if clinically relevant)** | Effusion (side/size/new vs stable; malignant only if stated); pneumothorax (size/tube). |
| **Cardiovascular & major thoracic vessels (only if significant)** | Enlarged main pulmonary artery (include value if given; “may be seen with pulmonary hypertension”); pericardial effusion (trace/mild/moderate); important aortic abnormality/ectasia (include value if given). |
| **Mediastinum / lymph nodes (only if significant)** | Bulky or enlarging nodes; interval change; “likely reactive” vs “suspicious” only if stated or clearly implied in Findings. |
| **Extra-thoracic (only if highlighted as significant/new)** | Upper-abdomen or other findings (e.g., adrenal lesion, AAA) **only if** Findings calls it significant/new or provides follow-up context. |
| **Recommendation (optional; only if explicitly required by Findings)** | Include as final numbered item starting with “Recommend …”. |

*All other normal/unchanged details may be omitted.*

### 2) Style & tone rules (institutional)
- **Objectivity:** factual radiology language; no narrative (“the study shows…”).
- **Prioritization:** PE → right-heart strain → dominant lung abnormality → other significant findings → recommendation (if any).
- **Certainty language:** definitive when criteria met; hedge only when unavoidable and supported.
  - Definite: “Acute pulmonary embolism in …”
  - Qualified: “Findings compatible with …”, “may represent …”, “suggestive of …”
- **Brevity:** 1–2 sentences per item; keep total impression typically ≤ 8–10 lines.
- **Standard abbreviations:** PE, RV/LV, IVC, CTPA. Avoid institution-specific acronyms.
- **No invention:** do not add etiologies, interval change, or follow-up unless supported by Findings.

### 3) Canonical ordering (observed in institutional examples)
1. **Pulmonary embolism status** (present/absent/limited/residual; interval change if stated).
2. **Right-heart strain** (only if Findings mentions).
3. **Dominant pulmonary parenchymal abnormality** (infection/bronchiolitis/pneumonia/OP-pattern/infarct; distribution).
4. **Pleura** (effusion/pneumothorax if relevant).
5. **Other major clinically significant findings** (mass/adenopathy/MPA enlargement/aorta/pericardial effusion/esophagus).
6. **Recommendation** (optional; only if explicitly required).

### 4) Required item #1: Pulmonary embolism phrasing patterns
Choose exactly ONE pattern for item #1:

A) Definitive negative:
- `1. No pulmonary embolism.`
  (Optional only if style commonly uses it)
- `1. No pulmonary embolism or other acute finding within the chest.`

B) Limited study but central arteries negative:
- `1. No evidence of pulmonary embolism in the main and lobar pulmonary arteries; evaluation of the subsegmental arteries is limited due to {reason}.`

C) No acute PE but residual/chronic PE with interval change:
- `1. No evidence of acute pulmonary embolism; minimal residual emboli in the {location}, {decreased/unchanged} compared with prior.`
- `1. No evidence of acute pulmonary embolism; chronic pulmonary emboli in the {location}, unchanged.`

D) Positive PE:
- `1. {Acute/Chronic} {occlusive/nonocclusive} pulmonary embolus in the {right/left/bilateral} {main/lobar/segmental/subsegmental} pulmonary arteries, {key extent/location}, {interval change if stated}.`

### 5) Optional item patterns (use only if supported by Findings)

**Right-heart strain**
- `#. No evidence of right heart strain.`
- `#. Evidence of right heart strain ({RV/LV ≈ x / septal flattening/bowing / IVC reflux}).`
- `#. Possible right heart strain ({reason}).`

**Dominant lung abnormality (choose ONE main pattern)**
- Bronchitis/bronchiolitis:
  - `#. Bronchial wall thickening with tree-in-bud nodularity/mucus impaction, most compatible with infectious/inflammatory bronchiolitis/bronchitis.`
- Pneumonia/aspiration:
  - `#. Lobar consolidation in the {lobe}, consistent with pneumonia versus aspiration pneumonitis.`
- Ground-glass / organizing pneumonia / COVID-style wording (only if Findings supports it):
  - `#. Peripheral ground-glass opacities with an organizing pneumonia pattern; imaging features can be seen with COVID-19 pneumonia.`
  - Optional differential line (only if consistent with style in Findings/Impression examples):
    - `#. Other processes such as influenza pneumonia and organizing pneumonia (e.g., drug toxicity or connective tissue disease) can produce a similar pattern.`
- Infarct / PE sequelae (only if supported):
  - `#. Patchy peripheral opacities in the {lobes}, possibly sequelae of pulmonary infarct.`
- Entirely negative lungs (only if institution commonly states it):
  - `#. No focal consolidation to suggest pneumonia.`

**Pleura**
- `#. No pleural effusion or pneumothorax.`
- `#. {Small/moderate/large} {right/left} pleural effusion, {new/unchanged}; {may be malignant} if stated.`
- `#. {Small/moderate} pneumothorax, {with/without} chest tube.`

**Other major findings (include only if management-relevant)**
- Pulmonary hypertension indicator:
  - `#. Mildly enlarged main pulmonary artery ({value} cm), which may be seen with pulmonary hypertension.`
- Pericardial effusion:
  - `#. {Trace/mild/moderate} pericardial effusion.`
- Mass / malignancy / bulky nodes:
  - `#. {Large mass description} with {bronchovascular encasement/mediastinal invasion} and {bulky adenopathy}, consistent with {known malignancy if stated}.`
- Aorta:
  - `#. Stable ectatic/aneurysmal ascending aorta measuring {value} cm.`
- Esophagus/hiatal hernia:
  - `#. Distal esophageal wall thickening/hiatal hernia, which may represent esophagitis.`

**Recommendation (final item only; if explicitly required)**
- `#. Recommend {follow-up CT in X months / echocardiography / clinical correlation} as indicated.`

'''