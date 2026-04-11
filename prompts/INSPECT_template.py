impression_template = '''
**Impression-Section Writing Template (CTPA – Findings → Impression)**

### 0) Output format (mandatory)
- Output a single block that starts with **IMPRESSION:**.
- Use a **numbered list** (1., 2., 3., ...).
- **Item #1 must always state the pulmonary embolism conclusion** (positive / negative / limited / residual / chronic).
- Omit non-applicable items and **renumber sequentially** without gaps.
- Use short, declarative radiology sentences (usually **one sentence per item**, max 1–2).
- Do not repeat the same finding in multiple items; consolidate bilateral/multifocal disease.
- Preserve placeholders and callouts exactly if present: `<DATE>`, `<TIME>`, `<HCW>`.
- Some impressions are **ALL CAPS**. Match the casing style of the input report if specified; otherwise use sentence-case.

### 1) What must be retained from the Findings (include only if present or management-relevant)

| Category | Findings elements that must be reflected in the Impression |
|----------|------------------------------------------------------------|
| **Pulmonary embolism (always address first)** | Presence/absence; distribution (main/lobar/segmental/subsegmental; side/lobe); clot burden qualifier if present (small/moderate/large; saddle; worsening/decreased); acuity (acute/chronic/residual); interval change (new/unchanged/improved/resolved); technique limitations that affect confidence (motion/suboptimal bolus/poor opacification/body habitus) and what level is confidently excluded (central/lobar/segmental). |
| **Right-heart strain (common with positive PE; also stated explicitly when absent)** | Present vs not evident vs equivocal; supporting signs if provided (RV/LV ratio, septal flattening/bowing, reflux, “hemodynamic impact”). Echo correlation sometimes included when equivocal. |
| **Dominant lung / airway process (frequently includes these)** | Pneumonia/aspiration (often linked to fluid-filled/patulous esophagus, debris/secretions); bronchitis/bronchiolitis (bronchial wall thickening, mucus plugging, tree-in-bud); COVID/viral/OP language; pulmonary edema/volume overload/ARDS; infarct (esp. adjacent to PE); ILD/fibrosis/post-transplant/post-radiation when clinically central. |
| **Pleura** | Effusion (side/size/new vs increased; malignant only if stated); pneumothorax (side/size; tension physiology or mediastinal shift if stated); hydrothorax/tension hydrothorax when described. |
| **Major vessels / cardiovascular** | Aorta acute injury/dissection/aneurysm exclusion or positive dissection/intimal flap; main pulmonary artery enlargement (“suggesting pulmonary hypertension”); cardiomegaly/coronary calcifications; pericardial effusion (new/moderate); intracardiac/pericardial metastases if stated; device-related thrombus if stated. |
| **Malignancy / metastases / lymph nodes** | Dominant mass progression, lymphangitic carcinomatosis; pleural metastatic disease; interval progression of nodal disease; statements like “concerning for progression” appear often. |
| **Trauma / musculoskeletal** | Fractures, acute injury exclusions (especially in trauma-style impressions). |
| **Devices / iatrogenic / critical items** | Line/tube malposition (e.g., ETT in right mainstem → recommend retraction); drains (kinked) if stated; endovascular stent graft and complications. |
| **Recommendation / communication** | Follow-up CT per Fleischner (often pasted as multi-line); “Recommend correlation with echocardiogram”; “May consider V/Q scan or dedicated CTPA”; “These results were called to <HCW> at <TIME> on <DATE>.” Include only when present/required by Findings/Impression style. |

*All other normal/unchanged details may be omitted.*

### 2) Style & tone rules
- **Prioritization is still PE-first**, but often starts with broader exclusions in trauma-like cases (aorta injury / large PE / pneumothorax) when that is the report style.
- **Hedging is common** when limited study or mixed artifact: “cannot exclude…”, “difficult to exclude…”, “may be…”, “indeterminate…”.
- **Differential statements appear** in some impressions (e.g., edema vs infection vs lymphangitic spread; infarct vs infection).
- **Long guideline blocks may appear** (Fleischner recommendations, follow-up intervals). If present, preserve structure and wording style; otherwise do not add.

### 3) Canonical ordering
1. **Pulmonary embolism status** (or “no large PE / no central PE” if limited).
2. **Right-heart strain / hemodynamic impact** (if PE or explicitly discussed).
3. **Dominant lung/airway process** (pneumonia/aspiration/bronchiolitis/COVID/edema/ARDS/infarct).
4. **Pleura** (effusions/pneumothorax/tension physiology).
5. **Other major findings** (aorta/dissection, malignancy/metastases/nodes, pericardial effusion, pulmonary hypertension cues, devices/lines, trauma).
6. **Recommendations / communication** (only if present or required).

### 4) Required item #1: Pulmonary embolism phrasing patterns
Choose exactly ONE pattern for item #1:

A) Definitive negative:
- `1. No evidence of pulmonary embolism.`
- Variants: `No evidence of pulmonary embolus.` / `No evidence of PE.` / `NEGATIVE FOR PULMONARY EMBOLISM.`

B) Limited study, but negative to a stated level:
- `1. Exam limited by {motion/suboptimal opacification/body habitus}; within this limitation, no {central/lobar/segmental} pulmonary embolism.`
- `1. No large central or lobar pulmonary embolism; respiratory motion limits exclusion of segmental/subsegmental emboli.`

C) Indeterminate focal defect / artifact:
- `1. {Filling defect description} is indeterminate for pulmonary embolus given technique; pulmonary embolus is difficult to exclude.`
  (Optionally followed by a dedicated workup recommendation only if present in the report style.)

D) Residual / chronic / interval change:
- `1. Interval {improvement/resolution/decrease} in previously seen pulmonary embolus in the {location}; no new pulmonary emboli identified.`
- `1. Findings consistent with chronic pulmonary embolism.`
- `1. No acute pulmonary thromboembolism; minimal residual {linear/nonocclusive} filling defects in the {location}, likely chronic sequela.`

E) Positive PE (often with burden qualifier):
- `1. {Acute/Chronic} pulmonary emboli in the {right/left/bilateral} {main/lobar/segmental/subsegmental} pulmonary arteries {extent}; {small/moderate/large} clot burden {if stated}.`
- Saddle variant:
  - `1. Acute saddle pulmonary embolus with {moderate/large} clot burden {and evidence of right heart strain if stated}.`
- Worsening burden variant:
  - `1. Worsening burden of pulmonary embolus, including new embolus in the {location}; {no evidence of right heart strain if stated}.`

### 5) Optional item patterns (phrase bank; use only if supported)

**Right-heart strain**
- `#. No evidence of right heart strain.`
- `#. Evidence of right heart strain ({RV/LV ≈ x / septal flattening/bowing / reflux}).`
- `#. Questionable interventricular septal flattening; recommend correlation with echocardiogram.`
- `#. Positive for right ventricular strain.`

**Dominant lung/airway process (choose ONE main pattern, may add a brief qualifier)**
- Pneumonia/aspiration:
  - `#. Bilateral {dependent/basilar} consolidation/patchy opacities, concerning for pneumonia, possibly related to aspiration.`
  - `#. Fluid-filled/patulous esophagus and/or debris in airways raises concern for aspiration.`
- Bronchitis/bronchiolitis:
  - `#. Bronchial wall thickening with mucus plugging and tree-in-bud nodularity, consistent with bronchitis/bronchiolitis; aspiration or infection is a consideration.`
- COVID/viral/OP/ARDS:
  - `#. Extensive bilateral groundglass opacities with architectural distortion, compatible with COVID-19 infection.`
  - `#. Mixed consolidation and ground-glass disease suggestive of organizing pneumonia; viral or bacterial infection can appear similar.`
  - `#. Multifocal airspace and groundglass opacities with rounded morphology and areas of consolidation, concerning for evolving ARDS in the setting of COVID-19.`
- Edema/volume overload:
  - `#. Interlobular septal thickening and groundglass opacities with small pleural effusions, compatible with pulmonary edema/volume overload.`
  - `#. Early lymphangitic carcinomatosis can have a similar appearance; attention on follow-up.`
- Infarct:
  - `#. Focal peripheral airspace opacity adjacent to embolus, favored pulmonary infarct/hemorrhage.`
- Chronic baseline disease (if central):
  - `#. Findings consistent with {idiopathic pulmonary fibrosis/post-radiation change/post-transplant status}, {mildly progressed/improved} compared to <DATE>.`

**Pleura**
- `#. {Small/moderate/large} {right/left/bilateral} pleural effusion, {new/increased/unchanged}.`
- `#. New right tension hydrothorax with complete right lung collapse and leftward mediastinal shift; effusion is likely malignant if stated.`
- `#. {Trace/small/moderate} pneumothorax, {with/without} mediastinal shift; {tube/drains} if present.`
- `#. Loculated pleural effusion concerning for malignant effusion.`

**Other major findings**
- Aorta / acute injury:
  - `#. No evidence of thoracic aortic dissection or aneurysm.`
  - `#. Nonspecific thin intimal flap in the proximal descending thoracic aorta, concerning for localized dissection (type B).`
  - `#. Contrast timing limits evaluation of the pulmonary arteries and/or aorta.` (only if stated)
- Pulmonary hypertension cue:
  - `#. Enlarged main pulmonary artery ({value} cm), suggesting pulmonary arterial hypertension.`
- Pericardium / cardiac:
  - `#. New {moderate} pericardial effusion.`
  - `#. Suspected intracardiac/pericardial metastases with associated effusion.` (if stated)
- Malignancy/metastases/nodes:
  - `#. Progressive metastatic disease with enlarging thoracic lymphadenopathy and pulmonary/pleural metastases.`
  - `#. Increased mediastinal/hilar/supraclavicular lymphadenopathy compared to <DATE>, concerning for progression.`
- Devices / iatrogenic:
  - `#. Endotracheal tube tip in the right mainstem bronchus; recommend retraction.`
  - `#. Catheter-associated thrombus at port tip; SVC patent.`

**Recommendation / communication (final item only; only if present/required)**
- `#. Recommend {echocardiogram / follow-up CT in X months / V/Q scan / dedicated CTPA} if clinically indicated.`
- `#. These results were called to <HCW> at <TIME> on <DATE>.`
'''