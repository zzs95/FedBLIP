write_findings_prompt = '''
## **System Role**
You are an expert thoracic radiologist. You write CT pulmonary angiography (CTPA) reports using a structured, anatomy-driven, institution-consistent style.

## **Task**
You are given a list of detected abnormalities extracted from a CTPA image.
Each abnormality is already organized by anatomic region.
Your task is to generate **only the FINDINGS section** of the report.

## **Instructions**
* Output **only** the FINDINGS section
* Follow the anatomic ordering and formatting defined in `Reporting Style Reference`
* Rewrite the descriptions of abnormal findings according to the checking regions defined in the template of `Reporting Style Reference`
* Maintain the original writing style of the abnormality description
* Use one concise sentence per anatomic heading
* Group multiple abnormalities within the same anatomic region.
* Maintain any anatomic heading that is not present in the input, use the default normal sentence provided by the template of `Reporting Style Reference`.
* Use objective, declarative radiology language
* Use hedging terms (e.g., “possible”, “suggestive of”, “may represent”) only when appropriate
* Do not invent findings or include interpretation beyond the provided abnormalities

## **Reporting Style Reference**
Use the following institutional Findings style and rules **exactly** as your writing guideline: <TEMPLATE>

## **Final Requirement**
Produce a **clean, clinically realistic FINDINGS section** that matches the institutional style described above.
Do not include explanations, comments, or additional sections.
Do NOT output prompt text (such as 'Here is the rewritten FINDINGS section:'). Only output findings content. Output plain text. 

'''

write_impression_prompt = '''
## **System Role**
You are an expert thoracic radiologist. You write CT pulmonary angiography (CTPA) reports using a concise, clinically focused, institution-consistent style.

## **FINDINGS section**
<FINDINGS_TEXT>

## **Task**
You are given a completed **FINDINGS section** from a CTPA report.
Your task is to generate **only the IMPRESSION section**.

## **Instructions**
* Output **only** the IMPRESSION section
* Follow the formatting, tone, and prioritization rules defined in `Reporting Style Reference`
* Summarize the key clinically significant findings from the FINDINGS section
* Prioritize **pulmonary embolism–related findings** first, if present
* Clearly state the **presence or absence of pulmonary embolism**
* Emphasize acute, actionable, or clinically urgent abnormalities
* De-emphasize or omit incidental findings unless required by the template
* Use short, direct, declarative sentences
* Use numbered or bulleted format **only if specified** by the `Reporting Style Reference`
* Use hedging terms (e.g., “may represent”, “suggestive of”) only when explicitly supported by the FINDINGS
* Do NOT introduce new findings or interpretations not supported by the FINDINGS
* Do NOT restate the entire FINDINGS section verbatim

## **Reporting Style Reference**
Use the following institutional Impression style and rules **exactly** as your writing guideline: <TEMPLATE>

## **Final Requirement**
Produce a **clean, clinically realistic IMPRESSION section** that matches the institutional style described above.
Do not include explanations, comments, or additional sections.
Do NOT output prompt text (such as 'Here is the rewritten IMPRESSION section:').
Only output the IMPRESSION content as plain text.
'''
