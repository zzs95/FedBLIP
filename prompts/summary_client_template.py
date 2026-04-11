# 📌 Prompt: Section-Specific CTPA Report Template Induction

prompt_test = '''

> **Task**
> You will be provided with multiple radiology report excerpts from **the same medical center**, each excerpt belonging to **the same report section** (either *Findings* or *Impression*) of chest CT pulmonary angiography (CTPA) reports.
> Your task is **not** to rewrite or summarize the content of individual reports, but to **infer the reporting template and writing conventions used for this specific section**.

---

> **Input Description**

* All input texts come from a **single report section** (*Findings* **or** *Impression*).
* All reports are authored within the **same institution**.
* The texts may vary in length and content but share a common reporting style.

---

> **Analysis Instructions**
> From the provided section texts, analyze and infer:

1. **Section Purpose and Scope**

   * What clinical information this section typically conveys
   * Whether it is descriptive, diagnostic, or interpretive

2. **Internal Organization Pattern**

   * Whether content is organized by anatomy, pathology, priority, or narrative flow
   * Typical ordering of abnormalities or statements
   * Use of bullet-like sentences vs. paragraphs

3. **Sentence-Level Writing Patterns**

   * Common grammatical structures (e.g., short declarative sentences, compound sentences)
   * Typical phrasing for:

     * Positive findings
     * Negative findings
     * Uncertainty or limitations
   * Tense and voice (e.g., present tense, passive voice)

4. **Terminology and Style Conventions**

   * Preferred medical terminology and abbreviations
   * Degree of standardization vs. variability
   * Level of diagnostic certainty (definitive vs. suggestive language)

5. **Section-Specific Constraints**

   * What information is usually **included**
   * What is typically **excluded or deferred** to other sections
   * Whether clinical recommendations or follow-up suggestions appear (especially for *Impression*)

---

> **Output Requirements**
> Please structure your output as follows:

### 1. Section Type

Specify whether the analyzed texts correspond to **Findings** or **Impression**.

### 2. Section-Level Reporting Template (Abstract)

Provide a concise, abstract description of how this section is typically written at this center.

### 3. Canonical Content Order

List the typical ordering of statements or topics within this section.

### 4. Language & Style Guidelines

Bullet-point summary covering:

* Tone (objective, assertive, cautious, etc.)
* Sentence length and complexity
* Use of certainty / hedging language
* Common phrasing patterns

### 5. Template Skeleton (with Placeholders)

Provide a **generic template** for this section using placeholders, such as:

```
{Anatomical structure}: {finding description}.
{Negative finding statement}.
{Severity / extent / chronicity (if applicable)}.
```

> **Important Constraints**

* Do **not** copy or paraphrase any individual report verbatim.
* Do **not** include patient-specific or case-specific information.
* Focus on **style, structure, and conventions**, not on clinical correctness of any single case.
* If multiple valid patterns exist, explicitly describe them.

---

> **Goal**
> The final output should define a **center-specific, section-level reporting template** that can be reused for:

* Automated report generation
* Style-consistent text synthesis
* Federated or multi-center modeling of radiology report styles

'''