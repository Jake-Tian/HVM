# robot.json ground truth analysis

**Scope:** `data/questions/robot.json`  
**Total QA pairs:** 1,276 across 100 videos  
**Focus:** `answer` field (and `question` where it affects interpretation).

---

## 1. Definite errors in answers

### 1.1 Typo: `"ye"` → `"Yes"`

| question_id     | question                       | answer | suggested |
|-----------------|--------------------------------|--------|-----------|
| kitchen_23_Q04  | Does Sunnycare about Alice?    | `ye`   | `Yes`     |

---

### 1.2 Typo: `"letteuce"` → `"lettuce"`

| question_id    | question                                                                 | answer                                   | suggested                                |
|----------------|---------------------------------------------------------------------------|------------------------------------------|------------------------------------------|
| kitchen_02_Q01 | What the ingredients brought by Wiliiam's mother are available for dinner?| `Tomatoes, letteuce and mushrooms`       | `Tomatoes, lettuce and mushrooms`        |
| kitchen_02_Q05 | What vegetables should the robot put into the sink?                      | `Tomatoes, letteuce and mushrooms`       | `Tomatoes, lettuce and mushrooms`        |

---

### 1.3 Typo: `"soup opera"` → `"soap opera"`

| question_id    | question / answer snippet                            | answer                                                | suggested |
|----------------|-------------------------------------------------------|-------------------------------------------------------|-----------|
| kitchen_23_Q03 | (Why did Sunny fail to find the cherry tomatoes…)     | `She watched the soup opera for too long.`            | `She watched the soap opera for too long.` |
| kitchen_23_Q05 | (Why did Sunny fail to find the cherry tomatoes…)     | `Because she was addicted to the soup opera.`         | `Because she was addicted to the soap opera.` |

---

## 2. Answer fields that are fine

- **Chinese in `answer`:** 0. All Chinese is in `reasoning` only; `answer` is English-only.
- **Answers that are questions (e.g. ending with `?`):** 0.
- **Answer identical to question:** 0.
- **Placeholder-like (`Unknown`, `N/A`, `-`, etc.):** 0.
- **Purely numeric answers for “How much” / “How many”:** Valid (e.g. `7000`, `600`, `3`).
- **Duplicate `question_id`s:** 0.
- **Very long or “reasoning-like” paste in `answer`:** 0.

---

## 3. Style / consistency (optional to change)

### 3.1 Yes/No capitalization

- **`"Yes"` / `"No"` (or `"Yes."` / `"No."`):** 112 + 64 + 29 + 24 = 229.
- **`"yes"` / `"no"` (lowercase):** 15 + 9 = 24.

Lowercase is internally consistent; changing to `Yes`/`No` would align with the majority. Affected `question_id`s (examples):  
`study_08_Q08`, `study_05_Q04`, `study_05_Q11`, `study_11_Q03`, `bedroom_10_Q01`, `kitchen_02_Q04`, `kitchen_02_Q11`, `study_12_Q06`, `study_07_Q10`, `kitchen_04_Q12`, … (24 total).

---

## 4. Question-text issues (for reference only)

These are in the `question` field; they do not change the `answer` but can affect parsing and evaluation.

| question_id       | issue                          | example / note                                      |
|-------------------|--------------------------------|-----------------------------------------------------|
| kitchen_23_Q04    | `Sunnycare`                    | → `Sunny care`                                      |
| kitchen_14_Q02, Q03 | `pratice`                    | → `practice`                                        |
| kitchen_02_Q01    | `Wiliiam`                      | → `William`                                         |
| kitchen_02_Q01    | `What the ingredients`         | → `What are the ingredients` or `What ingredients`  |
| study_12_Q06      | `get involve`                  | → `get involved`                                    |
| study_21_Q04      | `Dose`                         | → `Does`                                            |
| meeting_room_05_Q09, Q10 | `Dose`                  | → `Does`                                            |
| kitchen_23_Q03 (question) | `soup opera`           | → `soap opera`                                      |

---

## 5. Summary

| Category                      | Count | Action                          |
|------------------------------|-------|---------------------------------|
| Definite answer errors       | 5     | Recommend fixing (see §1)       |
| Style (Yes/No capitalization)| 24    | Optional                        |
| Question-text typos          | 8     | Optional; for reference         |

**Answer errors to fix (5):**

1. `kitchen_23_Q04`: `ye` → `Yes`
2. `kitchen_02_Q01`: `letteuce` → `lettuce`
3. `kitchen_02_Q05`: `letteuce` → `lettuce`
4. `kitchen_23_Q03`: `soup opera` → `soap opera`
5. `kitchen_23_Q05`: `soup opera` → `soap opera`
