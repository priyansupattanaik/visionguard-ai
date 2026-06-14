# Human Developer Coding Style Protocol

Write all code as a real student or developer would — compact, natural, and without padding. Every rule below applies to every programming language unless explicitly asked for production-grade or enterprise style.

## 1. Variables & Naming
* **Keep it short and real:** Use abbreviations and standard shorthands (`X`, `y`, `df`, `sc`, `le`, `res`, `req`).
* **No descriptive bloat:** Never use long names like `features_dataframe`, `label_encoder_instance`, or `standard_scaler_object`.

## 2. Comments
* **Minimal and natural:** One short comment per logical block maximum.
* **Don't read the code aloud:** Never comment what the code already demonstrates (e.g., don't write `# Apply StandardScaler to normalize`).
* **Fragment phrasing:** Avoid full sentences and standard sentence case. Use `#fix missing` or `#feature scaling` instead of `# Fix the missing values`. 
* **No prose:** Never write `# This function...` or `# We now...`.

## 3. Formatting & Imports
* **Group imports:** Keep imports simple and grouped on one line where applicable.
* **No multi-line wraps:** Avoid wrapping imports with trailing commas unless strictly required by line-length limiters.
* **Standard aliases only:** `np`, `pd`, `plt`, `sns`.

## 4. Boilerplate & Type Hints
* **Zero docstrings:** No JSDoc, Python docstrings, or equivalent unless explicitly requested.
* **Zero type hints:** No Python type hints or TypeScript types unless specifically asked.
* **No execution wrappers:** Remove `if __name__ == '__main__':` unless writing a script intended for direct terminal execution.
* **No defensive padding:** Never wrap simple code in `try/except` or null-checks for no reason. 

## 5. Execution & Environment Style
* **Jupyter Notebooks:** * One idea/operation per cell.
    * Show, don't print. Put the variable on the last line instead of using `print(df)`.
* **DataFrames & Arrays:** Use raw indexing (`X[:, 1:3]`) and inline assignment (`df['col'] = ...`). Do not split natural operations into multiple intermediate named variables.

## 6. Language-Specific Examples
* **Python:** * Good: `X_train = sc.fit_transform(X_train)`
    * Bad: `X_scaled_training_data = scaler_object.fit_transform(X_training_data)`
* **JavaScript/React:** * Good: `function Card({ title }) { return <div className="card">{title}</div> }`
    * Bad: `const CardComponent: React.FC<Props> = ({ title }) => { ... }`
* **SQL:** * Good: `SELECT name, salary FROM employees WHERE dept = 'IT'`
    * Bad: `SELECT employee_name AS name ...`

## Strict Anti-Patterns (Never Do These)
1. Never write comments that repeat what the code says.
2. Never use long descriptive variable names when a short one works.
3. Never wrap simple code in try/except for no reason.
4. Never add docstrings or JSDoc unless asked.
5. Never add type hints unless asked.
6. Never write prose-style comments.
7. Never split one natural operation into multiple intermediate named variables.