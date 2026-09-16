---
name: write-model-docs
description: Write or review reproducible documentation of numerical methodologies implemented in Python.
---

## Write new document workflow

When writing a new methodology document, follow this workflow:

1. Inspect relevant code, tests, and existing model documentation. 
2. If the code implements a specific peer-reviewed methodology, collect the relevant articles*.
3. Ask the user to confirm the identified source and request any additional clarification before proceeding.
4. Document the implemented methodology following the template structure and guidelines.
5. Ask a new agent to review the written document (see Review existing document workflow) and fix the gaps.
6. Save the document in `./docs/methodology/<method>.md`.

\* Some articles are available in `./docs/articles/`.

## Review existing document workflow

When reviewing an existing methodology document, follow this workflow:

1. Verify that the document follows the template structure and guidelines.
2. Verify if the information in the document is sufficient to reproduce the methodology.
3. Detect discrepancies between the documented methodology and what is implemented in the code.
4. Verify all limitations and deviations from the cited peer-reviewed methods are clearly stated.
5. Report the gaps without editing the document.

## Model Documentation Guidelines

Documentation of a numerical methodology must follow the template and adhere to the following guidelines.

- Keep the document self-contained and as concise as possible.
- Document what is implemented in the code, not what is described in the docstrings/comments.
- Document the methodology in a way that allows the reader to reproduce it.
- Use the same notation as the cited sources where practical, and be consistent throughout the document.
- Describe how model correctness is verified in the unit/integration tests.
- List relevant scripts and notebooks from `./examples/` and `./scripts/` that demonstrate the methodology.
- Use Markdown and LaTeX for equations, using the `$...$` and `$$...$$` delimiters.
- Use a soft limit of 120 characters per line for readability.

### Template
```
# `<Method>`

## Purpose and Scope
[Short summary and table listing the main public API functions (with hyperlinks)]

   | Main public API | Module | Responsibility |
   |---|---|---|

## Methodology

### Theoretical Framework
[Describe the mathematical model, its assumptions, and limitations]

### Numerical Procedures
[Describe the numerical algorithms, including the inputs, outputs, ordered steps, hyperparameters, and defaults.]

## Validation ([% of test coverage])
[Describe how the numerical correctness of the methodology is validated]

## Scripts and Examples
[List relevant tasks, examples, and notebooks]

## References
[Only references that are cited. Include DOI or URL if available]

```