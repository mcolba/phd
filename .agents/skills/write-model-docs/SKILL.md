---
name: write-model-docs
description: Write or review reproducible documentation of numerical methodologies implemented in Python.
---

## Write new document

When writing a new methodology document, follow this workflow:

1. Inspect code, tests, and existing model documentation. 
2. If the code implements a specific peer-reviewed methodology, identify its source.
3. Before writing the document, ask the user to confirm the identified source and request any additional clarification.
4. Document the implemented methodology following the template structure and guidelines.
5. Compare the implemented methodology with the cited sources and clearly state any deviation.
6. Save the document in `./docs/methodology/<method>.md`.

## Review existing document

When reviewing an existing methodology document, follow this workflow:

1. Verify the document follows the template structure and guidelines.
2. Verify the documented methodology matches what is implemented in the code (do not rely on docstrings or comments).
3. Verify all limitations and deviations from the cited peer-reviewed methods are clearly stated.
4. Report the gaps without editing the document.

## Model Documentation Template

Documentation of a numerical methodology must follow this template and adhere to the following guidelines.

- Keep the document self-contained and as concise as possible.
- Document what is implemented in the code, not what is described in the docstrings or comments.
- Document the methodology in a way that allows a reader without repository knowledge to reproduce it.
- Use the same notation as the cited sources where practical and be consistent throughout the document.
- Use Markdown and LaTeX for equations, using the `$...$` and `$$...$$` delimiters.
- Use a soft limit of 120 characters per line for readability.

```
# `<Method>`

## Purpose and Scope
[Short summary and table listing the main public API functions (with hyperlinks)]

   | Main public API | Module | Responsibility |
   |---|---|---|

## Methodology
[Define assumptions, notation, equations, and deviations from cited methods]

## Algorithms
[Describe inputs, outputs, ordered steps, parameters and defaults]

## Examples and Validation
[List relevant tasks, examples, and notebooks, and provide a short overview of the testing strategy and coverage]

## References
[Only references that are cited. Include DOI or URL if available]

```