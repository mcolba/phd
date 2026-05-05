---
name: implement-feature
description: Use this skill when implementing a feature from a specification or issue.
---

Follow this workflow when implementing new features:

**1. Understand the requirements**

  - Carefully read the request.
  - Critically assess methodological and architectural implications.

**2. Ask for clarification (optional)**
  
  - Flag potential issues or ambiguities in the request.
  - If something is not clear, ask the user for clarifications.
  
**3. Plan the implementation**

  - Create a concise implementation plan before editing.
  - Follow the repository's [architecture](../../../docs/architecture.md) guidelines.
  - Avoid code duplication by reusing existing functions and abstractions.
  
**4. Implement the feature**:

  - Make the smallest coherent code change that satisfies the request.
  - Do not refactor unrelated code unless it was part of the request.
  - Follow the repository's [style](../../../docs/style.md) guidelines.
  - Keep naming, error handling, logging, and API shape consistent with nearby code.

**5. Validate the implementation**: 

  - Run relevant tests, linters, type checks.
  - For complex changes, create a concise, temporary test script to validate the implementation.
  

After completion, ask the user if they want to implement a unit test. If they do, delegate the task to the `implement-unit-test` skill.