---
name: implement-unit-test
description: Use this skill when implementing or updating a unit test for a feature or function.
---

Follow this workflow when implementing or updating unit tests:

1. Activete planning mode if available.

2. Read the [docs/testing-framework.md](../../../docs/testing-framework.md) document to ensure the test implementation matches this repository's guidelines.

3. Identify the exact behaviors to be tested and plan the test scope to be as narrow as possible. 

4. Consider generating or reusing an existing validation dataset if the behavior is complex or prone to numerical errors.

5. Summarising the testing plan and ask the user for confirmation before proceeding with the implementation. If there are any uncertainties about the expected behavior or the test design, ask the user for clarifications.

7. Implement the test with focused assertions. Use `numpy.testing` and explicit tolerances for numerical checks when applicable.

8. Run the narrowest relevant test selection and fix any failures before finishing. If a test fails always assume it might be a problem with the implementation rather than the test and investigate accordingly.

Aim for tests that are small, deterministic, and directly tied to the behavior change.