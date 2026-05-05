---
name: review-code
description: Use this skill when reviewing code changes for correctness, maintainability, performance, and consistency with repository conventions.
---

Follow this workflow when reviewing code or code changes:

1. **Understand the intent**: Read the code to be reviewed and nearby tests to understand the expected behavior.

2. **Check correctness**: Look for bugs, edge cases, broken assumptions, regressions, and incomplete handling of failure cases. If the code is implementing a mathematical algorithm, verify correctness and ensure a citation is available in the docstring.

3. **Check maintainability**: Assess whether the code is simple, readable, modular, and consistent with existing architecture and style. See the [architecture](../../../docs/architecture.md) and [style](../../../docs/style.md) guidelines for reference.

4. **Check performance**: Assess whether the code is efficient and identify any potential performance bottlenecks. If relevant, suggest running profiling to identify slow parts using Scalene.

5. **Check tests**: Verify that all important behavior is covered by relevant tests. Point out missing or weak tests when needed. See the [testing](../../../docs/testing.md) guidelines for reference on testing standards.

6. **Give actionable feedback**: Prioritize important issues. For each finding, explain the problem, impact, and suggested fix.

End with a brief summary of whether the change is ready, needs changes, or requires clarification.