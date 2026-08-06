---
name: review-python-code
description: Use this skill when reviewing code changes for correctness, maintainability, performance, and consistency with repository conventions.
---

Follow this workflow when reviewing code or code changes:

1. **Understand the intent**: Read the code to be reviewed and nearby tests to understand the expected behavior.

2. **Check correctness**: Look for bugs, edge cases, broken assumptions, and regressions. If the code is implementing a mathematical algorithm, independently verify the mathematical steps and the correctness of the algorithm.

3. **Check repository requirements**: Assess whether the code is consistent with the repository [architecture](../../../docs/architecture.md) and [python-style](../../../docs/python-style.md) guidelines.

4. **Check performance**: Assess whether the code is efficient and identify any potential performance bottlenecks. If relevant, run profiling to measure the performance impact.

5. **Check tests**: Verify that all important behavior is covered by relevant tests. Point out missing or weak tests. See the [testing-framework](../../../docs/testing-framework.md) guidelines for reference on testing standards.

6. **Produce actionable feedback**: Prioritize important issues. For each finding, explain the problem, its impact, and the suggested fix.

End with a brief summary of whether the change is ready, needs changes, or requires clarification.