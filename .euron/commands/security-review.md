---
description: Review and fix the open security findings
---
Read `.euron/security/assessment.md`. For each open finding, worst first: open the file, decide whether
it is a real risk, and either fix it (smallest safe change, then run the tests) or explain why
it is safe and mark the line with a `euron: ignore-security` comment. Do not touch unrelated code.
Finish with a short list: fixed, accepted as safe, still open. $ARGUMENTS
