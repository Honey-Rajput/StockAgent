- Never print, log or commit secrets (.env values, API keys, tokens, passwords).
- Use parameterised queries for SQL; never build SQL from user input with string formatting.
- Validate and limit external input; never `eval` or `exec` it.
- Do not disable TLS verification, authentication or CSRF checks to make something work.
- Open findings are tracked in `.euron/security/assessment.md` (rescanned after every task): fix the ones in
  files you change, and never add a new one.
