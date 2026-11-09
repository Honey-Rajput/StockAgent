<!-- Written by Euron Code after each task. It is rewritten automatically: put your own rules in
     another file in .euron/rules/. -->

# Learned in this project


## Security (from .euron/security/assessment.md)
- This code base has had: sql built from a string (SEC020). Use parameterised queries (placeholders + parameters), never f-strings, % or + in SQL. Do not add more; fix the existing ones when you touch those files.
