# Submission checklist

This checklist separates package checks from decisions requiring the final
manuscript and submission form. It is not a guarantee of acceptance or
protection against desk rejection. Policies below were checked for ICLR 2027;
use the policy for the actual submission year.

## Source archive

- [ ] Extract the final ZIP in a fresh directory and run the README smoke.
- [ ] Verify installation, CLI help, stream construction, audit, training,
  reporting, and the included scientific tests from the extracted files.
- [ ] Record which optional dependencies, GPU tests, and real-data tests were
  actually exercised; do not report skipped paths as passed.
- [ ] Scan both filenames and contents for submitting-team names,
  affiliations, emails, personal paths, identifying repository links,
  credentials, machine names, notebook outputs, and embedded media metadata.
- [ ] Exclude Git history, caches, compiled files, local environment exports,
  datasets, generated streams, checkpoints, tracking logs, and internal notes.
- [ ] Keep the Apache license and legitimate third-party notices/citations.
- [ ] Match documented commands and configurations to the included files.
- [ ] Optionally compare the final ZIP checksum after upload/download to
  detect transfer corruption. This is not a training requirement.

## Manuscript and form: human confirmation required

- [ ] Keep the paper and supplement anonymous, including acknowledgments,
  links, PDF metadata, screenshots, and self-citations. Use the ICLR 2027
  template and keep initial main text within nine pages; references and
  appendices follow the conference rules. These are explicit submission
  requirements in the [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- [ ] Include the required AI-use section and complete the AI-use information
  in the submission form. Describe the actual assistance, including code or
  artifact preparation, and independently review its outputs. Requirements
  and task-specific disclosure categories are in the
  [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
- [ ] Check the abstract/full-paper deadlines, OpenReview profiles,
  authorship information, reciprocal-reviewing obligations, submission
  limits, and dual-submission rules in the
  [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- [ ] Ensure every table/figure is traceable to the stated configurations,
  seeds, selection criteria, compute budget, and eligible experiment cells.
  State uncertainty and failures; confirm all comparative claims against
  actual outputs rather than code availability.
- [ ] Describe preprocessing, global-derived features, train/validation/test
  separation, task order, ownership, LP negative sampling, and metric
  definitions. Disclose undefined NC-Domain metrics and ineligible LC-Domain
  cells wherever relevant; see [reproduction](reproducibility.md).
- [ ] Check dataset permissions, citations, privacy limitations, and other
  ethical considerations. Include reproducibility and ethics statements
  where appropriate under the
  [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

The package cleanup cannot verify a manuscript that is not included, recover
missing experimental evidence, or certify undisclosed changes to a protocol.
