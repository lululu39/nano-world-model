# Repository language and previews

- Keep all repository content in English. This includes documentation, comments,
  notebook markdown and outputs, examples, UI labels, filenames, and generated
  artifacts. Do not introduce Chinese text.
- Keep pull request titles and descriptions in English.
- A reviewer preview must use the same English notebook source and content that
  readers receive. Execution outputs may be included; do not create a separate
  translated or customized reviewer version.
- Apply the same language rule to preview pages and exported review artifacts,
  even when they are generated outside the Git checkout.
- Conversation with the maintainer may use their preferred language; that does
  not change the language of the repository or its previews.
- Run `python3 scripts/check_repository_language.py` before submitting changes.
  The CI check rejects Chinese/Han text in tracked text files and filenames,
  including JSON-escaped notebook text. Review rendered media for English labels
  as well, since text embedded in images or videos is not covered by that check.

# Experiment tracking

- Run smoke/debug checks and authorized formal training in the same HTCondor
  job and GPU allocation. On success, start formal training immediately without
  releasing the GPU and submitting another training job. On failure, stop
  training and retain a bounded debug session in that allocation. Keep
  diagnostic weights, RNG, data position and logs separate; initialize formal
  training from its configured starting state. Diagnostics create no W&B run.
- Future formal NanoWM research runs use W&B project `nanowm`, entity
  `yibozhong657-none`. Override legacy launcher project names when necessary.
- Resume an existing run in its original project and with the same run ID.
  Diagnostic jobs and independent evaluation do not create W&B runs.
