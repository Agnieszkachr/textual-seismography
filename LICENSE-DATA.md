# Licence for data and results

The **code** in this repository is under the MIT Licence (see `LICENSE`).

Everything in `output/` — the per-verse model scores, the derived tables and the
generated dashboards — is released under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/

You may share and adapt this material for any purpose, including commercially,
provided you give appropriate credit. Suggested attribution:

> Ziemińska, Agnieszka Blanka (2026). *Textual Seismography: detecting redactional
> seams in the Book of Isaiah with dual-model transformer perplexity.* Data and
> results. https://github.com/Agnieszkachr/textual-seismography

## Third-party material

- **Hebrew text.** The Westminster Leningrad Codex, retrieved at runtime from the
  [OpenScriptures morphhb](https://github.com/openscriptures/morphhb) project and
  cached in `src/` (not redistributed here). Licensed CC BY 4.0 by that project.
- **English renderings** shown in the dashboard tooltips are taken from the
  **World English Bible**, which is in the public domain.
- **Plotly.js** v2.26.0 is bundled directly inside `docs/isaiah.html` so that the
  dashboard works without a network connection. Copyright 2012-2023 Plotly, Inc.,
  MIT licence; its copyright header is preserved in the file.
- **Models.** `Norod78/hebrew-gpt_neo-small` and `dicta-il/dictabert` are
  distributed by their respective authors under their own licences; this
  repository pins specific revisions but does not redistribute the weights.
