Building the ibleatools documentation
=====================================

The site is a Quarto website. The API reference under docs/api/ is GENERATED from the
package docstrings by quartodoc and is not checked in -- everything else in docs/ is
hand-written.

Prerequisites
-------------

1. Quarto (a standalone binary, not a Python package):
     https://quarto.org/docs/get-started/
   Check with:
     quarto --version

2. The docs dependency group (quartodoc + griffe), plus the package itself:
     uv sync --group docs


Local development
-----------------

1. Regenerate the API reference from the docstrings (run after editing any docstring
   listed in the `quartodoc` section of docs/_quarto.yml):
     uv run quartodoc build --config docs/_quarto.yml

2. Live-preview the site:
     quarto preview docs/

   The preview server reloads automatically when .qmd files change.
   Re-run step 1 manually whenever docstrings change -- quartodoc output is not
   watched by the preview server.

3. One-off full render:
     quarto render docs/


Adding a page
-------------

Pages are organised on the Diataxis framework, and the directory says which kind a page
is. Put a new page in the one that matches what the reader needs:

  tutorials/     learning-oriented -- one guided journey, start to finish
  how-to/        task-oriented -- "how do I <specific thing>"
  reference/     information-oriented -- specs and lookup tables
  explanation/   understanding-oriented -- concepts, rationale, trade-offs

Then add it to the navbar in docs/_quarto.yml, and to the documentation table in
docs/index.qmd. A page reachable from neither is effectively invisible.

Adding an API symbol: add it to the relevant `quartodoc.sections` entry in
docs/_quarto.yml and re-run `quartodoc build`. Do not hand-write API pages.


Code blocks and figures
-----------------------

Code blocks are STATIC (```python), not executed at render time. Nothing in this site
requires credentials, raw data or a GPU to build.

Figures are therefore pre-rendered and committed under docs/figures/. The code that
produced each one is shown on its page, so a reader can reproduce it. When you change
such code, regenerate the figure and commit it in the same change -- otherwise the page
and the image disagree and nothing will tell you.

The figures on how-to/plot-features-along-probe.qmd are generated from the feature
tables committed at tests/fixtures/features/2025_W28/, so they need no credentials.


Production build
----------------

Built and deployed by GitHub Actions on every push to main
(.github/workflows/documentation.yaml), which runs the same steps as above:

    quartodoc build  ->  quarto render  ->  deploy to GitHub Pages

Pull requests build the site as a check but do not deploy.

The published site is https://int-brain-lab.github.io/ibleatools/

Do not commit docs/_site/ or docs/api/; both are generated (and git-ignored).
