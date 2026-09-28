"""feedback demo web app.

This package owns the stakeholder-facing demo: a Konva draw canvas on the left,
a result panel on the right, and on-canvas popups that guide a child to where
their drawn arithmetic is right, missing, or wrong. The recognizer reads the
drawing as drawn (it never corrects the maths) and the grading engine (the
high-fidelity in-process mock) grades
it; the demo only surfaces both.

The app is additive: a new FastAPI app (:mod:`.app`) plus a single in-memory
session (:mod:`.session`), wired to the ``demo`` CLI subcommand. It depends on
the :mod:`src.grading` package for the engine client, contract, and pixel<->
grid mapping, and on :mod:`src.inference` for recognition; it never imports the
set-maker (it copies the proven patterns) and contains no grading or mapping
logic inline.
"""
