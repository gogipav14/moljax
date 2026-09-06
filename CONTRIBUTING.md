# Contributing to moljax

Contributions are welcome. This document covers licensing, the branch and
review model, and what CI expects.

## Licensing of contributions

moljax is MIT licensed. **By opening a pull request you confirm that you are
able to license your contribution under the MIT license**, either because you
own the copyright or because you have permission from whoever does.

This matters more than it usually would. Contributions to this project often
come out of funded research, and employers, universities and grant agencies
frequently retain rights to work produced under their funding. If your
contribution was written as part of funded or employed research, please check
with the body that funded it *before* opening the pull request. Discovering an
ownership problem after code has shipped in a tagged release is far worse for
you than a short delay now.

If your parent project is under a copyleft license such as GPL, note that the
direction matters: MIT code can be used inside a GPL project, but GPL-derived
code cannot be relicensed into this MIT one without permission from the rights
holder.

If you are unsure, say so in the pull request and we will work it out before
merging.

## Branches and review

- `main` is protected. It requires a pull request, a review, and passing CI.
  Force pushes and branch deletion are disabled.
- Release tags (`v*`) are protected and cannot be moved or deleted. `v1.0.0`
  in particular is the archival commit cited by the published paper and must
  never change.
- Long-lived topic branches are fine, and preferred over a pull request per
  increment. Stack your work on a branch and open one pull request when the
  increment is coherent.

## What CI runs

Every pull request runs the test suite on Python 3.10 and 3.12, CPU-only,
via `JAX_PLATFORMS=cpu`, and `ruff check .` with the pinned Ruff (0.16.5,
also in the `dev` extra). The test jobs are required checks; the lint job is
being promoted to one as well, now that the style backlog that predated CI
is cleared, so treat a lint finding in your change as blocking.

```bash
pip install -e ".[dev,viz]"
pytest
ruff check .
```

`E402` (module-level import not at top of file) is waived for `tests/`,
`benchmarks/` and `examples/` only. Those scripts run
`jax.config.update("jax_enable_x64", True)` before any array is created, and
the imports that follow that call trip the rule. The package itself has no
such sites and carries the full rule set.

## Tests

- New behavior needs a test. Numerical claims need a test against an
  independent reference (an analytic solution, a dense computation, or a
  well-established library), not just a regression snapshot.
- Mark anything that takes more than about a minute with `@pytest.mark.slow`.
  Nothing is deselected by default: `pytest` runs all 450 tests, about six
  minutes on CPU, and that is what CI runs. The marker is for skipping them
  locally with `pytest -m "not slow"`.
- Benchmarks live in `benchmarks/` and are not part of the test suite.

## Benchmarks and comparisons

If you add a benchmark that compares moljax against another library, the
filename, the docstring and any committed results must make the comparison
unambiguous, including what is held fixed and what is not. A comparison
between methods of different order, or between fixed-step and adaptive
stepping, is easy to misread out of context, and committed result files
outlive the discussion that produced them.

Keep committed result data proportionate. Large generated artifacts are better
regenerated from a script than stored in git history.

## Reporting problems

Open an issue with a minimal reproduction, the output of

```bash
python -c "import jax, moljax; print(jax.__version__, moljax.__version__, jax.devices())"
```

and what you expected instead.
