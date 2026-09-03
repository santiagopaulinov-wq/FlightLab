# v1.0 release audit

## Verdict

The implemented FlightLab scope is technically release-capable: public module
boundaries agree with the README, architecture, verification documentation, and
representative example; locked tests and static checks pass; source and wheel
artifacts build; and the wheel installs and runs the representative workflow in
a fresh runtime-only environment.

The repository is **not yet ready to tag or publish as v1.0.0**. The remaining
release blockers are:

1. no owner-selected license file or package license metadata;
2. package metadata still intentionally reports the development version
   `0.1.0`;
3. the release-candidate changes are not committed and the worktree is not
   clean; and
4. no `v1.0.0` tag exists.

License selection cannot be inferred from source code or history and requires
the repository owner's direction. Version change, final commit, and tagging
belong to the subsequent explicit release-preparation step. Publishing and
pushing require separate authorization.

## Audited boundaries

- Public objects are imported from their owning modules; the empty package
  root promises no convenience re-exports.
- Python metadata requires `>=3.12`; the audit used CPython 3.12.3.
- NumPy is the sole wheel runtime requirement. Pytest, Ruff, and SciPy remain
  locked development dependencies; the wheel-only environment contained no
  SciPy installation.
- `uv.lock` agrees with `pyproject.toml` and resolves ten packages for the
  development environment.
- The README describes the implemented aircraft, control, simulation,
  experiment, persistence, campaign, analysis, and V&V boundaries.
- The representative example uses only existing public FlightLab APIs and is
  covered by a focused end-to-end integration test.

## Artifact results

The audit command was:

```console
uv build --no-sources --out-dir /tmp/flightlab-release-audit/dist
```

Both `flightlab-0.1.0.tar.gz` and
`flightlab-0.1.0-py3-none-any.whl` built successfully. The wheel is a pure
Python `py3-none-any` artifact containing only the `flightlab` modules and
`.dist-info` metadata; it contains no tests, caches, local databases, editor
settings, examples, or development dependencies. There is no runtime package
data to declare.

The source distribution contains package source, `pyproject.toml`, README,
lock file, Python version file, changelog, roadmap/checkpoint documents,
architecture/reproducibility/verification/audit documents, the representative
example, and the complete test suite. Cache directories, bytecode, editor
settings, virtual environments, databases, and build output are excluded.

## Installed-wheel check

A new CPython 3.12.3 virtual environment was created and the built wheel was
installed directly. The installed metadata reported:

```text
name: flightlab
version: 0.1.0
requires-python: >=3.12
runtime requirement: numpy>=2.5.2
installed numpy: 2.5.2
installed scipy: absent
import location: the fresh environment's site-packages directory
```

Using that installed artifact, `examples/longitudinal_campaign.py` completed
and created a fresh SQLite database containing campaign
`longitudinal-pitch-pole-scale-v1` and its three ordered experiment runs.

## Verification results

```text
focused end-to-end workflow tests: 4 passed in 2.34s
focused V&V tests: 188 passed in 5.59s
full suite: 1631 passed in 16.01s
Ruff: passed
git diff --check: passed
uv lock --check: passed
```

The final counts must be reconfirmed after the license and version are set, but
those metadata-only release-preparation changes should not alter scientific
behavior.

## Hygiene and release notes

`.gitignore` covers virtual environments, Python bytecode, pytest/Ruff caches,
editor settings, build directories, egg-info, and generated SQLite databases.
The existing `.vscode/settings.json` was not modified. Build/audit artifacts
were written under `/tmp`, not into the repository.

`CHANGELOG.md` now contains factual Unreleased v1.0 candidate notes derived
from implemented code and Git history. At release preparation, rename that
section to `1.0.0` with the release date after the final version is approved.

## Remaining checklist before v1.0.0

1. Obtain the owner's license choice; add the corresponding license file and
   PEP 639 `license`/`license-files` package metadata.
2. Confirm the intended public distribution name `flightlab` is available and
   is the name the owner intends to publish.
3. Review the candidate changelog wording and choose the v1.0.0 release date.
4. Set package version `1.0.0` and update the changelog heading/date.
5. Run `uv lock --check` or update the lock only if the version change requires
   it.
6. Rebuild wheel and sdist with `uv build --no-sources`; re-inspect metadata and
   contents, including the selected license.
7. Install the final wheel into a new runtime-only environment; verify metadata,
   import, and the representative workflow.
8. Run focused workflow/V&V tests, the full suite, Ruff, and
   `git diff --check`.
9. Commit all intended release files and confirm a clean worktree.
10. Create the `v1.0.0` tag only after explicit authorization. Publish and push
    only when separately authorized.

## Release-preparation resolution

The blockers identified by this audit were subsequently resolved on
2026-09-03 during explicit v1.0.0 release preparation:

- the repository owner selected the MIT License;
- `LICENSE` and PEP 639 MIT package metadata were added;
- the public distribution name remains `flightlab`;
- the package version and lockfile were updated to `1.0.0`;
- the changelog heading was finalized as `1.0.0 — 2026-09-03`;
- the final wheel and source distribution were rebuilt and inspected;
- the final wheel installed successfully into a fresh CPython 3.12.3
  environment with NumPy as its sole runtime dependency and without SciPy;
- the representative installed-wheel campaign completed and persisted all
  three expected runs;
- 4 focused workflow tests, 188 focused V&V tests, and all 1631 tests passed;
- Ruff, `uv lock --check`, and `git diff --check` passed.

The remaining repository action is the release commit and confirmation of a
clean tracked worktree. Creation of the `v1.0.0` tag, pushing, and publication
remain subject to explicit owner authorization.

The earlier `0.1.0` findings in this document are intentionally retained as
the historical evidence that motivated these release-preparation actions.
