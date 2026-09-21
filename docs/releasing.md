# Releasing `msms-controls`

The workflow in `.github/workflows/publish.yml` does the build and the upload.
Everything below that needs a password or an account is yours to do — the
workflow never sees a token, and nobody should paste one into this repository.

## One-time setup

### 1. PyPI account and Trusted Publisher

Create an account at <https://pypi.org/account/register/> and turn on two-factor
authentication when prompted.

Then register this repository as a trusted publisher, *before* the project
exists, at <https://pypi.org/manage/account/publishing/>. Fill the "pending
publisher" form with exactly:

| field | value |
|---|---|
| PyPI Project Name | `msms-controls` |
| Owner | `xulinpan` |
| Repository name | `cheminformatics` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

This is what replaces an API token. PyPI will trust an upload that comes from
this workflow in this repository and from nowhere else.

Repeat the same form at <https://test.pypi.org/manage/account/publishing/> with
environment name `testpypi` if you want the dry run below.

### 2. GitHub environments

In the repository, Settings → Environments, create two environments named `pypi`
and `testpypi`. They can be empty; the names are what the trusted-publisher
records above are matched against. Adding yourself as a required reviewer on
`pypi` means every upload waits for your approval, which is worth the extra
click.

### 3. Zenodo, for the DOI

Sign in at <https://zenodo.org> with your GitHub account, go to
<https://zenodo.org/account/settings/github/>, and switch this repository on.
Zenodo archives every *subsequent* release and mints a DOI for it, so do this
before cutting the release below or it will be missed.

`.zenodo.json` in the repository root supplies the title, authors, licence and
keywords, so the archived record does not need editing by hand.

## Dry run

Actions → publish → Run workflow → target `testpypi`. This builds, checks and
uploads to TestPyPI, where a bad release can be thrown away. Install it back to
confirm the artifact is sound:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ msms-controls
```

TestPyPI will not accept the same version twice, so bump the version in
`packages/msms-controls/pyproject.toml` before a second attempt.

## Releasing

1. Make sure `version` in `packages/msms-controls/pyproject.toml` is what you
   intend, and that `CHANGELOG.md` describes it.
2. Tag and push:

   ```bash
   git tag msms-controls-v0.1.0
   git push origin msms-controls-v0.1.0
   ```

   The prefix matters. The repository holds two distributions whose versions
   diverge, so the workflow ignores any release tag that does not name one, and
   it fails the build if the tag's version disagrees with `pyproject.toml`.
3. On GitHub, Releases → Draft a new release → choose that tag → publish.
4. The workflow builds, runs `twine check --strict`, and uploads to PyPI. Zenodo
   archives the same release and issues the DOI.
5. Put the DOI in `CITATION.cff` and in the manuscript's *Availability of data
   and materials*.

## Afterwards

`pip install msms-controls` should work from a clean environment within a minute
or two. Check that the README renders correctly on the project page — `twine
check --strict` catches most but not all rendering faults.

Releasing `dbf2` is deliberately not automated. It is the research code for one
paper rather than a tool, and it is meant to be installed from git.
