# Container validation — 3 October 2026

## Scope and evidence

The source-only, uncommitted packaging candidate is separate from the frozen
Apertus experiment at `1435b4b520e94465f4d86114a897f73eae9539da`. Its ZIP SHA256 is
`fe16eec66760e19ffbda2552b4d9fe909793afeee98f573bd4694a4b3164c16a`.
The executor verified all 48 archived file hashes before and after validation.
This documentation supplement was prepared afterward and is not part of that
48-file test artifact. No runtime source changes are introduced here.

The findings below are execution receipts reviewed from the existing builder
session. They are not an independent rerun of the private raw pilot arrays.
Eighteen synthetic packaging contract tests also passed independently in the
separate review environment.

## Exact Dockerfile build and runtime

The unchanged candidate Dockerfile built with exit status 0:

    docker build --pull=false --network=default --progress=plain -f Dockerfile -t splitalign:candidate-exact .

Build network use was limited to ordinary packages from official Debian
repositories and default PyPI. No provider credentials, build secrets or model
requests were used. The reported image/tag digest after the normal in-make rebuild was
`sha256:e7b91ea0721e6fa525e9a62e58682dc514db2f0564c104b4b39d7159324c5f01`.
This is a reported manifest-list digest, not an independently inspected
image-config digest.

The unchanged `make run` recipe executed its normal `docker-build` step using
`IMAGE=splitalign:candidate-exact`. A transparent wrapper forced build networking
for dependencies and `--pull never --network none` for runtime; it did not skip
the build or substitute the derivative image. Actual `APERTUS_*` and `LLM_*`
environment values were cleared. Only the mock backend and explicitly selected
read-only synthetic inputs were used, with fresh output:

    pipeline --backend mock --split val --lang all --limit 1 --offset 0 --max-requests 80 --max-tokens 400000 --bootstrap 0

Run `20261003T185338Z-541e827b` returned `MAKE_EXIT=0`. It emitted six prediction
files: three invented IDs across both methods, each with verified 4/4 token-label
vector lengths, plus run-scoped details, evaluations, summaries and viewer
evidence. The constant synthetic labels yield null correlation; no performance
claim follows. The source-only extraction has no Git commit identity, and its
manifest records this honestly.

## Draft-branch source relationship

This proposal retains the tested runtime changes, but is not byte-identical to
the 48-file archive. Fifteen existing test/viewer files retain the public
PR-head bytes rather than the archive's extra final blank line. Only trailing
LF bytes differ in those files. The two new experimental report drafts are
omitted; the README and packaging notes are narrower, later documentation.
Eighteen synthetic contract tests were rerun successfully on this proposed
tree. Docker evidence above remains attributed to the exact archive, not to a
fresh Docker test of this branch.

## Earlier stages retained

The first exact build with networking disabled failed at `apt-get install
nodejs` with exit 100 because dependencies were unavailable offline. A separately
labeled derivative image then passed configuration checks and the mock pipeline
`20261003T181652Z-1a78fe37`. That derivative result must not be relabeled as the
later exact Dockerfile build. Eighteen contract tests passed; actual-container
fake-configuration checks on the derivative rejected missing bounds and
conflicting aliases before dispatch. Those particular configuration cases were
not repeated against the final exact image.

## Remaining gates

- The organizer's plain clean-checkout `make run` contract is not certified:
  this candidate deliberately requires explicit input selection and bounded
  arguments. Final input/output and coverage requirements must be resolved
  before choosing safe defaults or implementing a separate final runner.
- The inspected official guide/form require model predictions but do not
  specify an exact submission split, minimum coverage or prediction schema.
  Three reused development records are not certified sufficient.
- Final private prediction arrays have not been independently replayed.
  Frozen-pilot scores and detailed output/cost audit remain executor receipts.
- No full corpus-dependent test suite, real-provider end-to-end run,
  clean Git-checkout release, publication or organizer submission is claimed.
- Original negative results, failed starts, cache-planning incident and frozen
  artifacts remain unchanged. No further inference or new gold/test data was
  authorized by this validation.

Sources: [submission guide](https://hackapertus.notion.site/how-to-submit-a-project),
[UZH challenge](https://hackapertus.notion.site/track-2a-uzh),
[submission portal](https://hackapertus.ch/online-hack/submissions),
[organizer template](https://github.com/HackApertus/project-template).
