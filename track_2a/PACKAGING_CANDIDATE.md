# Packaging candidate

Draft reproducibility-only changes based on public PR head
`23237d96734dfefdf2280cf4ba6f4b53497126d4`; the frozen experiment source is
`1435b4b520e94465f4d86114a897f73eae9539da`. This candidate did not produce
the frozen pilot. Existing reports, runs, predictions, cache-incident records
and frozen experimental source remain preserved.

## What changed

- The Docker Make recipe forwards official `LLM_NAME`, `LLM_BASE_URL`,
  `LLM_API_KEY`, and the required `APERTUS_BOUND_SPEC_JSON` by variable name.
  No credential values are written into commands or source.
- `ApertusClient.from_env` accepts the official names and legacy aliases.
  When Apertus is selected, conflicting values fail with variable names only.
  Official configuration
  selects the Apertus backend in the packaging launcher, unless an explicit
  backend was selected. Partial Apertus configuration fails instead of silently
  mocking; an explicit mock selection skips Apertus configuration validation.
- Docker's default is `check-config`. It validates local configuration only,
  without importing the data pipeline or dispatching a model call. This is
  available through **`make preflight`**, not `make run`. A plain `make run`
  fails before Docker execution; explicitly selected pipeline arguments and
  data cause it to invoke the existing complete pipeline once. No organizer
  compliance claim follows from the mocked contract.
- Docker's build context excludes all data, prior results, caches, credentials
  and generated viewer evidence. A requested run needs an absolute selected
  `SPLITALIGN_DATA_DIR` mounted read-only at `/app/track_2a/data`.
- For `make run`, `RUN_ARGS` must explicitly begin with `pipeline` and select
  a split, language (`de`, `fr`, `it` or explicit `all`),
  positive limit, nonnegative offset, positive request cap and positive
  conditional token cap. Explicit `all` keeps a selected multilingual pipeline
  in one invocation with its existing shared budget. There is no implicit
  language or unlimited run in this launcher. The original
  development CLI remains available; this is an accidental-use guard at the
  packaging boundary, not an access-control sandbox.
- The README documents the packaging boundary and links to the existing
  historical report. Final report-path/content changes are deferred.

## Safe check performed

`make -C track_2a test-contract` runs standard-library tests. Credentials and
the bound in those tests are synthetic and labeled test-only. Network entry
points and model dispatch are blocked. A temporary Docker executable captures
Make's argv and fake environment; that forwarded configuration is then passed
to the real configuration loader without dispatch. A stub replaces the
pipeline for explicit-selection argument forwarding. No input records are
fabricated or bundled.

Configuration can be checked with `make preflight`; Docker must be installed and the
image dependencies available. With no model configuration this reports mock
configuration only. With official LLM variables it also requires a separately
reviewed bound specification. Success does not authenticate the key, contact
the provider, validate the bound's conservatism, run inference or score data.

## Remaining gates

- The initial cloud review had no Docker. On 3 October 2026, the existing
  execution environment successfully built the exact candidate Dockerfile and
  ran the unchanged make recipe against explicitly selected synthetic fixtures
  with the mock backend and container networking disabled. See
  [container validation](docs/CONTAINER_VALIDATION.md). This is not a clean Git
  checkout, a frozen-pilot replay, a real-provider test or a full-suite pass.
  Dataset-dependent tests still require approved inputs and separate validation.
- The organizer template documents `make run` and the three LLM variables,
  but the reviewed material did not publish an exact final input/output CLI or
  schema. This patch does not invent one. End-to-end organizer compatibility
  remains unverified.
- The selected directory must use the existing development layout:
  `gold/dev/{train,val}/gold_admin_{de,fr,it}.jsonl` as applicable, plus the
  existing `manifest/dev_ids.json`. These are descriptions of the current
  development loader, not an official final-evaluation schema.
- The loader reads a whole selected language file before applying limit and
  offset. Selecting a range is not proof that only the frozen pilot was read.
  Exact pilot reproduction requires separately verified isolated input files
  and an initially empty output/cache directory. No such files are supplied
  by this packaging candidate; no fresh data or pilot rerun is authorized.
- Final prediction-array replay, final artifact publication, submission coverage,
  full test execution, final packaging acceptance and submission remain open.
- Final report content, required report path and rendered two-page limit are
  outside this reproducibility-only change.

## Sources

- [Frozen project source](https://github.com/sharonbasovich/splitalign/tree/1435b4b520e94465f4d86114a897f73eae9539da)
- [Organizer project template](https://github.com/HackApertus/project-template/blob/main/README.md)
