# Demo assets

**Historical recordings — not the corrected demo.** Both videos were recorded
on the pre-repair viewer (main @ `ad1446e`) showing run
`20261001T211451Z-8de1aab7`. That run used prompt version v1, whose judge
prompt named German as the target language for the fr and it documents too
(see `report/technical_report.md` §5.0). The recordings therefore show real
Apertus output of a defect-affected configuration, without the run-status,
known-defect and exporter-vs-inference provenance lines the viewer now
displays. They are kept unchanged as evidence of what was shown at the time;
no corrected multilingual run or demo exists yet.

- `splitalign-demo-83s.mp4` — 83-second walkthrough of the evidence viewer on
  real Apertus dev/val output (run 20261001T211451Z-8de1aab7, prompt v1): doc
  list, aligned spans, per-token heat maps, omissions/additions, provenance
  banner, keyboard nav.
  Playable URL: https://github.com/sharonbasovich/splitalign/blob/main/track_2a/demo/splitalign-demo-83s.mp4
- `fix-verification.mp4` — mode-selector-sync fixture test + 390px
  no-overflow check (pre-repair viewer code).

The static viewer itself is in `track_2a/viewer/` (serve with any static file
server, e.g. `python3 -m http.server -d track_2a/viewer`; evidence.js requires
HTTP, not file://). Its current `evidence.js` is a re-export of the same run by
the repaired exporter and shows the defect notice.
