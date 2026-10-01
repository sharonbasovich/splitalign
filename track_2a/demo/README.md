# Demo assets

- `splitalign-demo-83s.mp4` — 83-second walkthrough of the evidence viewer on
  real Apertus dev/val output (run 20261001T211451Z-8de1aab7): doc list, aligned
  spans, per-token heat maps, omissions/additions, provenance banner, keyboard nav.
  Playable URL: https://github.com/sharonbasovich/splitalign/blob/main/track_2a/demo/splitalign-demo-83s.mp4
- `fix-verification.mp4` — mode-selector-sync fixture test + 390px
  no-overflow check.

The static viewer itself is in `track_2a/viewer/` (serve with any static file
server, e.g. `python3 -m http.server -d track_2a/viewer`; evidence.js requires
HTTP, not file://).
