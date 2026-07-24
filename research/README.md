# Research Portal assets (Round 2 perceptual study)

The Research Portal (`/research`, source `research.html`) reads `stimuli.json` and serves videos from
these folders. Everything here is static, so it works locally (`npm run dev`) and on Vercel.

## Layout

```
research/
  stimuli.json          manifest: 48 stimuli (4 actions x 3 genres x 4 allocation levels) + metadata
  sources/              the 4 source action videos (clean names) + rendered original skeletons
  stimuli/              the 48 translated stimulus videos (+ their motion JSON)
```

## Generating the stimuli (on your Mac)

Put the 4 source videos in `~/Desktop/Videos/` (waving.mp4, shaking hands.mp4, walk_riku_cam1.mp4,
sorting.mp4), then:

```
cd pipeline
pip install mediapipe opencv-python matplotlib
brew install ffmpeg          # for mp4 output; without it, renders fall back to .gif
python make_stimuli.py --videos ~/Desktop/Videos
```

This copies the sources under clean names, extracts skeletons, grafts every genre x allocation level at a
constant allocation (0.20 / 0.40 / 0.60 / 0.80), renders each to mp4, and rewrites `stimuli.json`.

Then commit `research/sources`, `research/stimuli`, and `research/stimuli.json` so Vercel serves them.

## Data

Participant responses always autosave to the browser's localStorage and can be exported as CSV from the
Admin tab (raw = one row per participant x stimulus; summary = grouped by action x genre x allocation
level). Each stimulus collects 2 ratings: identity_preservation (how much it looks like the original
action) and preference (how much you like the movement).

## Central storage (Google Sheet) - optional

To collect all participants centrally (across devices), each response can also be POSTed to a Google Sheet.

1. Create a Google Sheet.
2. Extensions -> Apps Script. Replace the code with the script below and Save.
3. Deploy -> New deployment -> type "Web app" -> Execute as: Me -> Who has access: Anyone -> Deploy.
   Copy the Web app URL (ends with `/exec`).
4. Put that URL in `config.json` (`{"endpoint": "https://script.google.com/macros/s/.../exec"}`) and
   commit, so every device uses it. (Or paste it in the Admin tab for just this browser.)

Responses land in a `responses` tab (one row each, matching the raw CSV columns), participant background
in a `participants` tab, and the "Send test row" button writes to a `_ping` tab.

```javascript
function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var name = data.__sheet || "responses";
    delete data.__sheet;
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName(name) || ss.insertSheet(name);
    var keys = Object.keys(data);
    if (sh.getLastRow() === 0) sh.appendRow(keys);
    var header = sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0];
    keys.forEach(function (k) {
      if (header.indexOf(k) < 0) { header.push(k); sh.getRange(1, header.length).setValue(k); }
    });
    sh.appendRow(header.map(function (k) { return data[k] !== undefined ? data[k] : ""; }));
    return ContentService.createTextOutput(JSON.stringify({ ok: true }));
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ ok: false, error: String(err) }));
  }
}
```

There is no other server; without an endpoint set, nothing is uploaded.
