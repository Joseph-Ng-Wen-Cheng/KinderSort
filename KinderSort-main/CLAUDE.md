# CLAUDE.md — Kindergarten Student Photo Sorter

## Project Identity
- **Project Name:** KinderSort — Student Photo Organiser
- **Purpose:** Automatically sort kindergarten event photos into individual student folders using offline face recognition
- **Target User:** Non-technical kindergarten teacher (Windows PC, no coding knowledge)
- **Final Deliverable:** A single `.exe` file the teacher can double-click with no installation required

---

## Developer Context
- Developer is technical (Python, full-stack background)
- Development machine: Windows or Ubuntu, limited RAM, **no GPU** — all code must run on CPU only
- Claude Code is used for development and testing
- Teacher's machine: Windows, no Python, no terminal access

---

## Tech Stack
| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Cross-platform, rich libraries |
| Face Recognition | `face_recognition` (dlib) | Free, offline, CPU-capable |
| Image Handling | `Pillow` | Lightweight, reliable |
| GUI | `tkinter` | Built-in Python, no extra install |
| Packaging | `PyInstaller` | Produces single `.exe` |

---

## Core Functional Requirements

### Input
1. **Reference Folder** — contains one clear photo per student, named by student name
   - Example: `Ali.jpg`, `Siti.png`, `Kumar.jpeg`
2. **Events Folder** — contains subfolders of event photos (mixed, unsorted)
   - Example: `Sports_Day/`, `Concert/`, `Field_Trip/`
3. **Output Folder** — where sorted results will be written

### Processing Logic
- Load and encode all reference faces on startup
- Walk through every image in every event subfolder
- Detect all faces in each photo
- Match each detected face against all student encodings
- If match found → **copy** (not move) the photo into `Output/StudentName/`
- One photo can be copied to **multiple student folders** (group shots)
- Unmatched photos → copy to `Output/_unmatched/` folder
- Skip non-image files silently

### Output Structure
```
/Output/
    Ali/
        Sports_Day_IMG_001.jpg
        Concert_IMG_045.jpg
    Siti/
        Sports_Day_IMG_001.jpg   ← same photo, also has Siti
        Field_Trip_IMG_023.jpg
    _unmatched/
        blurry_photo.jpg
        no_face_detected.jpg
```

---

## GUI Requirements (Tkinter)
- Clean, simple window — single screen, no tabs
- Three folder selector rows (Reference / Events / Output) with Browse buttons
- Large **Start Sorting** button
- Progress bar (determinate, shows % complete)
- Status label showing current file being processed
- Summary box at end: total photos processed, matched, unmatched
- Error messages shown in dialog boxes (not terminal)
- Window title: `KinderSort — Student Photo Organiser`
- Minimum window size: 500x400

---

## Performance & Constraints
- Must work with **no GPU** — use CPU only (dlib default)
- Target: process 500 photos in under 15 minutes on low-spec machine
- Tolerance: `face_recognition` distance threshold = `0.5` (slightly strict to avoid false matches)
- If a photo has no detectable face, skip gracefully and log to `_unmatched/`
- RAM-friendly: process one image at a time, do not batch-load all into memory

---

## File Naming Convention
- When copying, **prefix filename with event folder name** to avoid collisions
  - `Sports_Day__IMG_001.jpg` (double underscore as separator)
- If same filename already exists in output folder, append `_2`, `_3` etc.

---

## Error Handling Rules
- Reference photo with no detectable face → show warning dialog, skip that student
- Corrupted/unreadable image → log to `_unmatched/`, continue processing
- Output folder not writable → show error dialog, stop gracefully
- All errors must be caught — **never crash silently**

---

## Packaging Instructions (PyInstaller)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "KinderSort" main.py
```
- `--onefile` → single `.exe`
- `--windowed` → no terminal window popup
- Output in `/dist/KinderSort.exe`
- Include any required dlib model files using `--add-data` if needed

---

## File Structure
```
kindersort/
├── CLAUDE.md          ← this file
├── main.py            ← entry point, GUI
├── sorter.py          ← face recognition logic
├── utils.py           ← file helpers, naming, logging
├── requirements.txt   ← all dependencies pinned
└── dist/
    └── KinderSort.exe ← final deliverable
```

---

## Coding Standards
- All functions must have docstrings
- Use `pathlib.Path` not `os.path` for file operations
- Use `shutil.copy2` to copy files (preserves metadata)
- Log all actions to `kindersort_log.txt` in the output folder
- No hardcoded paths anywhere
- Code must be readable — teacher may show it to another developer later

---

## Out of Scope
- No cloud upload, no internet required, fully offline
- No database, no login, no user accounts
- No video processing — images only (jpg, jpeg, png, bmp, webp)
- No facial emotion or age detection
- No mobile app version

---

## Definition of Done
- [ ] GUI launches with double-click on `.exe`
- [ ] Correctly sorts test set of 50 photos with 10 reference students
- [ ] Group photos copied to all matching student folders
- [ ] Unmatched photos go to `_unmatched/`
- [ ] Progress bar updates during processing
- [ ] Summary shown after completion
- [ ] No crash on corrupted or faceless images
- [ ] Log file written to output folder
- [ ] `.exe` runs on clean Windows machine with no Python installed
