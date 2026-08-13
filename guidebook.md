# KinderSort — Teacher's Guide

*How to sort your students' event photos automatically*

---

## What KinderSort Does

KinderSort looks at each photo from a school event and finds your students' faces.
It then automatically puts each photo into the right student's folder — so you don't
have to sort hundreds of photos by hand!

**One photo can appear in multiple folders** — for example, a group shot with three
students will be copied to all three students' folders.

---

## Before You Start

You need three folders ready on your computer:

### 1. Reference Photos Folder
One clear, front-facing photo of each student.
- Name each photo with the student's full name
- Examples: `Ali.jpg`, `Siti.png`, `Kumar.jpeg`
- Make sure the face is clearly visible and well-lit

### 2. Events Folder
A folder containing **subfolders** — one subfolder per event.
- Example structure:
  ```
  Events/
      Sports_Day/
          IMG_001.jpg
          IMG_002.jpg
      Field_Trip/
          IMG_003.jpg
  ```
- ⚠️ **Important:** Photos must be inside a subfolder (the event name), not directly in the Events folder.

### 3. Output Folder
An empty folder where KinderSort will save the sorted results.
You can create a new empty folder anywhere on your computer.

---

## Step-by-Step Guide

### Step 1 — Open KinderSort

Double-click the **KinderSort.exe** file to launch the app.

![KinderSort on launch](guidebook_assets/01_launch.png)

You will see three folder selector rows at the top.

---

### Step 2 — Select Your Reference Photos Folder

Click the **Browse…** button next to "Reference Photos".

Navigate to your folder of student reference photos and click **Select Folder**.

![Reference folder selected](guidebook_assets/02_reference_selected.png)

The path to your folder will appear in the box.

---

### Step 3 — Select Your Events Folder

Click the **Browse…** button next to "Events Folder".

Navigate to your Events folder (the one containing event subfolders) and click **Select Folder**.

![Events folder selected](guidebook_assets/03_events_selected.png)

---

### Step 4 — Select Your Output Folder

Click the **Browse…** button next to "Output Folder".

Navigate to your empty output folder and click **Select Folder**.

When all three folders are selected, the app is ready.

![All three folders selected](guidebook_assets/04_all_folders_set.png)

---

### Step 5 — Start Sorting

Click the large green **Start Sorting** button.

KinderSort will begin processing your photos. You will see:
- A **progress bar** filling up as photos are processed
- A **status line** showing the current photo filename

![Sorting in progress](guidebook_assets/05_sorting_in_progress.png)

> ⏱️ Processing time depends on the number of photos. Expect about 1–2 minutes per 10 photos on a normal computer.

You can click **Cancel** at any time to stop — photos processed so far will be saved.

---

### Step 6 — Review the Results

When sorting is complete, you will see a summary:

![Sorting complete](guidebook_assets/06_sorting_complete.png)

The summary shows:
- **Total images found** — how many photos were scanned
- **Matched (sorted)** — how many photos were placed into student folders
- **Unmatched** — photos where no student face was recognised
- **Skipped (errors)** — photos that could not be opened

---

## Understanding Your Output Folder

After sorting, your Output folder will look like this:

```
Output/
    Ali/
        Sports_Day__IMG_001.jpg
        Concert__IMG_045.jpg
    Siti/
        Sports_Day__IMG_001.jpg   ← same group photo, copied here too
        Field_Trip__IMG_023.jpg
    _unmatched/
        blurry_photo.jpg
        background_only.jpg
    kindersort_log.txt
```

**Each student has their own folder** containing all photos where their face was detected.

The **`_unmatched` folder** contains:
- Photos where no faces were detected (e.g. landscape shots, blurry images)
- Photos where faces were detected but didn't match any student (e.g. teachers, parents)

The **`kindersort_log.txt` file** is a detailed record of everything KinderSort did.
You don't need to read it normally, but it's useful if something seems wrong.

---

## Common Problems & Solutions

### "No face was detected" warning on startup
**Cause:** One of your reference photos doesn't show the student's face clearly enough.

**Fix:** Replace that student's reference photo with a clearer, front-facing photo.
Good lighting and a plain background work best.

### Many photos end up in `_unmatched`
**Possible causes:**
1. Reference photo quality is poor — use a clearer photo
2. Event photos are very blurry or taken from far away
3. Students are wearing face masks or hats in the event photos

**Fix:** Try better reference photos first. KinderSort cannot recognise faces that
are partially covered or turned away.

### "Missing folders" error when clicking Start
**Fix:** Make sure all three folder fields are filled in before clicking Start Sorting.

### The app seems stuck / progress bar not moving
**Cause:** Face recognition is CPU-intensive — it is still working.

**Fix:** Wait patiently. A batch of 100 photos may take 10–15 minutes on an older computer.
Watch the status line — if it shows a new filename, it is still running.

### Output folder photos have long names like `Sports_Day__IMG_001.jpg`
This is normal! The event folder name is added as a prefix so you always know
which event a photo came from.

---

## Tips for Better Results

1. **Use a clear, recent reference photo** — front-facing, good lighting, no sunglasses
2. **One face per reference photo** — if a reference photo has multiple faces, KinderSort uses the first face detected
3. **Consistent lighting** helps — very dark or backlit event photos may not match well
4. **Re-run with a higher tolerance** if you're missing matches — ask your IT support to adjust the `DISTANCE_THRESHOLD` in `sorter.py` from `0.5` to `0.6`
5. **Keep events in subfolders** — KinderSort only reads photos inside named subfolders of the Events folder

---

*KinderSort — Student Photo Organiser | Runs fully offline, no internet required*
