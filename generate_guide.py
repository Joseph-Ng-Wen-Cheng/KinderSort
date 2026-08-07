"""
generate_guide.py — Automated screenshot guidebook generator for KinderSort.

Launches the KinderSort GUI as a subprocess, automates UI interactions using
pywinauto, captures screenshots at key states, then writes guidebook.md with
embedded image references and an optional .docx export.

Requirements:
    pip install pyautogui pywinauto pillow python-docx

Usage:
    python generate_guide.py
"""
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pyautogui
import pywinauto
from pywinauto import Application

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSETS_DIR = Path("guidebook_assets")
PYTHON_EXE = Path(".venv/Scripts/python.exe")
APP_TITLE = "KinderSort — Student Photo Organiser"
REF_FOLDER = str(Path("referencePhoto").resolve())
EVENTS_FOLDER = str(Path("Events").resolve())
OUTPUT_FOLDER = str(Path("Output").resolve())

# Paths to pre-run screenshots (if app already ran)
WAIT_SECONDS = 4  # Time to wait for window to fully render


def setup_assets_dir() -> None:
    """Create the guidebook_assets directory if it doesn't exist."""
    ASSETS_DIR.mkdir(exist_ok=True)
    print(f"Assets directory: {ASSETS_DIR.resolve()}")


def wait_for_window(title: str, timeout: int = 20) -> Application:
    """Poll until a window with the given title appears, then return it.

    Uses win32 backend which works reliably with tkinter applications.

    Args:
        title: Exact window title to search for.
        timeout: Seconds to wait before raising TimeoutError.

    Returns:
        Connected pywinauto Application instance.

    Raises:
        TimeoutError: If window doesn't appear within timeout seconds.
    """
    print(f"Waiting for window '{title}'...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            app = Application(backend="win32").connect(title=title, timeout=2)
            print("Window found!")
            return app
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise TimeoutError(f"Window '{title}' did not appear within {timeout}s")


def screenshot(name: str) -> Path:
    """Take a screenshot and save to guidebook_assets/{name}.

    Args:
        name: Filename without extension (e.g. '01_launch').

    Returns:
        Path to the saved screenshot file.
    """
    time.sleep(0.5)  # Brief pause to let UI settle
    path = ASSETS_DIR / f"{name}.png"
    pyautogui.screenshot(str(path))
    print(f"  Screenshot saved: {path.name}")
    return path


def fill_folder_entry(win, row_index: int, folder_path: str) -> None:
    """Click the Browse button for a folder row and handle the file dialog.

    In tkinter, all button labels are empty strings via win32 API (tkinter doesn't
    expose button text via Win32 WM_GETTEXT). Buttons are in creation order:
    index 0=Browse ref, 1=Browse events, 2=Browse output, 3=Start, 4=Cancel.

    Args:
        win: pywinauto window wrapper (win32 backend).
        row_index: 0-based index of the folder row (0=Reference, 1=Events, 2=Output).
        folder_path: Absolute path string to enter in the dialog.
    """
    try:
        all_buttons = win.descendants(class_name="Button")
        # Button order in tkinter: Browse(0), Browse(1), Browse(2), Start(3), Cancel(4)
        if row_index < len(all_buttons):
            all_buttons[row_index].click_input()
            time.sleep(1.5)  # Wait for dialog to open
        else:
            print(f"  Warning: Button index {row_index} out of range ({len(all_buttons)} buttons found)")
            return
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: Button click failed: {e}")
        return

    # Navigate to path in the open tkinter askdirectory dialog
    # tkinter uses the Windows SHELL folder browser — type path into address bar
    try:
        pyautogui.hotkey("alt", "d")  # Focus address bar
        time.sleep(0.4)
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)
        pyautogui.typewrite(folder_path, interval=0.02)
        pyautogui.press("enter")
        time.sleep(1.2)
        pyautogui.press("enter")  # Confirm selection ("Select Folder" / OK)
        time.sleep(0.8)
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: Dialog navigation failed: {e}")


def run_guide_capture() -> None:
    """Main automation routine — launch app, capture screenshots, write guide."""
    setup_assets_dir()

    # -----------------------------------------------------------------------
    # Launch the app
    # -----------------------------------------------------------------------
    print("\n[1/7] Launching KinderSort...")
    proc = subprocess.Popen(
        [str(PYTHON_EXE), "main.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    try:
        app = wait_for_window(APP_TITLE, timeout=20)
        win = app.window(title=APP_TITLE)
        win.set_focus()
        time.sleep(WAIT_SECONDS)

        # -----------------------------------------------------------------------
        # State 1: Fresh launch
        # -----------------------------------------------------------------------
        print("[1/7] State 1: App on first launch (blank)")
        win.set_focus()
        screenshot("01_launch")

        # -----------------------------------------------------------------------
        # State 2: Set Reference folder
        # -----------------------------------------------------------------------
        print("[2/7] State 2: Selecting Reference Photos folder")
        fill_folder_entry(win, 0, REF_FOLDER)
        win.set_focus()
        screenshot("02_reference_selected")

        # -----------------------------------------------------------------------
        # State 3: Set Events folder
        # -----------------------------------------------------------------------
        print("[3/7] State 3: Selecting Events folder")
        fill_folder_entry(win, 1, EVENTS_FOLDER)
        win.set_focus()
        screenshot("03_events_selected")

        # -----------------------------------------------------------------------
        # State 4: Set Output folder → all three selected (ready state)
        # -----------------------------------------------------------------------
        print("[4/7] State 4: Selecting Output folder (all three selected)")
        fill_folder_entry(win, 2, OUTPUT_FOLDER)
        win.set_focus()
        screenshot("04_all_folders_set")

        # -----------------------------------------------------------------------
        # State 5: Click Start Sorting → capture mid-progress
        # -----------------------------------------------------------------------
        print("[5/7] State 5: Starting sort and capturing progress")
        try:
            all_buttons = win.descendants(class_name="Button")
            # Start Sorting is button index 3 (after 3 Browse buttons)
            if len(all_buttons) >= 4:
                all_buttons[3].click_input()
                print("  Clicked Start Sorting (button index 3)")
            else:
                print(f"  Warning: Not enough buttons found ({len(all_buttons)})")
        except Exception as e:  # noqa: BLE001
            print(f"  Warning: Could not click Start: {e}")

        # Wait ~20s then screenshot mid-progress
        time.sleep(20)
        win.set_focus()
        screenshot("05_sorting_in_progress")

        # -----------------------------------------------------------------------
        # State 6: Wait for completion
        # -----------------------------------------------------------------------
        print("[6/7] State 6: Waiting for sorting to complete...")
        # Poll every 10s for up to 15 minutes
        deadline = time.time() + 900
        while time.time() < deadline:
            time.sleep(10)
            # Check if the Start button (index 3) is re-enabled (indicates completion)
            try:
                all_buttons = win.descendants(class_name="Button")
                if len(all_buttons) >= 4 and all_buttons[3].is_enabled():
                    print("  Sorting complete!")
                    break
            except Exception:  # noqa: BLE001
                pass

        try:
            win.set_focus()
        except Exception:  # noqa: BLE001
            pass
        screenshot("06_sorting_complete")

        print("\n[7/7] All screenshots captured!")
        print(f"Screenshots in: {ASSETS_DIR.resolve()}")

    finally:
        # Don't kill the process — user may want to inspect results
        print(f"\nApp still running (PID {proc.pid}). Close it manually when done.")


# --- New: explain sorting result helper -----------------------------------


def explain_sorting_result(
    total_images: int,
    matched: int,
    unmatched: int,
    skipped: int,
    per_student_counts: Optional[Dict[str, int]] = None,
    low_confidence: Optional[int] = None,
) -> str:
    """
    Produce a human-friendly explanation of the sorting result.

    Parameters:
      - total_images: Number of photos scanned.
      - matched: Photos that were placed into one or more student folders.
      - unmatched: Photos where no student match was selected.
      - skipped: Photos that couldn't be opened or processed.
      - per_student_counts: Optional mapping student name -> number of photos assigned.
      - low_confidence: Optional count of matches considered low-confidence.

    Returns:
      A multi-paragraph string explaining what the numbers mean and next steps.
    """
    if total_images <= 0:
        return "No images were processed."

    lines = []
    pct = lambda n: f"{(n / total_images * 100):.1f}%"

    lines.append(
        f"Summary: {total_images} images scanned — {matched} matched ({pct(matched)}), "
        f"{unmatched} unmatched ({pct(unmatched)}), {skipped} skipped due to errors."
    )

    # Interpretation
    lines.append(
        "Interpretation:\n"
        "- Matched: these photos were placed into one or more student folders because the system "
        "found faces that matched your reference photos.\n"
        "- Unmatched: no confident match was found; these photos are in the `_unmatched` folder.\n"
        "- Skipped: files that could not be opened or processed (corrupt files, unsupported formats)."
    )

    if low_confidence:
        lines.append(
            f"Low-confidence matches: {low_confidence} photos were matched but the system had low"
            " confidence. These are good candidates for a quick manual review — you may want to remove"
            " or replace the reference photo if many low-confidence matches involve the same student."
        )

    # Per-student highlights
    if per_student_counts:
        total_assigned = sum(per_student_counts.values())
        top_items = sorted(per_student_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        lines.append("Per-student sample counts (top results):")
        for name, count in top_items:
            lines.append(f"- {name}: {count} photos")
        if total_assigned != matched:
            lines.append(
                f"(Note: a single photo may be assigned to multiple students; total assignments = {total_assigned})"
            )

    # Actionable next steps
    lines.append(
        "Next steps & tips:\n"
        "1. Review `_unmatched` first — these are photos where no student was recognised. Look for common causes:\n"
        "   • Very small faces (zoomed-out group shots)\n"
        "   • Motion blur or very poor lighting\n"
        "   • Faces covered by hats/sunglasses/masks\n"
        "2. If many photos are in `_unmatched` for a particular event, ensure event photos are inside subfolders named for the event.\n"
        "3. For low-confidence matches, open the student's folder and quickly scan those images. If there are false positives, "
        "consider improving that student's reference photo (clear, front-facing) and re-run.\n"
        "4. To change sensitivity, adjust the DISTANCE_THRESHOLD in sorter.py — lower values are stricter (fewer false positives), "
        "higher values are more permissive (more matches but possibly more false positives).\n"
        "5. If you need detailed per-image information, open `kindersort_log.txt` in the Output folder. It typically contains the image name, "
        "matched student(s) and (where available) the matching confidence or distance for each image — useful for triage."
    )

    return "\n\n".join(lines)


def write_guidebook_md() -> None:
    """Write the teacher guidebook as guidebook.md with embedded screenshots."""
    print("\nWriting guidebook.md...")

    # Create an example dynamic explanation using plausible numbers.
    # When producing the final guide from within the sorter, replace these example
    # values with the real metrics and call explain_sorting_result(...) to get
    # a real-time summary for teachers.
    example_explanation = explain_sorting_result(
        total_images=120,
        matched=95,
        unmatched=20,
        skipped=5,
        per_student_counts={"Ali": 12, "Siti": 10, "Kumar": 9, "Zara": 8},
        low_confidence=7,
    )

    content = f"""# KinderSort — Teacher's Guide

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
