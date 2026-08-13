"""
quick_screenshots.py — Reliable GUI screenshots via programmatic tkinter state injection.

Instead of automating dialogs, this script directly sets the StringVars in the
KinderSortApp instance and takes screenshots at each key state using PIL.ImageGrab.

Usage:
    python quick_screenshots.py
"""

import sys
import threading
import time
from pathlib import Path

# Inject paths relative to this script
HERE = Path(__file__).parent.resolve()
REF_FOLDER = str(HERE / "referencePhoto")
EVENTS_FOLDER = str(HERE / "Events")
OUTPUT_FOLDER = str(HERE / "Output")
ASSETS_DIR = HERE / "guidebook_assets"

ASSETS_DIR.mkdir(exist_ok=True)


def grab_window_screenshot(name: str, root=None) -> None:
    """Capture the app window using PIL.ImageGrab and save to guidebook_assets/.

    On Windows, ImageGrab.grab() captures the full screen. We crop to the
    window bounds using the tkinter geometry.

    Args:
        name: Output filename stem (e.g. '01_launch').
        root: tkinter root window to get geometry from.
    """
    from PIL import ImageGrab

    time.sleep(0.3)  # Let UI settle

    if root:
        root.update_idletasks()
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        w = root.winfo_width()
        h = root.winfo_height()
        # Add a small margin for the title bar
        title_bar = 30
        img = ImageGrab.grab(bbox=(x, y - title_bar, x + w, y + h))
    else:
        img = ImageGrab.grab()

    out_path = ASSETS_DIR / f"{name}.png"
    img.save(str(out_path))
    print(f"  Screenshot saved: {name}.png ({img.width}x{img.height})")


def run_with_screenshots() -> None:
    """Launch KinderSortApp and inject state for each screenshot."""
    # Import app here so tkinter initializes in the main thread
    sys.path.insert(0, str(HERE))
    from main import KinderSortApp

    app = KinderSortApp()
    app.update()

    # --- State 1: Fresh launch ---
    print("[1/6] State 1: Fresh launch")
    app.after(500, lambda: grab_window_screenshot("01_launch", app))

    def state2():
        """Set reference folder and screenshot."""
        print("[2/6] State 2: Reference folder selected")
        app._reference_var.set(REF_FOLDER)
        app.update_idletasks()
        grab_window_screenshot("02_reference_selected", app)
        app.after(600, state3)

    def state3():
        """Set events folder and screenshot."""
        print("[3/6] State 3: Events folder selected")
        app._events_var.set(EVENTS_FOLDER)
        app.update_idletasks()
        grab_window_screenshot("03_events_selected", app)
        app.after(600, state4)

    def state4():
        """Set output folder → all three set → screenshot."""
        print("[4/6] State 4: All three folders set (ready state)")
        app._output_var.set(OUTPUT_FOLDER)
        app.update_idletasks()
        grab_window_screenshot("04_all_folders_set", app)
        app.after(800, state5)

    def state5():
        """Click Start Sorting and take screenshot mid-progress."""
        print("[5/6] State 5: Starting sort...")
        # Trigger the start directly
        app._on_start()
        app.update_idletasks()
        # Take screenshot after ~15s to catch mid-progress
        app.after(15000, state5b)

    def state5b():
        """Take the mid-progress screenshot."""
        grab_window_screenshot("05_sorting_in_progress", app)
        print("  Mid-progress screenshot taken, waiting for completion...")

    def check_done():
        """Poll until sort finishes, then take final screenshot."""
        if app._start_btn["state"] == "normal":
            print("[6/6] State 6: Sort complete")
            app.update_idletasks()
            grab_window_screenshot("06_sorting_complete", app)
            print("\nAll screenshots saved to guidebook_assets/")
            app.after(1500, app.destroy)  # Close after 1.5s
        else:
            app.after(2000, check_done)  # Check again in 2s

    # Start after 1s so state 1 screenshot renders properly
    app.after(1000, state2)
    app.after(15000 + 2000 + 2000, check_done)  # Start polling after state5 + some buffer

    app.mainloop()


if __name__ == "__main__":
    # Create Output folder if needed
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    run_with_screenshots()
