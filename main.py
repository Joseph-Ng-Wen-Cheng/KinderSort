"""
main.py — KinderSort GUI entry point (updated for responsiveness and logging).

Improvements:
- Stronger input validation before starting:
  - Ensure Reference folder contains at least one supported image file.
  - Ensure Events folder contains at least one supported image file (recursively).
  - Prevent selecting the same path for Reference/Events/Output.
  - Test that Output folder is creatable and writable by attempting a small temp file.
  - Provide clear messagebox dialogs for each invalid input case.
- Helper functions added for image detection and writability checks.
- Other behavior preserved.
"""

import threading
import time
import queue
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tempfile

from sorter import PhotoSorter
from utils import setup_logger

# supported image extensions
_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class KinderSortApp(tk.Tk):
    """Main application window for KinderSort — Student Photo Organiser."""

    MIN_WIDTH = 500
    MIN_HEIGHT = 420
    _QUEUE_POLL_MS = 120  # how often the GUI polls the worker queue
    _TICK_INTERVAL_MS = 400  # spinner tick interval (lower CPU usage than 250ms)

    def __init__(self) -> None:
        """Initialise the window, build all widgets, and configure layout."""
        super().__init__()
        self.title("KinderSort v1.1 — Student Photo Organiser")
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.resizable(True, True)

        # StringVars for the three folder paths
        self._reference_var = tk.StringVar()
        self._events_var = tk.StringVar()
        self._output_var = tk.StringVar()

        # Mode selector
        self._mode_var = tk.StringVar(value="auto")

        # Cancellation flag shared between GUI and worker thread
        self._cancel_flag = threading.Event()

        # Worker -> GUI queue (single place to marshal updates)
        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()

        # Spinner / elapsed timer state
        self._spinner_frames = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
        self._spinner_idx = 0
        self._sort_start_time: float | None = None
        self._ticker_id: str | None = None

        # UI logger (will be set when starting)
        self.logger: logging.Logger | None = None

        self._build_ui()

        # Start polling the queue so messages from worker are handled on main thread
        self.after(self._QUEUE_POLL_MS, self._process_queue)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and pack all widgets into the main window."""
        root_frame = tk.Frame(self, padx=16, pady=16)
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        tk.Label(
            root_frame,
            text="KinderSort — Student Photo Organiser",
            font=("Helvetica", 14, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        # Folder selector rows
        folders_frame = tk.LabelFrame(root_frame, text="Folders", padx=8, pady=8)
        folders_frame.pack(fill=tk.X, pady=(0, 12))

        self._build_folder_row(folders_frame, "Reference Photos:", self._reference_var, 0)
        self._build_folder_row(folders_frame, "Events Folder:", self._events_var, 1)
        self._build_folder_row(folders_frame, "Output Folder:", self._output_var, 2)

        folders_frame.columnconfigure(1, weight=1)

        # Mode selector (below folders)
        mode_frame = tk.Frame(root_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(mode_frame, text="Mode:", anchor="w").pack(side=tk.LEFT)
        self._mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self._mode_var,
            values=["auto", "fast", "balanced", "accurate"],
            state="readonly",
            width=12,
        )
        self._mode_combo.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(mode_frame, text="(auto picks lighter settings on low-end machines)", fg="#555555").pack(side=tk.LEFT, padx=(8, 0))

        # Start / Cancel buttons
        btn_frame = tk.Frame(root_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 12))

        self._start_btn = tk.Button(
            btn_frame,
            text="Start Sorting",
            font=("Helvetica", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#388E3C",
            activeforeground="white",
            padx=16,
            pady=8,
            command=self._on_start,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._cancel_btn = tk.Button(
            btn_frame,
            text="Cancel",
            font=("Helvetica", 11),
            padx=16,
            pady=8,
            state=tk.DISABLED,
            command=self._on_cancel,
        )
        self._cancel_btn.pack(side=tk.LEFT)

        # Progress section
        self._build_progress_section(root_frame)

        # Summary box
        self._build_summary_box(root_frame)

    def _build_folder_row(
        self,
        parent: tk.Widget,
        label_text: str,
        string_var: tk.StringVar,
        row: int,
    ) -> None:
        """Create a label + read-only entry + browse button row inside parent.

        Args:
            parent: Container widget (expects grid layout).
            label_text: Text displayed on the left label.
            string_var: StringVar bound to the entry widget.
            row: Grid row index.
        """
        tk.Label(parent, text=label_text, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )

        entry = tk.Entry(parent, textvariable=string_var, state="readonly", width=40)
        entry.grid(row=row, column=1, sticky="ew", pady=4)

        btn = tk.Button(
            parent,
            text="Browse…",
            command=lambda v=string_var: self._browse_folder(v),
        )
        btn.grid(row=row, column=2, padx=(8, 0), pady=4)

    def _build_progress_section(self, parent: tk.Widget) -> None:
        """Build the progress bar and status label."""
        progress_frame = tk.LabelFrame(parent, text="Progress", padx=8, pady=8)
        progress_frame.pack(fill=tk.X, pady=(0, 12))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 4))

        self._status_label = tk.Label(
            progress_frame, text="Ready.", anchor="w", wraplength=460
        )
        self._status_label.pack(fill=tk.X)

        self._timer_label = tk.Label(
            progress_frame, text="", anchor="w", fg="#555555"
        )
        self._timer_label.pack(fill=tk.X)

    def _build_summary_box(self, parent: tk.Widget) -> None:
        """Build the read-only summary text box shown after completion."""
        summary_frame = tk.LabelFrame(parent, text="Summary", padx=8, pady=8)
        summary_frame.pack(fill=tk.BOTH, expand=True)

        self._summary_text = tk.Text(
            summary_frame, height=5, state=tk.DISABLED, wrap=tk.WORD
        )
        self._summary_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_folder(self, string_var: tk.StringVar) -> None:
        """Open a directory chooser and update string_var with the selection."""
        folder = filedialog.askdirectory(title="Select folder")
        if folder:
            string_var.set(folder)

    def _on_start(self) -> None:
        """Validate inputs then launch the worker thread for all heavy work."""
        ref = self._reference_var.get().strip()
        events = self._events_var.get().strip()
        output = self._output_var.get().strip()

        # Basic presence check
        if not ref or not events or not output:
            messagebox.showerror(
                "Missing folders",
                "Please select all three folders before starting.",
            )
            return

        ref_path = Path(ref)
        events_path = Path(events)
        output_path = Path(output)

        # Validate paths and contents
        ok, msg = self._validate_inputs(ref_path, events_path, output_path)
        if not ok:
            messagebox.showerror("Input validation failed", msg)
            return

        # Disable start, enable cancel before launching thread
        self._start_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._cancel_flag.clear()
        self._clear_summary()
        self._progress_var.set(0)
        self._set_status("Loading reference photos…")
        self._start_ticker()

        # Prepare logger and sorter
        logger = setup_logger(output_path)
        self.logger = logger
        mode = self._mode_var.get()
        sorter = PhotoSorter(ref_path, events_path, output_path, logger, mode=mode)

        # Start background worker thread; it posts messages to self._msg_queue
        thread = threading.Thread(
            target=self._run_sorting, args=(sorter,), daemon=True
        )
        thread.start()

    def _run_sorting(self, sorter: PhotoSorter) -> None:
        """Worker thread: load references, then sort all photos."""
        try:
            skipped_names = sorter.load_references(
                progress_callback=lambda c, t, n: self._msg_queue.put(("ref_progress", c, t, n))
            )
        except Exception as exc:
            # Post full traceback string for the GUI to display
            self._msg_queue.put(("error", f"Reference load failed: {exc}"))
            return

        if skipped_names:
            self._msg_queue.put(("ref_warning", skipped_names))

        if not sorter._student_encodings:
            self._msg_queue.put(("error", "No student faces could be loaded. Please check your Reference folder."))
            return

        try:
            summary = sorter.sort_all(
                progress_callback=lambda c, t, fn: self._msg_queue.put(("progress", c, t, fn)),
                cancelled=lambda: self._cancel_flag.is_set(),
            )
            self._msg_queue.put(("done", summary))
        except Exception as exc:
            self._msg_queue.put(("error", f"Unexpected error during sorting: {exc}"))

    def _start_ticker(self) -> None:
        """Start the spinning clock emoji and elapsed timer."""
        self._sort_start_time = time.monotonic()
        self._spinner_idx = 0
        self._tick()

    def _tick(self) -> None:
        """Update spinner and elapsed time every _TICK_INTERVAL_MS ms."""
        if self._sort_start_time is None:
            return
        elapsed = int(time.monotonic() - self._sort_start_time)
        minutes, seconds = divmod(elapsed, 60)
        spinner = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._timer_label.config(text=f"{spinner}  {minutes:02d}:{seconds:02d} elapsed")
        self._ticker_id = self.after(self._TICK_INTERVAL_MS, self._tick)

    def _stop_ticker(self, final_elapsed: int | None = None) -> None:
        """Stop the spinner and show final elapsed time."""
        if self._ticker_id:
            self.after_cancel(self._ticker_id)
            self._ticker_id = None
        if final_elapsed is not None:
            minutes, seconds = divmod(final_elapsed, 60)
            self._timer_label.config(text=f"✅  Done in {minutes:02d}:{seconds:02d}")
        else:
            self._timer_label.config(text="")
        self._sort_start_time = None

    def _on_cancel(self) -> None:
        """Signal the worker thread to stop after the current image."""
        self._cancel_flag.set()
        self._cancel_btn.config(state=tk.DISABLED)
        self._set_status("Cancelling… (finishing current image)")
        if self.logger:
            self.logger.info("User requested cancellation.")

    # ------------------------------------------------------------------
    # Worker->GUI queue processing (runs on main thread)
    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        """Poll the message queue and update UI on the main thread."""
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                typ = msg[0]
                if typ == "ref_progress":
                    _, current, total, name = msg
                    self._set_status(f"Loading references [{current}/{total}]: {name}…")
                elif typ == "ref_warning":
                    _, skipped = msg
                    self._show_ref_warning(skipped)
                elif typ == "progress":
                    _, current, total, filename = msg
                    self._apply_progress(current, total, filename)
                elif typ == "done":
                    _, summary = msg
                    self._on_done(summary)
                elif typ == "error":
                    _, message = msg
                    # Log and show error dialog
                    if self.logger:
                        self.logger.error("Worker error: %s", message)
                    self._on_error(message)
                else:
                    if self.logger:
                        self.logger.debug("Unknown queue message: %s", msg)
        except queue.Empty:
            pass
        finally:
            # Poll again later
            self.after(self._QUEUE_POLL_MS, self._process_queue)

    # ------------------------------------------------------------------
    # Cross-thread callbacks (now driven via the queue)
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        """Compatibility stub; progress updates come via queue._process_queue."""
        self._apply_progress(current, total, filename)

    def _apply_progress(self, current: int, total: int, filename: str) -> None:
        """Apply progress update on main thread."""
        pct = (current / total * 100) if total else 0
        self._progress_var.set(pct)
        self._set_status(f"[{current}/{total}] {filename}")

    def _on_done(self, summary: dict[str, int]) -> None:
        """Show summary and re-enable controls after successful completion."""
        elapsed = int(time.monotonic() - self._sort_start_time) if self._sort_start_time else None
        self._stop_ticker(final_elapsed=elapsed)
        self._start_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._progress_var.set(100)

        cancelled = self._cancel_flag.is_set()
        status = "Sorting cancelled." if cancelled else "Sorting complete."
        self._set_status(status)

        lines = [
            status,
            "",
            f"Total images found : {summary['total']}",
            f"Matched (sorted)   : {summary['matched']}",
            f"Unmatched          : {summary['unmatched']}",
            f"Skipped (errors)   : {summary['skipped']}",
        ]
        self._write_summary("\n".join(lines))

        if summary["total"] == 0:
            messagebox.showwarning(
                "No images found",
                "No photos were found in the Events folder.\n\n"
                "Make sure your Events folder contains photos (or sub-folders with photos).\n"
                "Supported formats: .jpg  .jpeg  .png  .bmp  .webp",
            )

    def _on_error(self, message: str) -> None:
        """Show an error dialog and re-enable controls."""
        self._stop_ticker()
        self._start_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._set_status("An error occurred.")
        try:
            messagebox.showerror("Unexpected error", message)
        except Exception:
            if self.logger:
                self.logger.exception("Failed to show error dialog: %s", message)

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, ref_path: Path, events_path: Path, output_path: Path) -> tuple[bool, str]:
        """Validate user-selected folders before starting.

        Returns:
            (ok, message) where ok is False and message describes the problem.
        """
        # existence checks
        if not ref_path.exists() or not ref_path.is_dir():
            return False, f"Reference folder does not exist or is not a directory:\n{ref_path}"
        if not events_path.exists() or not events_path.is_dir():
            return False, f"Events folder does not exist or is not a directory:\n{events_path}"

        # disallow picking the same folder for multiple roles
        try:
            ref_resolved = ref_path.resolve()
            events_resolved = events_path.resolve()
            output_resolved = output_path.resolve()
        except Exception:
            # fallback to raw paths if resolve() fails
            ref_resolved = ref_path
            events_resolved = events_path
            output_resolved = output_path

        if ref_resolved == events_resolved:
            return False, "Reference and Events folders must be different folders."
        if ref_resolved == output_resolved or events_resolved == output_resolved:
            return False, "Output folder should be different from Reference and Events folders."

        # Reference must contain at least one supported image file
        if not self._has_images_in_dir(ref_path):
            return False, (
                "Reference folder does not contain any supported image files.\n"
                f"Supported formats: {', '.join(sorted(_SUPPORTED_IMAGE_EXTS))}\n"
                f"Please add one clear photo per student to the Reference folder."
            )

        # Events must contain at least one supported image (recursively)
        if not self._has_images_in_tree(events_path):
            return False, (
                "Events folder does not contain any supported image files.\n"
                "Make sure the Events folder contains images or sub-folders with images."
            )

        # Ensure output folder can be created/written to
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"Cannot create output folder:\n{exc}"

        writable, reason = self._test_output_writable(output_path)
        if not writable:
            return False, f"Output folder is not writable: {reason}"

        return True, "OK"

    def _has_images_in_dir(self, path: Path) -> bool:
        """Return True if the directory contains at least one supported image file (non-recursive)."""
        try:
            for p in path.iterdir():
                if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTS:
                    return True
        except Exception:
            # in case of permission errors etc.
            return False
        return False

    def _has_images_in_tree(self, path: Path, max_checks: int = 5000) -> bool:
        """Return True if any supported image exists under path (recursive).
        Limits the number of files checked to avoid very long scans on huge trees.
        """
        count = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    count += 1
                    if p.suffix.lower() in _SUPPORTED_IMAGE_EXTS:
                        return True
                if count >= max_checks:
                    break
        except Exception:
            return False
        return False

    def _test_output_writable(self, path: Path) -> tuple[bool, str]:
        """Try to create and remove a tiny temp file in output folder to verify writability."""
        try:
            # Use a small named temp file inside the output folder
            fd, tmp_path = tempfile.mkstemp(prefix=".kinder_sort_test_", dir=str(path))
            try:
                with open(fd, "wb") as f:
                    f.write(b"x")
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
            return True, ""
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Worker->GUI queue processing (runs on main thread)
    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        """Poll the message queue and update UI on the main thread."""
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                typ = msg[0]
                if typ == "ref_progress":
                    _, current, total, name = msg
                    self._set_status(f"Loading references [{current}/{total}]: {name}…")
                elif typ == "ref_warning":
                    _, skipped = msg
                    self._show_ref_warning(skipped)
                elif typ == "progress":
                    _, current, total, filename = msg
                    self._apply_progress(current, total, filename)
                elif typ == "done":
                    _, summary = msg
                    self._on_done(summary)
                elif typ == "error":
                    _, message = msg
                    # Log and show error dialog
                    if self.logger:
                        self.logger.error("Worker error: %s", message)
                    self._on_error(message)
                else:
                    if self.logger:
                        self.logger.debug("Unknown queue message: %s", msg)
        except queue.Empty:
            pass
        finally:
            # Poll again later
            self.after(self._QUEUE_POLL_MS, self._process_queue)

    # ------------------------------------------------------------------
    # Cross-thread callbacks (now driven via the queue)
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        """Compatibility stub; progress updates come via queue._process_queue."""
        self._apply_progress(current, total, filename)

    def _apply_progress(self, current: int, total: int, filename: str) -> None:
        """Apply progress update on main thread."""
        pct = (current / total * 100) if total else 0
        self._progress_var.set(pct)
        self._set_status(f"[{current}/{total}] {filename}")

    def _on_done(self, summary: dict[str, int]) -> None:
        """Show summary and re-enable controls after successful completion."""
        elapsed = int(time.monotonic() - self._sort_start_time) if self._sort_start_time else None
        self._stop_ticker(final_elapsed=elapsed)
        self._start_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._progress_var.set(100)

        cancelled = self._cancel_flag.is_set()
        status = "Sorting cancelled." if cancelled else "Sorting complete."
        self._set_status(status)

        lines = [
            status,
            "",
            f"Total images found : {summary['total']}",
            f"Matched (sorted)   : {summary['matched']}",
            f"Unmatched          : {summary['unmatched']}",
            f"Skipped (errors)   : {summary['skipped']}",
        ]
        self._write_summary("\n".join(lines))

        if summary["total"] == 0:
            messagebox.showwarning(
                "No images found",
                "No photos were found in the Events folder.\n\n"
                "Make sure your Events folder contains photos (or sub-folders with photos).\n"
                "Supported formats: .jpg  .jpeg  .png  .bmp  .webp",
            )

    def _on_error(self, message: str) -> None:
        """Show an error dialog and re-enable controls."""
        self._stop_ticker()
        self._start_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._set_status("An error occurred.")
        try:
            messagebox.showerror("Unexpected error", message)
        except Exception:
            if self.logger:
                self.logger.exception("Failed to show error dialog: %s", message)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        """Update the status label text."""
        self._status_label.config(text=text)

    def _write_summary(self, text: str) -> None:
        """Write text into the read-only summary box."""
        self._summary_text.config(state=tk.NORMAL)
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert(tk.END, text)
        self._summary_text.config(state=tk.DISABLED)

    def _clear_summary(self) -> None:
        """Clear the summary text box."""
        self._write_summary("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the KinderSort GUI application."""
    app = KinderSortApp()
    app.mainloop()


if __name__ == "__main__":
    main()
