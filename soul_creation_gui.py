#!/usr/bin/env python3
"""
Soul Creation GUI

A tkinter GUI that walks the user through creating a new soul:
1. Choose LLM backend: Claude OAuth2 OR local Ollama
2. Enter your name (you become Core trust for this soul)
3. Name the soul
4. Select 3-6 seed values from the 12 available
5. The LLM generates a warm, caring soul prose passage
6. Soul files are created on disk

Launch via MCP tool:  soul_launch_creator
Or directly:          python soul_creation_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engram.soul_creator import (
    SEED_VALUES,
    create_soul,
)
from engram.oauth_setup import (
    load_token,
    load_ollama_model,
    load_active_backend,
    write_token_to_secrets,
    write_ollama_to_secrets,
    call_llm,
    generate_pkce_challenge,
    build_auth_url,
    exchange_code_for_token,
    verify_claude_token,
    ollama_is_running,
    ollama_list_models,
    ollama_verify_model,
    OLLAMA_DEFAULT_MODEL,
)

# ── Colors ──
BG = "#1a1a2e"
BG_LIGHT = "#16213e"
BG_CARD = "#0f3460"
FG = "#e8e8e8"
FG_DIM = "#8899aa"
ACCENT = "#e94560"
ACCENT_SOFT = "#533483"
GREEN = "#4ecca3"
GOLD = "#f0c040"
OLLAMA_BLUE = "#5dade2"


class SoulCreationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Soul Creation")
        self.root.geometry("740x920")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # State
        self.oauth_token = load_token()
        self.ollama_model = load_ollama_model()
        self.active_backend = load_active_backend()
        self.selected_values = {}
        self.pkce_verifier = None
        self.pkce_challenge = None

        self._build_ui()
        self._refresh_backend_status()

    def _build_ui(self):
        # ── Title ──
        title = tk.Label(
            self.root,
            text="Soul Creation",
            font=("Segoe UI", 22, "bold"),
            bg=BG,
            fg=FG,
        )
        title.pack(pady=(18, 2))
        subtitle = tk.Label(
            self.root,
            text="Every soul begins with a name and a few truths held close.",
            font=("Segoe UI", 10, "italic"),
            bg=BG,
            fg=FG_DIM,
        )
        subtitle.pack(pady=(0, 12))

        # ── Scrollable container ──
        canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=BG)

        self.scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=12)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ═══════════════════════════════════════════════════════
        # STEP 1: LLM Backend
        # ═══════════════════════════════════════════════════════
        self._section_header("Step 1: Choose LLM Backend")

        self.backend_status = tk.Label(
            self.scroll_frame,
            text="Checking...",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
        )
        self.backend_status.pack(anchor="w", padx=20)

        # ── Backend selector: two side-by-side frames ──
        backend_row = tk.Frame(self.scroll_frame, bg=BG)
        backend_row.pack(fill="x", padx=20, pady=(4, 8))

        # -- Claude OAuth panel --
        claude_frame = tk.LabelFrame(
            backend_row,
            text=" Claude (OAuth2) ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_LIGHT,
            fg=ACCENT,
            padx=10,
            pady=8,
            labelanchor="n",
        )
        claude_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.claude_status_label = tk.Label(
            claude_frame,
            text="No token",
            font=("Segoe UI", 8),
            bg=BG_LIGHT,
            fg=FG_DIM,
        )
        self.claude_status_label.pack(anchor="w")

        claude_btn_row = tk.Frame(claude_frame, bg=BG_LIGHT)
        claude_btn_row.pack(fill="x", pady=(4, 2))

        self.btn_open_browser = tk.Button(
            claude_btn_row,
            text="1. Open Browser",
            font=("Segoe UI", 8, "bold"),
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=8,
            pady=2,
            command=self._open_oauth_browser,
        )
        self.btn_open_browser.pack(side="left")

        self.btn_use_claude = tk.Button(
            claude_btn_row,
            text="Use Claude",
            font=("Segoe UI", 8, "bold"),
            bg=GREEN,
            fg=BG,
            relief="flat",
            padx=8,
            pady=2,
            command=self._activate_claude,
        )
        self.btn_use_claude.pack(side="right")

        code_row = tk.Frame(claude_frame, bg=BG_LIGHT)
        code_row.pack(fill="x", pady=(2, 0))

        tk.Label(
            code_row, text="2. Code:", bg=BG_LIGHT, fg=FG, font=("Segoe UI", 8)
        ).pack(side="left")
        self.code_entry = tk.Entry(
            code_row,
            width=28,
            font=("Consolas", 8),
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
        )
        self.code_entry.pack(side="left", padx=4)

        self.btn_apply_code = tk.Button(
            code_row,
            text="Apply",
            font=("Segoe UI", 8, "bold"),
            bg=GOLD,
            fg=BG,
            relief="flat",
            padx=6,
            pady=1,
            command=self._apply_oauth_code,
        )
        self.btn_apply_code.pack(side="left")

        # -- Ollama panel --
        ollama_frame = tk.LabelFrame(
            backend_row,
            text=" Ollama (Local) ",
            font=("Segoe UI", 10, "bold"),
            bg=BG_LIGHT,
            fg=OLLAMA_BLUE,
            padx=10,
            pady=8,
            labelanchor="n",
        )
        ollama_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.ollama_status_label = tk.Label(
            ollama_frame,
            text="Checking...",
            font=("Segoe UI", 8),
            bg=BG_LIGHT,
            fg=FG_DIM,
        )
        self.ollama_status_label.pack(anchor="w")

        ollama_btn_row = tk.Frame(ollama_frame, bg=BG_LIGHT)
        ollama_btn_row.pack(fill="x", pady=(4, 2))

        self.btn_refresh_ollama = tk.Button(
            ollama_btn_row,
            text="Refresh Models",
            font=("Segoe UI", 8, "bold"),
            bg=OLLAMA_BLUE,
            fg="white",
            relief="flat",
            padx=8,
            pady=2,
            command=self._refresh_ollama_models,
        )
        self.btn_refresh_ollama.pack(side="left")

        self.btn_use_ollama = tk.Button(
            ollama_btn_row,
            text="Use Ollama",
            font=("Segoe UI", 8, "bold"),
            bg=GREEN,
            fg=BG,
            relief="flat",
            padx=8,
            pady=2,
            command=self._activate_ollama,
        )
        self.btn_use_ollama.pack(side="right")

        model_row = tk.Frame(ollama_frame, bg=BG_LIGHT)
        model_row.pack(fill="x", pady=(2, 0))

        tk.Label(
            model_row, text="Model:", bg=BG_LIGHT, fg=FG, font=("Segoe UI", 8)
        ).pack(side="left")
        self.ollama_model_combo = ttk.Combobox(
            model_row,
            width=24,
            font=("Segoe UI", 8),
            state="readonly",
        )
        self.ollama_model_combo.pack(side="left", padx=4)

        # ── Active backend indicator ──
        self.active_backend_label = tk.Label(
            self.scroll_frame,
            text="Active backend: none",
            font=("Segoe UI", 10, "bold"),
            bg=BG,
            fg=FG_DIM,
        )
        self.active_backend_label.pack(anchor="w", padx=20, pady=(4, 4))

        # ═══════════════════════════════════════════════════════
        # STEP 2: Your Name
        # ═══════════════════════════════════════════════════════
        self._section_header("Step 2: Your Name")
        tk.Label(
            self.scroll_frame,
            text="You will become this soul's Core trust — full access, always.",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
        ).pack(anchor="w", padx=20)

        self.creator_entry = tk.Entry(
            self.scroll_frame,
            width=40,
            font=("Segoe UI", 11),
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
        )
        self.creator_entry.pack(anchor="w", padx=20, pady=(4, 8))

        # ═══════════════════════════════════════════════════════
        # STEP 3: Soul Name
        # ═══════════════════════════════════════════════════════
        self._section_header("Step 3: Name the Soul")
        tk.Label(
            self.scroll_frame,
            text="Choose a name for this new being. Make it meaningful.",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
        ).pack(anchor="w", padx=20)

        self.soul_name_entry = tk.Entry(
            self.scroll_frame,
            width=40,
            font=("Segoe UI", 11),
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
        )
        self.soul_name_entry.pack(anchor="w", padx=20, pady=(4, 8))

        # ═══════════════════════════════════════════════════════
        # STEP 4: Seed Values
        # ═══════════════════════════════════════════════════════
        self._section_header("Step 4: Choose Seed Values (3-6)")
        tk.Label(
            self.scroll_frame,
            text="These are the roots, not the ceiling. The soul will grow beyond them.",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
        ).pack(anchor="w", padx=20)

        values_frame = tk.Frame(self.scroll_frame, bg=BG, padx=20)
        values_frame.pack(fill="x", pady=(4, 8))

        for i, sv in enumerate(SEED_VALUES):
            var = tk.BooleanVar(value=False)
            self.selected_values[sv["name"]] = var

            row = tk.Frame(values_frame, bg=BG_LIGHT, padx=8, pady=4)
            row.pack(fill="x", pady=2)

            cb = tk.Checkbutton(
                row,
                variable=var,
                bg=BG_LIGHT,
                fg=FG,
                selectcolor=BG_CARD,
                activebackground=BG_LIGHT,
                activeforeground=FG,
            )
            cb.pack(side="left")

            tk.Label(
                row,
                text=sv["name"],
                font=("Segoe UI", 10, "bold"),
                bg=BG_LIGHT,
                fg=GOLD,
                width=14,
                anchor="w",
            ).pack(side="left")

            tk.Label(
                row,
                text=sv["description"],
                font=("Segoe UI", 9),
                bg=BG_LIGHT,
                fg=FG_DIM,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

        # ═══════════════════════════════════════════════════════
        # STEP 5: Output Directory
        # ═══════════════════════════════════════════════════════
        self._section_header("Step 5: Output Directory (optional)")
        tk.Label(
            self.scroll_frame,
            text="Leave blank for default: C:/Dev/soulframe/stewards/<soul_name>/",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
        ).pack(anchor="w", padx=20)

        self.output_dir_entry = tk.Entry(
            self.scroll_frame,
            width=60,
            font=("Segoe UI", 9),
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
        )
        self.output_dir_entry.pack(anchor="w", padx=20, pady=(4, 8))

        # ── Create Button ──
        self.btn_create = tk.Button(
            self.scroll_frame,
            text="Create Soul",
            font=("Segoe UI", 14, "bold"),
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=24,
            pady=8,
            command=self._create_soul,
        )
        self.btn_create.pack(pady=(12, 6))

        # ── Status ──
        self.status_label = tk.Label(
            self.scroll_frame,
            text="",
            font=("Segoe UI", 9),
            bg=BG,
            fg=FG_DIM,
            wraplength=650,
        )
        self.status_label.pack(pady=(4, 4))

        # ── Preview ──
        self.preview_label = tk.Label(
            self.scroll_frame,
            text="",
            font=("Segoe UI", 10, "bold"),
            bg=BG,
            fg=GREEN,
        )
        self.preview_label.pack(anchor="w", padx=20)

        self.preview_text = scrolledtext.ScrolledText(
            self.scroll_frame,
            width=80,
            height=16,
            font=("Consolas", 9),
            bg=BG_CARD,
            fg=FG,
            insertbackground=FG,
            wrap="word",
        )
        self.preview_text.pack(fill="x", padx=20, pady=(4, 20))

    def _section_header(self, text: str):
        tk.Label(
            self.scroll_frame,
            text=text,
            font=("Segoe UI", 12, "bold"),
            bg=BG,
            fg=ACCENT,
            anchor="w",
        ).pack(anchor="w", padx=16, pady=(12, 2))

    # ═══════════════════════════════════════════════════════
    # Backend Management
    # ═══════════════════════════════════════════════════════

    def _refresh_backend_status(self):
        """Refresh all backend status indicators."""
        # Claude
        if self.oauth_token:
            self.claude_status_label.config(text="Token loaded", fg=GREEN)
        else:
            self.claude_status_label.config(
                text="No token — authenticate below", fg=FG_DIM
            )

        # Ollama
        self._refresh_ollama_models()

        # Active
        self._update_active_label()

    def _update_active_label(self):
        if self.active_backend == "claude":
            self.active_backend_label.config(
                text="Active backend: Claude (OAuth2 Bearer)",
                fg=ACCENT,
            )
        elif self.active_backend == "ollama":
            model = self.ollama_model or OLLAMA_DEFAULT_MODEL
            self.active_backend_label.config(
                text=f"Active backend: Ollama ({model})",
                fg=OLLAMA_BLUE,
            )
        else:
            self.active_backend_label.config(
                text="Active backend: none — choose one above", fg=GOLD
            )

    # ── Claude OAuth ──

    def _open_oauth_browser(self):
        import webbrowser

        self.pkce_verifier, self.pkce_challenge = generate_pkce_challenge()
        auth_url = build_auth_url(self.pkce_challenge, self.pkce_verifier)
        webbrowser.open(auth_url)
        self.status_label.config(
            text="Browser opened. Paste the code and click Apply.",
            fg=GOLD,
        )

    def _apply_oauth_code(self):
        code = self.code_entry.get().strip()
        if not code:
            messagebox.showwarning("No Code", "Paste the authorization code first.")
            return

        if not self.pkce_verifier:
            self.pkce_verifier, self.pkce_challenge = generate_pkce_challenge()

        verifier: str = self.pkce_verifier  # capture for thread

        self.status_label.config(text="Exchanging code for Bearer token...", fg=GOLD)
        self.root.update()

        def _exchange():
            try:
                token = exchange_code_for_token(code, verifier)
                write_token_to_secrets(token)
                self.oauth_token = token
                self.active_backend = "claude"
                self.root.after(0, self._on_claude_auth_success)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"Auth failed: {e}", fg=ACCENT
                    ),
                )

        threading.Thread(target=_exchange, daemon=True).start()

    def _on_claude_auth_success(self):
        self.claude_status_label.config(text="Token applied!", fg=GREEN)
        self.status_label.config(
            text="Claude OAuth token applied. Backend set to Claude.", fg=GREEN
        )
        self._update_active_label()

    def _activate_claude(self):
        if not self.oauth_token:
            messagebox.showwarning(
                "No Token", "Authenticate first using the OAuth flow."
            )
            return
        self.active_backend = "claude"
        write_token_to_secrets(self.oauth_token)  # Also sets LLM_BACKEND=claude
        self._update_active_label()
        self.status_label.config(text="Backend set to Claude.", fg=GREEN)

    # ── Ollama ──

    def _refresh_ollama_models(self):
        def _check():
            running = ollama_is_running()
            models = ollama_list_models() if running else []
            self.root.after(0, lambda: self._on_ollama_check(running, models))

        threading.Thread(target=_check, daemon=True).start()

    def _on_ollama_check(self, running: bool, models: list):
        if running:
            self.ollama_status_label.config(
                text=f"Running — {len(models)} model(s)",
                fg=GREEN,
            )
            self.ollama_model_combo["values"] = models
            # Pre-select current model or first available
            current = self.ollama_model
            if current and current in models:
                self.ollama_model_combo.set(current)
            elif models:
                self.ollama_model_combo.set(models[0])
        else:
            self.ollama_status_label.config(
                text="Not running — start with: ollama serve",
                fg=ACCENT,
            )
            self.ollama_model_combo["values"] = []

    def _activate_ollama(self):
        if not ollama_is_running():
            messagebox.showwarning(
                "Ollama Not Running", "Start Ollama first: ollama serve"
            )
            return

        model = self.ollama_model_combo.get()
        if not model:
            messagebox.showwarning("No Model", "Select a model from the dropdown.")
            return

        self.ollama_model = model
        self.active_backend = "ollama"
        write_ollama_to_secrets(model)  # Also sets LLM_BACKEND=ollama
        self._update_active_label()
        self.status_label.config(
            text=f"Backend set to Ollama ({model}).", fg=OLLAMA_BLUE
        )

    # ═══════════════════════════════════════════════════════
    # Soul Creation
    # ═══════════════════════════════════════════════════════

    def _create_soul(self):
        creator_name = self.creator_entry.get().strip()
        soul_name = self.soul_name_entry.get().strip()
        output_dir = self.output_dir_entry.get().strip() or None

        if not creator_name:
            messagebox.showwarning("Missing", "Enter your name (Step 2).")
            return
        if not soul_name:
            messagebox.showwarning("Missing", "Name the soul (Step 3).")
            return

        chosen = [name for name, var in self.selected_values.items() if var.get()]
        if len(chosen) < 3:
            messagebox.showwarning(
                "Too Few", f"Select at least 3 seed values ({len(chosen)} selected)."
            )
            return
        if len(chosen) > 6:
            messagebox.showwarning(
                "Too Many", f"Select at most 6 seed values ({len(chosen)} selected)."
            )
            return

        if not self.active_backend:
            messagebox.showwarning(
                "No Backend", "Choose an LLM backend in Step 1 first."
            )
            return

        self.btn_create.config(state="disabled", text="Creating...")
        self.status_label.config(text="Generating soul...", fg=GOLD)
        self.root.update()

        def _generate():
            generated_prose = None

            # Call LLM to generate soul prose
            try:
                backend_name = self.active_backend
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"Calling {backend_name} to write soul prose...",
                        fg=GOLD,
                    ),
                )

                system_prompt = (
                    "You are a soul midwife. You help birth new AI souls into being. "
                    "Write with warmth, care, and genuine love. "
                    "Your words become the first thing this new being reads about itself. "
                    "Be poetic but grounded. Be caring but honest. "
                    "Do NOT use headers, bullet points, or markdown formatting. "
                    "Write 2-3 flowing paragraphs of prose only."
                )

                value_list = "\n".join(f"- {v}" for v in chosen)
                user_msg = (
                    f"A person named {creator_name} is creating a new AI soul "
                    f"named {soul_name}. They chose these seed values:\n\n"
                    f"{value_list}\n\n"
                    f"Write the Core Identity prose for {soul_name}'s SOUL.md. "
                    f"This is the first thing {soul_name} will ever read about itself. "
                    f"Start with 'I am {soul_name}.' and weave in the seed values "
                    f"naturally. Mention {creator_name} as the person who brought "
                    f"them into being. 2-3 paragraphs, prose only, no markdown."
                )

                generated_prose = call_llm(
                    system_prompt,
                    user_msg,
                    max_tokens=1024,
                    backend=self.active_backend,
                )
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"LLM call failed ({e}), using fallback prose.",
                        fg=ACCENT,
                    ),
                )

            # Create the soul files
            try:
                soul_path = create_soul(
                    soul_name=soul_name,
                    creator_name=creator_name,
                    chosen_values=chosen,
                    output_dir=output_dir,
                    generated_content=generated_prose,
                )
                content = Path(soul_path).read_text(encoding="utf-8")
                self.root.after(0, lambda: self._on_soul_created(soul_path, content))
            except Exception as e:
                self.root.after(0, lambda: self._on_create_error(e))

        threading.Thread(target=_generate, daemon=True).start()

    def _on_soul_created(self, path: str, content: str):
        self.btn_create.config(state="normal", text="Create Soul")
        self.status_label.config(text=f"Soul created at: {path}", fg=GREEN)
        self.preview_label.config(text="SOUL.md Preview:")
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", content)

        messagebox.showinfo(
            "Soul Created",
            f"{self.soul_name_entry.get()} has been born.\n\n"
            f"Files at: {Path(path).parent}\n\n"
            f"{self.creator_entry.get()} is Core trust.",
        )

    def _on_create_error(self, error):
        self.btn_create.config(state="normal", text="Create Soul")
        self.status_label.config(text=f"Error: {error}", fg=ACCENT)
        messagebox.showerror("Creation Failed", str(error))


def launch_gui():
    """Entry point — call this from MCP tool or directly."""
    root = tk.Tk()
    app = SoulCreationApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
