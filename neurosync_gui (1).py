"""
NeuroSync EEG Stress Detection GUI
Real-time EEG-based stress monitoring and mindfulness feedback system
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import random
import math
import os
from datetime import datetime
from collections import deque

BG_DARK        = "#0a0a0f"
BG_PANEL       = "#111118"
BG_CARD        = "#16161f"
ACCENT_ORANGE  = "#e8631a"
ACCENT_RED     = "#d63031"
ACCENT_GREEN   = "#00b894"
ACCENT_BLUE    = "#0984e3"
TEXT_PRIMARY   = "#f0f0f5"
TEXT_SECONDARY = "#8888a0"
TEXT_DIM       = "#44445a"
BORDER         = "#222233"
LOW_MAX        = 0.35
MOD_MAX        = 0.65


class EEGSimulator:
    def __init__(self):
        self.channels = ["Fp1", "Fp2", "F3", "F4", "Fz"]
        self.t = 0
        self.stress_level = 0.2
        self.target_stress = 0.2

    def step(self):
        delta = 0.003
        diff = self.target_stress - self.stress_level
        if abs(diff) > delta:
            self.stress_level += delta if diff > 0 else -delta
        else:
            self.stress_level = self.target_stress

    def get_sample(self):
        self.t += 1.0 / 250
        self.step()
        s = self.stress_level
        alpha_amp = 25 * (1 - s)
        beta_amp = 10 * (1 + 2 * s)
        samples = {}
        for i, ch in enumerate(self.channels):
            phase = i * 0.4
            alpha = alpha_amp * math.sin(2 * math.pi * 10 * self.t + phase)
            beta = beta_amp * math.sin(2 * math.pi * 20 * self.t + phase * 1.5)
            noise = random.gauss(0, 5 + s * 8)
            samples[ch] = round(alpha + beta + noise, 3)
        return samples

    def get_band_powers(self):
        s = self.stress_level
        d = {
            "delta": 20,
            "theta": 15 + s * 10,
            "alpha": 35 - s * 20,
            "beta":  15 + s * 25,
            "gamma": 5  + s * 8,
        }
        d = {k: max(v + random.gauss(0, 2), 1) for k, v in d.items()}
        total = sum(d.values())
        return {k: round(v / total * 100, 1) for k, v in d.items()}

    def get_faa(self):
        return round(-0.3 * self.stress_level + random.gauss(0, 0.05), 3)


class NeuroSyncApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroSync - EEG Stress Detection Platform")
        self.configure(bg=BG_DARK)
        self.geometry("1280x800")
        self.minsize(900, 600)

        self.sim = EEGSimulator()
        self.running = False
        self.session_start = None
        self.log_entries = []
        self.stress_history = deque(maxlen=200)
        self.wave_buffers = {ch: deque(maxlen=300) for ch in ["Fp1", "Fp2", "F3", "F4", "Fz"]}
        self.participant = tk.StringVar(value="Participant_01")
        self.intervention_active = False
        self.breath_phase = 0.0
        self.breathing = False

        self._build_ui()
        self.after(100, self._tick)

    def _build_ui(self):
        top = tk.Frame(self, bg=BG_PANEL, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="NeuroSync", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 15, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(top, text="EEG Stress Detection & Mindfulness Platform",
                 bg=BG_PANEL, fg=TEXT_SECONDARY,
                 font=("Courier", 9)).pack(side="left")
        self.clock_lbl = tk.Label(top, text="00:00:00", bg=BG_PANEL,
                                   fg=TEXT_SECONDARY, font=("Courier", 11))
        self.clock_lbl.pack(side="right", padx=16)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=6, pady=6)

        left = tk.Frame(body, bg=BG_PANEL, width=210)
        left.pack(side="left", fill="y", padx=(0, 4))
        left.pack_propagate(False)
        self._build_left(left)

        right = tk.Frame(body, bg=BG_PANEL, width=250)
        right.pack(side="right", fill="y", padx=(4, 0))
        right.pack_propagate(False)
        self._build_right(right)

        center = tk.Frame(body, bg=BG_DARK)
        center.pack(side="left", fill="both", expand=True, padx=4)
        self._build_center(center)

        sb = tk.Frame(self, bg=BG_PANEL, height=26)
        sb.pack(fill="x")
        sb.pack_propagate(False)
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", side="top")
        self.status_var = tk.StringVar(value="Ready. Press START SESSION to begin.")
        tk.Label(sb, textvariable=self.status_var, bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Courier", 8), anchor="w").pack(side="left", padx=10)

    def _build_left(self, p):
        tk.Label(p, text="SESSION CONTROL", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(p, text="Participant ID", bg=BG_PANEL, fg=TEXT_SECONDARY,
                 font=("Courier", 8)).pack(anchor="w", padx=10)
        tk.Entry(p, textvariable=self.participant, bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Courier", 9), insertbackground=TEXT_PRIMARY,
                 relief="flat", bd=4).pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(p, text="Protocol Phase", bg=BG_PANEL, fg=TEXT_SECONDARY,
                 font=("Courier", 8)).pack(anchor="w", padx=10)
        self.phase_combo = ttk.Combobox(
            p, values=["Baseline", "Stress Induction", "Recovery"],
            font=("Courier", 9), state="readonly")
        self.phase_combo.set("Baseline")
        self.phase_combo.pack(fill="x", padx=10, pady=(0, 8))
        for text, cmd, color in [
            ("START SESSION", self._start, ACCENT_ORANGE),
            ("STOP SESSION",  self._stop,  TEXT_DIM),
            ("EXPORT LOG",    self._export, ACCENT_BLUE),
        ]:
            tk.Button(p, text=text, command=cmd, bg=BG_CARD, fg=color,
                      font=("Courier", 8, "bold"), relief="flat", bd=0,
                      padx=6, pady=6, cursor="hand2",
                      activebackground=BORDER,
                      activeforeground=color).pack(fill="x", padx=10, pady=2)
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(p, text="SIMULATION", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(2, 2))
        tk.Label(p, text="Stress Target Level", bg=BG_PANEL, fg=TEXT_SECONDARY,
                 font=("Courier", 8)).pack(anchor="w", padx=10)
        self.sim_slider = tk.Scale(
            p, from_=0, to=100, orient="horizontal",
            bg=BG_PANEL, fg=ACCENT_ORANGE, troughcolor=BG_CARD,
            highlightthickness=0, activebackground=ACCENT_ORANGE,
            font=("Courier", 8), command=self._update_sim)
        self.sim_slider.set(20)
        self.sim_slider.pack(fill="x", padx=10)
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(p, text="ELECTRODE STATUS", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(2, 4))
        self.elec_labels = {}
        for ch in ["Fp1", "Fp2", "F3", "F4", "Fz"]:
            row = tk.Frame(p, bg=BG_PANEL)
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=ch, bg=BG_PANEL, fg=TEXT_PRIMARY,
                     font=("Courier", 9, "bold"), width=4,
                     anchor="w").pack(side="left")
            dot = tk.Label(row, text="*", bg=BG_PANEL, fg=TEXT_DIM,
                           font=("Courier", 10))
            dot.pack(side="left", padx=2)
            imp = tk.Label(row, text="-- kOhm", bg=BG_PANEL, fg=TEXT_DIM,
                           font=("Courier", 8))
            imp.pack(side="left")
            self.elec_labels[ch] = (dot, imp)

    def _build_center(self, p):
        top_row = tk.Frame(p, bg=BG_DARK)
        top_row.pack(fill="x", pady=(0, 4))

        gc = tk.Frame(top_row, bg=BG_CARD)
        gc.pack(side="left", fill="y", padx=(0, 4))
        tk.Label(gc, text="STRESS INDEX", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Courier", 8, "bold")).pack(pady=(8, 2))
        self.gauge_canvas = tk.Canvas(gc, width=200, height=120,
                                       bg=BG_CARD, highlightthickness=0)
        self.gauge_canvas.pack()
        self.stress_lbl = tk.Label(gc, text="LOW", bg=BG_CARD, fg=ACCENT_GREEN,
                                    font=("Courier", 18, "bold"))
        self.stress_lbl.pack(pady=(2, 8))

        bc = tk.Frame(top_row, bg=BG_CARD)
        bc.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(bc, text="SPECTRAL BAND POWER (%)", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.band_canvas = tk.Canvas(bc, height=120, bg=BG_CARD, highlightthickness=0)
        self.band_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        fc = tk.Frame(top_row, bg=BG_CARD)
        fc.pack(side="left", fill="y", padx=(4, 0))
        tk.Label(fc, text="FRONTAL ASYMMETRY", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Courier", 8, "bold")).pack(pady=(8, 2))
        self.faa_canvas = tk.Canvas(fc, width=130, height=120,
                                     bg=BG_CARD, highlightthickness=0)
        self.faa_canvas.pack(padx=4)
        self.faa_lbl = tk.Label(fc, text="FAA: --", bg=BG_CARD,
                                 fg=TEXT_SECONDARY, font=("Courier", 9))
        self.faa_lbl.pack(pady=(2, 8))

        tc = tk.Frame(p, bg=BG_CARD)
        tc.pack(fill="x", pady=4)
        tk.Label(tc, text="STRESS TIMELINE", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.timeline_canvas = tk.Canvas(tc, height=65, bg=BG_CARD, highlightthickness=0)
        self.timeline_canvas.pack(fill="x", padx=8, pady=(0, 6))

        wc = tk.Frame(p, bg=BG_CARD)
        wc.pack(fill="both", expand=True, pady=4)
        tk.Label(wc, text="REAL-TIME EEG WAVEFORMS", bg=BG_CARD, fg=TEXT_SECONDARY,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.wave_canvas = tk.Canvas(wc, bg=BG_CARD, highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    def _build_right(self, p):
        tk.Label(p, text="INTERVENTION MODULE", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(p, text="Guided Breathing", bg=BG_PANEL, fg=TEXT_SECONDARY,
                 font=("Courier", 8)).pack(anchor="w", padx=10, pady=(2, 1))
        self.breath_canvas = tk.Canvas(p, width=210, height=150,
                                        bg=BG_DARK, highlightthickness=0)
        self.breath_canvas.pack(padx=10, pady=4)
        for text, cmd, color in [
            ("Start Breathing", self._start_breath, ACCENT_GREEN),
            ("Stop Breathing",  self._stop_breath,  TEXT_DIM),
        ]:
            tk.Button(p, text=text, command=cmd, bg=BG_CARD, fg=color,
                      font=("Courier", 8, "bold"), relief="flat", bd=0,
                      padx=6, pady=5, cursor="hand2",
                      activebackground=BORDER,
                      activeforeground=color).pack(fill="x", padx=10, pady=2)
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(p, text="ADAPTIVE FEEDBACK", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(2, 2))
        self.feedback = tk.Text(p, height=8, bg=BG_CARD, fg=TEXT_PRIMARY,
                                 font=("Courier", 8), relief="flat", bd=4,
                                 wrap="word", state="disabled")
        self.feedback.pack(fill="x", padx=10, pady=4)
        self._write_feedback("System idle.\n\nStart a session to receive adaptive mindfulness interventions.")
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(p, text="SESSION LOG", bg=BG_PANEL, fg=ACCENT_ORANGE,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(2, 2))
        log_frame = tk.Frame(p, bg=BG_CARD)
        log_frame.pack(fill="both", expand=True, padx=10, pady=4)
        self.log_text = tk.Text(log_frame, bg=BG_CARD, fg=TEXT_DIM,
                                 font=("Courier", 7), relief="flat", bd=4,
                                 wrap="word", state="disabled")
        sb2 = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb2.set)
        sb2.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

    def _write_feedback(self, text):
        self.feedback.config(state="normal")
        self.feedback.delete("1.0", "end")
        self.feedback.insert("end", text)
        self.feedback.config(state="disabled")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = "[" + ts + "] " + msg + "\n"
        self.log_entries.append(line)
        self.log_text.config(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.status_var.set(msg)

    def _start(self):
        if self.running:
            return
        self.running = True
        self.session_start = time.time()
        self._log("Session started -- " + self.participant.get())
        self._update_electrodes()

    def _stop(self):
        self.running = False
        self.breathing = False
        self._log("Session stopped.")

    def _export(self):
        fname = "neurosync_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        path = os.path.join(os.path.expanduser("~"), fname)
        with open(path, "w") as f:
            f.writelines(self.log_entries)
        messagebox.showinfo("Export", "Log saved to:\n" + path)

    def _update_sim(self, val):
        self.sim.target_stress = int(val) / 100.0

    def _start_breath(self):
        self.breathing = True

    def _stop_breath(self):
        self.breathing = False

    def _update_electrodes(self):
        if not self.running:
            return
        for ch, (dot, imp) in self.elec_labels.items():
            v = random.uniform(2, 9)
            col = ACCENT_GREEN if v < 10 else ACCENT_RED
            dot.config(fg=col)
            imp.config(fg=col, text=str(round(v, 1)) + " kOhm")
        self.after(3000, self._update_electrodes)

    def _tick(self):
        if self.running:
            sample = self.sim.get_sample()
            for ch, val in sample.items():
                self.wave_buffers[ch].append(val)
            self.stress_history.append(self.sim.stress_level)
            self._draw_gauge()
            self._draw_bands()
            self._draw_timeline()
            self._draw_waves()
            self._draw_faa()
            self._update_feedback()
            if self.session_start:
                e = int(time.time() - self.session_start)
                h, r = divmod(e, 3600)
                m, s = divmod(r, 60)
                self.clock_lbl.config(
                    text=str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2))
        if self.breathing:
            self.breath_phase += 0.02
            self._draw_breath()
        self.after(40, self._tick)

    def _draw_gauge(self):
        s = self.sim.stress_level
        c = self.gauge_canvas
        c.delete("all")
        w = c.winfo_width() or 200
        h = c.winfo_height() or 120
        cx, cy = w // 2, h - 10
        R = min(cx, cy) - 8
        c.create_arc(cx - R, cy - R, cx + R, cy + R,
                     start=0, extent=180, style="arc", outline=BORDER, width=10)
        col = ACCENT_GREEN if s <= LOW_MAX else (ACCENT_ORANGE if s <= MOD_MAX else ACCENT_RED)
        ext = int(s * 180)
        if ext > 0:
            c.create_arc(cx - R, cy - R, cx + R, cy + R,
                         start=0, extent=ext, style="arc", outline=col, width=10)
        angle = math.radians(s * 180)
        nx = cx + (R - 14) * math.cos(math.pi - angle)
        ny = cy - (R - 14) * math.sin(math.pi - angle)
        c.create_line(cx, cy, nx, ny, fill=TEXT_PRIMARY, width=2)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=TEXT_PRIMARY, outline="")
        c.create_text(cx, cy - R - 6, text=str(round(s, 2)),
                      fill=TEXT_PRIMARY, font=("Courier", 11, "bold"))
        lbl = "LOW" if s <= LOW_MAX else ("MODERATE" if s <= MOD_MAX else "HIGH")
        self.stress_lbl.config(text=lbl, fg=col)

    def _draw_bands(self):
        bands = self.sim.get_band_powers()
        c = self.band_canvas
        c.delete("all")
        w = c.winfo_width() or 300
        h = c.winfo_height() or 120
        names = list(bands.keys())
        vals = list(bands.values())
        n = len(names)
        bw = w // (n + 1)
        pad = bw // 4
        colors = [ACCENT_BLUE, "#a29bfe", ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED]
        for i, (name, val, col) in enumerate(zip(names, vals, colors)):
            x0 = pad + i * bw
            x1 = x0 + bw - pad
            bh = int((val / 100) * (h - 24))
            y1 = h - 4
            y0 = y1 - bh
            c.create_rectangle(x0, y0, x1, y1, fill=col, outline="")
            c.create_text((x0 + x1) // 2, max(y0 - 2, 8),
                          text=str(round(val)),
                          fill=TEXT_SECONDARY, font=("Courier", 7), anchor="s")
            c.create_text((x0 + x1) // 2, y1 + 2,
                          text=name[0].upper() + name[1:],
                          fill=TEXT_DIM, font=("Courier", 7), anchor="n")

    def _draw_timeline(self):
        data = list(self.stress_history)
        c = self.timeline_canvas
        c.delete("all")
        w = c.winfo_width() or 600
        h = c.winfo_height() or 65
        if len(data) < 2:
            return
        for thresh, col in [(LOW_MAX, ACCENT_GREEN), (MOD_MAX, ACCENT_ORANGE)]:
            y = int((1 - thresh) * (h - 10)) + 4
            c.create_line(0, y, w, y, fill=col, dash=(3, 6))
        pts = []
        for i, v in enumerate(data):
            x = int(i * w / len(data))
            y = int((1 - v) * (h - 10)) + 4
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT_ORANGE, width=2, smooth=True)

    def _draw_waves(self):
        channels = ["Fp1", "Fp2", "F3", "F4", "Fz"]
        colors = [ACCENT_ORANGE, "#fd79a8", ACCENT_BLUE, ACCENT_GREEN, "#a29bfe"]
        c = self.wave_canvas
        c.delete("all")
        w = c.winfo_width() or 600
        h = c.winfo_height() or 220
        n = len(channels)
        row_h = h // n
        for i, (ch, col) in enumerate(zip(channels, colors)):
            y_off = i * row_h
            mid = y_off + row_h // 2
            c.create_line(0, mid, w, mid, fill=TEXT_DIM, dash=(2, 8))
            c.create_text(6, y_off + 4, anchor="nw", text=ch,
                          fill=TEXT_SECONDARY, font=("Courier", 8, "bold"))
            data = list(self.wave_buffers[ch])
            if len(data) < 2:
                continue
            mn, mx = min(data), max(data)
            rng = max(mx - mn, 1)
            pts = []
            for j, v in enumerate(data):
                x = int(j * w / len(data))
                y = int(mid - (v - mn - rng / 2) / rng * (row_h * 0.7))
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=col, width=1, smooth=True)

    def _draw_faa(self):
        faa = self.sim.get_faa()
        self.faa_lbl.config(text="FAA: " + ("+" if faa >= 0 else "") + str(round(faa, 3)))
        c = self.faa_canvas
        c.delete("all")
        w = c.winfo_width() or 130
        h = c.winfo_height() or 120
        mx, my = w // 2, h // 2
        c.create_line(10, my, w - 10, my, fill=BORDER)
        c.create_text(14, my + 10, text="R", fill=TEXT_DIM, font=("Courier", 7))
        c.create_text(w - 14, my + 10, text="L", fill=TEXT_DIM, font=("Courier", 7))
        bar = int(faa * (w - 20) * 2)
        col = ACCENT_GREEN if faa >= 0 else ACCENT_RED
        x0, x1 = (mx, mx + bar) if bar > 0 else (mx + bar, mx)
        if abs(bar) > 1:
            c.create_rectangle(x0, my - 8, x1, my + 8, fill=col, outline="")
        c.create_text(mx, my - 22, text="Frontal Alpha Asymmetry",
                      fill=TEXT_DIM, font=("Courier", 7))

    def _draw_breath(self):
        c = self.breath_canvas
        c.delete("all")
        w = c.winfo_width() or 210
        h = c.winfo_height() or 150
        cx, cy = w // 2, h // 2
        r_frac = 0.4 + 0.3 * (1 + math.sin(self.breath_phase * 0.5)) / 2
        r = int(r_frac * (min(cx, cy) - 10))
        for dr in range(15, 0, -3):
            c.create_oval(cx - r - dr, cy - r - dr,
                          cx + r + dr, cy + r + dr,
                          outline=ACCENT_ORANGE, width=1)
        c.create_oval(cx - r, cy - r, cx + r, cy + r,
                      fill="#1a0d00", outline=ACCENT_ORANGE, width=2)
        if r_frac < 0.6:
            label = "Breathe In"
        elif r_frac < 0.75:
            label = "Hold"
        else:
            label = "Breathe Out"
        c.create_text(cx, cy, text=label, fill=TEXT_PRIMARY,
                      font=("Courier", 10, "bold"))

    def _update_feedback(self):
        s = self.sim.stress_level
        phase = self.phase_combo.get()
        if s <= LOW_MAX:
            if self.intervention_active:
                self._log("Stress returned to LOW.")
                self.intervention_active = False
            msg = (
                "OK  LOW STRESS  (" + str(round(s, 2)) + ")\n\n"
                "You are in a good mental state.\n\n"
                "- Keep your current focus\n"
                "- Plan out your study session\n"
                "- Stretch every 45 minutes\n\n"
                "Phase: " + phase
            )
            self._write_feedback(msg)
        elif s <= MOD_MAX:
            if not self.intervention_active:
                self._log("Moderate stress -- breathing exercise triggered.")
                self.intervention_active = True
                self.breathing = True
            msg = (
                "!!  MODERATE STRESS  (" + str(round(s, 2)) + ")\n\n"
                "Guided breathing activated.\n\n"
                "- 4 sec in / 4 hold / 4 out\n"
                "- Close distracting tabs\n"
                "- Hydrate and adjust posture\n\n"
                "Phase: " + phase
            )
            self._write_feedback(msg)
        else:
            if not self.intervention_active:
                self._log("HIGH STRESS -- mandatory break recommended.")
                self.intervention_active = True
                self.breathing = True
            msg = (
                "!!  HIGH STRESS  (" + str(round(s, 2)) + ")\n\n"
                "TAKE A MANDATORY BREAK\n\n"
                "- Step away for 5-10 minutes\n"
                "- Contact a friend or family\n"
                "- Disable notifications\n"
                "- Try 5-4-3-2-1 grounding\n\n"
                "Phase: " + phase
            )
            self._write_feedback(msg)


if __name__ == "__main__":
    app = NeuroSyncApp()
    app.mainloop()
