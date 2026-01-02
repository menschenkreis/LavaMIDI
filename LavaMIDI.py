"""
Lava MIDI Architect v61 - Performance Optimized
- Multithreaded video processing (separate from UI)
- Reduced optical flow cost (ANALYSIS_WIDTH lowered, configurable)
- Numba JIT compilation for grid processing loops
- All v60 features preserved
"""

import cv2
import numpy as np
import mido
import time
import json
import os
import random
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple
from enum import Enum
import tkinter as tk
from tkinter import filedialog, colorchooser
import customtkinter as ctk
from PIL import Image, ImageTk

# Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[WARNING] Numba not installed. Install with: pip install numba --break-system-packages")
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PORT_NAME = "LavaPort"
DEFAULT_SETTINGS_FILE = "settings.json"
MIN_GRID, MAX_GRID = 2, 16

# OPTIMIZATION: Reduced from 160 to 100 for faster optical flow
ANALYSIS_WIDTH_DEFAULT = 100
ANALYSIS_WIDTH_MIN = 60
ANALYSIS_WIDTH_MAX = 200

DEFAULT_GRID_W, DEFAULT_GRID_H = 8, 8
DEFAULT_SENSITIVITY = 50
DEFAULT_MIN_VEL, DEFAULT_MAX_VEL = 40, 127
DEFAULT_ROOT_KEY, DEFAULT_OCTAVE = "C", 3
DEFAULT_NOTE_DURATION_MS = 500
COOLDOWN_TIME = 0.4

VISUAL_RAMP_TIME = 0.05
ZONE_ALPHA = 0.25

# Threading config
FRAME_QUEUE_SIZE = 2  # Small queue to avoid latency buildup


# ══════════════════════════════════════════════════════════════════════════════
# JIT-COMPILED PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True)
def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out: smooth acceleration and deceleration."""
    if t < 0.5:
        return 2.0 * t * t
    else:
        return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


@jit(nopython=True, cache=True)
def process_visual_easing(vis_current: np.ndarray, vis_from: np.ndarray, 
                          vis_to: np.ndarray, vis_start_times: np.ndarray,
                          now: float, ramp_duration: float) -> np.ndarray:
    """JIT-compiled eased visual interpolation."""
    h, w = vis_current.shape
    result = vis_current.copy()
    for y in range(h):
        for x in range(w):
            start_time = vis_start_times[y, x]
            if start_time > 0:
                elapsed = now - start_time
                if elapsed >= ramp_duration:
                    # Animation complete
                    result[y, x] = vis_to[y, x]
                else:
                    # Calculate eased progress
                    progress = elapsed / ramp_duration
                    eased = ease_in_out_quad(progress)
                    from_val = vis_from[y, x]
                    to_val = vis_to[y, x]
                    result[y, x] = from_val + (to_val - from_val) * eased
    return result


@jit(nopython=True, cache=True)
def calculate_trigger_mask(speeds: np.ndarray, timers: np.ndarray, 
                           threshold: float, now: float, 
                           cooldowns: np.ndarray) -> np.ndarray:
    """JIT-compiled trigger detection - returns mask of cells that should trigger."""
    h, w = speeds.shape
    triggers = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if speeds[y, x] > threshold:
                if (now - timers[y, x]) > cooldowns[y, x]:
                    triggers[y, x] = 1
    return triggers


@jit(nopython=True, cache=True)
def blend_overlay_cell(overlay: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       color_b: int, color_g: int, color_r: int, alpha: float):
    """JIT-compiled cell overlay blending."""
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                overlay[y, x, 0] = int(color_b * alpha + overlay[y, x, 0] * (1 - alpha))
                overlay[y, x, 1] = int(color_g * alpha + overlay[y, x, 1] * (1 - alpha))
                overlay[y, x, 2] = int(color_r * alpha + overlay[y, x, 2] * (1 - alpha))


@jit(nopython=True, cache=True)
def apply_zone_overlays_fast(overlay: np.ndarray, vis_current: np.ndarray,
                             zone_colors: np.ndarray, zone_mask: np.ndarray,
                             grid_w: int, grid_h: int, cw: int, ch: int):
    """JIT-compiled batch zone overlay application."""
    fh, fw = overlay.shape[:2]
    for gy in range(grid_h):
        for gx in range(grid_w):
            zone_idx = zone_mask[gy, gx]
            alpha = vis_current[gy, gx]
            
            if zone_idx >= 0 or alpha > 0.01:
                x1 = gx * cw
                y1 = gy * ch
                x2 = min((gx + 1) * cw - 1, fw - 1)
                y2 = min((gy + 1) * ch - 1, fh - 1)
                
                if zone_idx >= 0:
                    color = zone_colors[zone_idx]
                else:
                    color = np.array([255, 255, 255], dtype=np.uint8)
                
                for y in range(y1, y2 + 1):
                    for x in range(x1, x2 + 1):
                        if 0 <= y < fh and 0 <= x < fw:
                            overlay[y, x, 0] = np.uint8(color[0] * alpha + overlay[y, x, 0] * (1 - alpha))
                            overlay[y, x, 1] = np.uint8(color[1] * alpha + overlay[y, x, 1] * (1 - alpha))
                            overlay[y, x, 2] = np.uint8(color[2] * alpha + overlay[y, x, 2] * (1 - alpha))


# ══════════════════════════════════════════════════════════════════════════════
# THEME CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Centralized theme colors."""
    BG_DARK = "#0a0a0a"
    BG_PANEL = "#121212"
    BG_CARD = "#1a1a1a"
    BG_INPUT = "#222222"
    BG_HOVER = "#2a2a2a"
    
    ACCENT = "#7c3aed"
    ACCENT_HOVER = "#8b5cf6"
    ACCENT_DIM = "#5b21b6"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    
    TEXT_PRIMARY = "#fafafa"
    TEXT_SECONDARY = "#a1a1aa"
    TEXT_MUTED = "#71717a"
    
    BORDER = "#262626"
    BORDER_LIGHT = "#404040"
    
    ZONE_COLORS = [
        "#ef4444", "#f97316", "#eab308", "#22c55e",
        "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"
    ]

ctk.set_appearance_mode("Dark")


# ══════════════════════════════════════════════════════════════════════════════
# MUSICAL DATA
# ══════════════════════════════════════════════════════════════════════════════

SCALES = {
    "Minor Pentatonic": [0, 3, 5, 7, 10],
    "Major Pentatonic": [0, 2, 4, 7, 9],
    "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Chromatic": list(range(12)),
    "Blues": [0, 3, 5, 6, 7, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Hirajoshi": [0, 2, 3, 7, 8],
    "Whole Tone": [0, 2, 4, 6, 8, 10],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES = [str(i) for i in range(9)]
DIRECTIONS = ["Left→Right", "Right→Left", "Top→Down", "Bottom→Up", "Spiral", "Random"]


class VelocityCurve(Enum):
    LINEAR = "Linear"
    EXPONENTIAL = "Exponential"
    LOGARITHMIC = "Logarithmic"
    S_CURVE = "S-Curve"


class MotionMode(Enum):
    OPTICAL_FLOW = "Optical Flow"
    FRAME_DIFF = "Frame Diff"
    COLOR_TRACK = "Color Track"


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    """MIDI zone on the grid."""
    name: str
    channel: int
    color_hex: str
    cells: Set[Tuple[int, int]]
    octave_offset: int = 0
    mute: bool = False
    zone_type: str = "Note"
    cc_number: int = 1
    root_override: Optional[str] = None
    octave_override: Optional[int] = None
    scale_override: Optional[str] = None
    mapping_override: Optional[str] = None
    note_low: int = 0
    note_high: int = 127
    vel_min: Optional[int] = None
    vel_max: Optional[int] = None
    use_random_velocity: bool = False
    use_probability: bool = False
    probability: int = 100
    use_smoothing: bool = False
    smoothing: int = 30
    duration: int = DEFAULT_NOTE_DURATION_MS
    use_custom_color: bool = False
    track_hsv: Tuple[int, int, int] = (0, 0, 0)
    last_cc_value: int = -1

    def __post_init__(self):
        self.cells = set(tuple(c) for c in self.cells)
        self.color_bgr = self._hex_to_bgr(self.color_hex)

    def _hex_to_bgr(self, hex_col: str) -> Tuple[int, int, int]:
        hex_col = hex_col.lstrip('#')
        rgb = tuple(int(hex_col[i:i + 2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])

    def update_color(self, hex_col: str):
        self.color_hex = hex_col
        self.color_bgr = self._hex_to_bgr(hex_col)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "channel": self.channel, "color": self.color_hex,
            "cells": list(self.cells), "octave_offset": self.octave_offset, "mute": self.mute,
            "zone_type": self.zone_type, "cc_number": self.cc_number,
            "root_override": self.root_override, "octave_override": self.octave_override,
            "scale_override": self.scale_override, "mapping_override": self.mapping_override,
            "note_low": self.note_low, "note_high": self.note_high,
            "vel_min": self.vel_min, "vel_max": self.vel_max,
            "use_random_velocity": self.use_random_velocity,
            "use_probability": self.use_probability, "probability": self.probability,
            "use_smoothing": self.use_smoothing, "smoothing": self.smoothing,
            "duration": self.duration,
            "use_custom_color": self.use_custom_color, "track_hsv": self.track_hsv
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Zone':
        return cls(
            name=data["name"], channel=data["channel"], color_hex=data["color"],
            cells=set(tuple(c) for c in data["cells"]),
            octave_offset=data.get("octave_offset", 0), mute=data.get("mute", False),
            zone_type=data.get("zone_type", "Note"), cc_number=data.get("cc_number", 1),
            root_override=data.get("root_override"), octave_override=data.get("octave_override"),
            scale_override=data.get("scale_override"), mapping_override=data.get("mapping_override"),
            note_low=data.get("note_low", 0), note_high=data.get("note_high", 127),
            vel_min=data.get("vel_min"), vel_max=data.get("vel_max"),
            use_random_velocity=data.get("use_random_velocity", False),
            use_probability=data.get("use_probability", False),
            probability=data.get("probability", 100),
            use_smoothing=data.get("use_smoothing", False),
            smoothing=data.get("smoothing", 30),
            duration=data.get("duration", DEFAULT_NOTE_DURATION_MS),
            use_custom_color=data.get("use_custom_color", False),
            track_hsv=tuple(data.get("track_hsv", (0, 0, 0)))
        )


@dataclass
class Drone:
    """A continuous or looping background note."""
    active: bool = False
    note: str = "C"
    octave: int = 3
    velocity: int = 100
    channel: int = 1
    mode: str = "Loop"
    duration: float = 1.0
    pause: float = 1.0
    state: str = "IDLE"
    last_action_time: float = 0.0
    current_pitch: int = -1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ['state', 'last_action_time', 'current_pitch']}

    @classmethod
    def from_dict(cls, data: dict) -> 'Drone':
        return cls(**data)


@dataclass
class AppState:
    """Application state."""
    grid_w: int = DEFAULT_GRID_W
    grid_h: int = DEFAULT_GRID_H
    sensitivity: int = DEFAULT_SENSITIVITY
    root_key: str = DEFAULT_ROOT_KEY
    octave: int = DEFAULT_OCTAVE
    scale_name: str = "Minor Pentatonic"
    mapping_direction: str = "Left→Right"
    min_velocity: int = DEFAULT_MIN_VEL
    max_velocity: int = DEFAULT_MAX_VEL
    velocity_curve: str = VelocityCurve.LINEAR.value
    brightness: float = 0.0
    contrast: float = 1.0
    pixelate_view: bool = False
    video_blur_amount: int = 0
    show_grid: bool = True
    show_notes: bool = True
    motion_mode: str = MotionMode.OPTICAL_FLOW.value
    track_color_hsv: Tuple[int, int, int] = (15, 200, 200)
    motion_smoothing: float = 0.3
    motion_gain: float = 1.0  # NEW: Amplify motion/flow vectors
    video_source_type: str = "camera"
    video_file_path: str = ""
    log_midi_signals: bool = False
    manual_trigger_mode: bool = False
    manual_trigger_velocity: int = 64
    midi_active: bool = False
    show_flow_vectors: bool = False
    midi_port_name: str = PORT_NAME
    
    # OPTIMIZATION: Configurable analysis width
    analysis_width: int = ANALYSIS_WIDTH_DEFAULT

    drones: List[Drone] = field(default_factory=lambda: [Drone() for _ in range(6)])

    # Layout: 1/6 left, 4/6 video, 2/6 right (for 1400px window)
    layout_sash_h1: int = 233   # 1400 * 1/6
    layout_sash_h2: int = 1166  # 1400 * 5/6 (left + video)
    layout_sash_v: int = 650

    def to_dict(self) -> dict:
        d = {k: list(v) if isinstance(v, tuple) else v
             for k, v in self.__dict__.items() if k != 'midi_active' and k != 'drones'}
        d['drones'] = [drone.to_dict() for drone in self.drones]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'AppState':
        state = cls()
        if 'drones' in data:
            state.drones = [Drone.from_dict(d) for d in data['drones']]
            while len(state.drones) < 6:
                state.drones.append(Drone())
            state.drones = state.drones[:6]

        for k, v in data.items():
            if hasattr(state, k) and k != 'drones':
                setattr(state, k, tuple(v) if k == 'track_color_hsv' else v)
        return state


# ══════════════════════════════════════════════════════════════════════════════
# THREADED VIDEO PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class VideoProcessor(threading.Thread):
    """
    OPTIMIZATION: Separate thread for video capture and motion analysis.
    Decouples heavy OpenCV processing from the UI thread.
    """
    
    def __init__(self, app_state: AppState, frame_queue: queue.Queue, 
                 result_queue: queue.Queue, log_callback=None):
        super().__init__(daemon=True)
        self.app_state = app_state
        self.frame_queue = frame_queue  # Raw frames go here
        self.result_queue = result_queue  # Processed results go here
        self.log = log_callback or print
        
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.analyzer: Optional['MotionAnalyzer'] = None
        
        self._lock = threading.Lock()
        self._video_source_changed = threading.Event()
        
    def set_analyzer(self, analyzer: 'MotionAnalyzer'):
        with self._lock:
            self.analyzer = analyzer
    
    def init_video(self, source_type: str, file_path: str = ""):
        """Thread-safe video source initialization."""
        with self._lock:
            if self.cap:
                self.cap.release()
            
            if source_type == "file" and file_path:
                self.cap = cv2.VideoCapture(file_path)
                self.log(f"Video: {os.path.basename(file_path)}", "info")
            else:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self.log("Video: Webcam", "info")
            
            if not self.cap or not self.cap.isOpened():
                self.log("Failed to open video", "error")
    
    def run(self):
        """Main processing loop - runs in separate thread."""
        self.running = True
        
        while self.running:
            with self._lock:
                cap = self.cap
                analyzer = self.analyzer
            
            if not cap or not cap.isOpened():
                time.sleep(0.05)
                continue
            
            ret, frame = cap.read()
            if not ret:
                # Loop video files
                if self.app_state.video_source_type == "file":
                    with self._lock:
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                            if not ret:
                                time.sleep(0.01)
                                continue
                else:
                    time.sleep(0.01)
                    continue
            
            frame = cv2.flip(frame, 1)
            
            # Apply brightness/contrast
            if self.app_state.brightness != 0 or self.app_state.contrast != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=self.app_state.contrast, 
                                           beta=self.app_state.brightness)
            
            # Perform motion analysis
            speeds = None
            if analyzer:
                mode = MotionMode(self.app_state.motion_mode)
                speeds = analyzer.analyze(
                    frame, mode, 
                    self.app_state.motion_smoothing,
                    self.app_state.track_color_hsv
                )
            
            # Put result in queue (non-blocking, drop old frames if queue full)
            result = {
                'frame': frame,
                'speeds': speeds,
                'timestamp': time.time()
            }
            
            try:
                # Clear old frames to prevent latency buildup
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop frame if queue is full
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None


# ══════════════════════════════════════════════════════════════════════════════
# CORE COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class MIDIManager:
    """MIDI I/O handler."""

    def __init__(self, target_port: str, log_callback=None):
        self.port_name = target_port
        self.output: Optional[mido.ports.BaseOutput] = None
        self.active_notes: List[dict] = []
        self.log = log_callback or print
        self._lock = threading.Lock()
        self._connect()

    def get_available_ports(self) -> List[str]:
        try:
            return mido.get_output_names()
        except:
            return []

    def set_port(self, port_name: str):
        if port_name == self.port_name and self.output is not None:
            return
        self.close()
        self.port_name = port_name
        self._connect()

    def _connect(self):
        try:
            outputs = self.get_available_ports()
            target = next((n for n in outputs if n == self.port_name), None)
            if not target:
                target = next((n for n in outputs if self.port_name in n), None)
            if not target and outputs:
                target = outputs[0]

            if target:
                self.output = mido.open_output(target)
                self.port_name = target
                self.log(f"MIDI → {target}", "success")
            else:
                self.log("MIDI: No ports", "warning")
        except Exception as e:
            self.log(f"MIDI Error: {e}", "error")

    def send_note_on(self, note: int, velocity: int, channel: int, duration: float = 0.5):
        with self._lock:
            if self.output:
                self.output.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
                self.active_notes.append({
                    'note': note,
                    'channel': channel,
                    'off_time': time.time() + duration,
                    'name': f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"
                })

    def send_raw_note(self, note: int, velocity: int, channel: int, on: bool):
        with self._lock:
            if self.output:
                msg_type = 'note_on' if on else 'note_off'
                self.output.send(mido.Message(msg_type, note=note, velocity=velocity, channel=channel))

    def send_cc(self, cc: int, value: int, channel: int):
        with self._lock:
            if self.output:
                self.output.send(mido.Message('control_change', control=cc, value=value, channel=channel))

    def process_note_offs(self, log_enabled: bool = False):
        with self._lock:
            if not self.output:
                return
            now = time.time()
            remaining = []
            for n in self.active_notes:
                if now >= n['off_time']:
                    self.output.send(mido.Message('note_off', note=n['note'], velocity=0, channel=n['channel']))
                    if log_enabled:
                        self.log(f"  ↳ Off: {n['name']}", "midi")
                else:
                    remaining.append(n)
            self.active_notes = remaining

    def panic(self):
        with self._lock:
            if self.output:
                for ch in range(16):
                    self.output.send(mido.Message('control_change', channel=ch, control=123, value=0))
            self.active_notes.clear()

    def close(self):
        self.panic()
        with self._lock:
            if self.output:
                self.output.close()


class MotionAnalyzer:
    """Motion detection engine with configurable analysis width."""

    def __init__(self, grid_w: int, grid_h: int, analysis_width: int = ANALYSIS_WIDTH_DEFAULT):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.analysis_width = analysis_width
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.smoothed: Optional[np.ndarray] = None
        self.last_flow: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def set_analysis_width(self, width: int):
        """OPTIMIZATION: Adjustable analysis resolution."""
        with self._lock:
            self.analysis_width = max(ANALYSIS_WIDTH_MIN, min(ANALYSIS_WIDTH_MAX, width))

    def resize(self, grid_w: int, grid_h: int):
        with self._lock:
            self.grid_w, self.grid_h = grid_w, grid_h
            self.prev_gray = self.prev_frame = self.smoothed = self.last_flow = None

    def analyze(self, frame: np.ndarray, mode: MotionMode, smoothing: float = 0.3,
                track_hsv: Tuple[int, int, int] = None) -> np.ndarray:
        with self._lock:
            h, w = frame.shape[:2]
            flow_w = self.analysis_width
            flow_h = max(1, int(self.analysis_width * h / w))
            small = cv2.resize(frame, (flow_w, flow_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if mode == MotionMode.OPTICAL_FLOW:
                speeds = self._optical_flow(gray)
            elif mode == MotionMode.FRAME_DIFF:
                speeds = self._frame_diff(gray)
            elif mode == MotionMode.COLOR_TRACK:
                speeds = self._color_track(small, track_hsv or (15, 200, 200))
            else:
                speeds = np.zeros((self.grid_h, self.grid_w))

            self.prev_gray, self.prev_frame = gray, small.copy()

            if self.smoothed is None or self.smoothed.shape != speeds.shape:
                self.smoothed = speeds
            else:
                self.smoothed = smoothing * self.smoothed + (1 - smoothing) * speeds
            return self.smoothed.copy()

    def calculate_cell_color_motion(self, frame: np.ndarray, x: int, y: int,
                                    target_hsv: Tuple[int, int, int]) -> float:
        with self._lock:
            if self.prev_frame is None:
                return 0.0
            h, w = self.prev_frame.shape[:2]
            small_curr = cv2.resize(frame, (w, h))
            cw, ch = w / self.grid_w, h / self.grid_h
            x1, y1 = int(x * cw), int(y * ch)
            x2, y2 = int((x + 1) * cw), int((y + 1) * ch)
            if x1 >= w or y1 >= h:
                return 0.0
            x2, y2 = min(x2, w), min(y2, h)
            roi_curr = small_curr[y1:y2, x1:x2]
            roi_prev = self.prev_frame[y1:y2, x1:x2]
            if roi_curr.size == 0 or roi_prev.size == 0:
                return 0.0
            curr_hsv = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2HSV)
            prev_hsv = cv2.cvtColor(roi_prev, cv2.COLOR_BGR2HSV)
            h_t, s_t, v_t = target_hsv
            lower = np.array([max(0, h_t - 15), max(0, s_t - 50), max(0, v_t - 50)])
            upper = np.array([min(179, h_t + 15), min(255, s_t + 50), min(255, v_t + 50)])
            mask_curr = cv2.inRange(curr_hsv, lower, upper)
            mask_prev = cv2.inRange(prev_hsv, lower, upper)
            diff = cv2.absdiff(mask_curr, mask_prev)
            return np.mean(diff) / 255.0 * 8.0

    def _optical_flow(self, gray: np.ndarray) -> np.ndarray:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.last_flow = None
            return np.zeros((self.grid_h, self.grid_w))
        # OPTIMIZATION: Reduced pyramid levels and iterations for speed
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            pyr_scale=0.5, levels=2, winsize=12,  # Reduced from levels=3, winsize=15
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        self.last_flow = flow
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return cv2.resize(mag, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)

    def _frame_diff(self, gray: np.ndarray) -> np.ndarray:
        """OPTIMIZATION: Frame diff is ~100x faster than optical flow."""
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return np.zeros((self.grid_h, self.grid_w))
        diff = cv2.GaussianBlur(cv2.absdiff(self.prev_gray, gray), (5, 5), 0)
        return cv2.resize(diff.astype(float) / 255.0 * 10, (self.grid_w, self.grid_h))

    def _color_track(self, frame: np.ndarray, hsv_center: Tuple[int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_center
        mask = cv2.inRange(hsv,
                           np.array([max(0, h - 15), max(0, s - 60), max(0, v - 60)]),
                           np.array([min(179, h + 15), min(255, s + 60), min(255, v + 60)]))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        if self.prev_frame is not None:
            prev_hsv = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2HSV)
            prev_mask = cv2.inRange(prev_hsv,
                                    np.array([max(0, h - 15), max(0, s - 60), max(0, v - 60)]),
                                    np.array([min(179, h + 15), min(255, s + 60), min(255, v + 60)]))
            diff = cv2.absdiff(mask, prev_mask)
        else:
            diff = mask
        return cv2.resize(diff.astype(float) / 255.0 * 8, (self.grid_w, self.grid_h))

    def draw_flow_vectors(self, overlay: np.ndarray, scale: float = 3.0) -> None:
        with self._lock:
            if self.last_flow is None:
                return
            fh, fw = overlay.shape[:2]
            flow_h, flow_w = self.last_flow.shape[:2]
            step_x = max(1, flow_w // 16)
            step_y = max(1, flow_h // 16)
            for fy in range(0, flow_h, step_y):
                for fx in range(0, flow_w, step_x):
                    dx, dy = self.last_flow[fy, fx]
                    mag = np.sqrt(dx * dx + dy * dy)
                    if mag > 0.5:
                        px = int(fx / flow_w * fw)
                        py = int(fy / flow_h * fh)
                        end_x = int(px + dx * scale)
                        end_y = int(py + dy * scale)
                        cv2.arrowedLine(overlay, (px, py), (end_x, end_y),
                                        (0, 255, 100), 1, cv2.LINE_AA, tipLength=0.3)


class PitchCalculator:
    """MIDI pitch calculation."""

    def __init__(self, grid_w: int, grid_h: int):
        self.grid_w, self.grid_h = grid_w, grid_h
        self._random_map: Optional[np.ndarray] = None

    def resize(self, grid_w: int, grid_h: int):
        self.grid_w, self.grid_h = grid_w, grid_h
        self._random_map = None

    def calculate(self, x: int, y: int, root_midi: int, scale: List[int],
                  direction: str, octave_offset: int = 0) -> int:
        scale_len = len(scale)
        dir_map = {
            "Left→Right": x + y * scale_len,
            "Right→Left": (self.grid_w - 1 - x) + y * scale_len,
            "Top→Down": y + x * scale_len,
            "Bottom→Up": (self.grid_h - 1 - y) + x * scale_len,
            "Spiral": self._spiral_idx(x, y),
            "Random": self._random_idx(x, y),
        }
        idx = dir_map.get(direction, x + y * scale_len)
        note = root_midi + scale[idx % scale_len] + (idx // scale_len + octave_offset) * 12
        return int(np.clip(note, 0, 127))

    def _spiral_idx(self, x: int, y: int) -> int:
        cx, cy = self.grid_w / 2, self.grid_h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return int(dist / np.sqrt(cx ** 2 + cy ** 2) * self.grid_w * self.grid_h)

    def _random_idx(self, x: int, y: int) -> int:
        if self._random_map is None or self._random_map.shape != (self.grid_h, self.grid_w):
            np.random.seed(42)
            self._random_map = np.random.permutation(self.grid_w * self.grid_h).reshape(self.grid_h, self.grid_w)
        return int(self._random_map[y, x])


class VelocityMapper:
    @staticmethod
    def map(intensity: float, threshold: float, min_vel: int, max_vel: int,
            curve: VelocityCurve) -> int:
        norm = min(1.0, max(0, intensity - threshold) / 5.0)
        if curve == VelocityCurve.EXPONENTIAL:
            norm = norm ** 2
        elif curve == VelocityCurve.LOGARITHMIC:
            norm = np.log1p(norm * 9) / np.log(10)
        elif curve == VelocityCurve.S_CURVE:
            norm = 1 / (1 + np.exp(-10 * (norm - 0.5)))
        return int(np.clip(min_vel + (max_vel - min_vel) * norm, 0, 127))

    @staticmethod
    def random_between(min_vel: int, max_vel: int) -> int:
        return random.randint(min(min_vel, max_vel), max(min_vel, max_vel))


# ══════════════════════════════════════════════════════════════════════════════
# UI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class ModernSlider(ctk.CTkFrame):
    def __init__(self, parent, label: str, from_: float, to: float, default: float,
                 command=None, format_str="{:.0f}", **kwargs):
        super().__init__(parent, fg_color="transparent")
        self.format_str = format_str
        self.command = command
        self.from_ = from_
        self.to = to

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x")

        ctk.CTkLabel(header, text=label, font=("Inter", 10),
                     text_color=Theme.TEXT_SECONDARY).pack(side="left")

        self.entry_var = ctk.StringVar(value=format_str.format(default))
        self.entry = ctk.CTkEntry(header, textvariable=self.entry_var, width=40, height=18,
                                  font=("Inter", 10, "bold"), fg_color=Theme.BG_INPUT,
                                  border_width=0, justify="right")
        self.entry.pack(side="right")
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        self.slider = ctk.CTkSlider(self, from_=from_, to=to, fg_color=Theme.BG_INPUT,
                                    progress_color=Theme.ACCENT, button_color=Theme.TEXT_PRIMARY,
                                    button_hover_color=Theme.ACCENT_HOVER, height=14,
                                    command=self._on_slide, **kwargs)
        self.slider.set(default)
        self.slider.pack(fill="x", pady=(2, 0))

    def _on_slide(self, value):
        self.entry_var.set(self.format_str.format(value))
        if self.command:
            self.command(value)

    def _on_entry(self, event=None):
        try:
            val = float(self.entry_var.get())
            val = max(self.from_, min(self.to, val))
            self.slider.set(val)
            self.entry_var.set(self.format_str.format(val))
            if self.command:
                self.command(val)
        except ValueError:
            pass

    def set(self, value):
        self.slider.set(value)
        self.entry_var.set(self.format_str.format(value))

    def get(self):
        return self.slider.get()


class ModernDropdown(ctk.CTkFrame):
    def __init__(self, parent, label: str, values: list, default: str, command=None, **kwargs):
        super().__init__(parent, fg_color="transparent")
        if label:
            ctk.CTkLabel(self, text=label, font=("Inter", 10),
                         text_color=Theme.TEXT_SECONDARY).pack(anchor="w")
        self.var = ctk.StringVar(value=default)
        self.menu = ctk.CTkOptionMenu(self, values=values, variable=self.var, fg_color=Theme.BG_INPUT,
                                      button_color=Theme.BG_HOVER, button_hover_color=Theme.BORDER_LIGHT,
                                      dropdown_fg_color=Theme.BG_CARD, dropdown_hover_color=Theme.ACCENT,
                                      command=command, height=26, **kwargs)
        self.menu.pack(fill="x", pady=(2, 0))

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def configure(self, **kwargs):
        self.menu.configure(**kwargs)


class CollapsibleSection(ctk.CTkFrame):
    def __init__(self, parent, title: str, icon: str = "▼", expanded: bool = True):
        super().__init__(parent, fg_color=Theme.BG_CARD, corner_radius=6)
        self.expanded = expanded
        self.icon_open, self.icon_closed = "▾", "▸"

        header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        header.pack(fill="x", padx=10, pady=(8, 0))
        header.bind("<Button-1>", self._toggle)

        self.icon_label = ctk.CTkLabel(header, text=self.icon_open if expanded else self.icon_closed,
                                       font=("Inter", 11), text_color=Theme.TEXT_MUTED, width=14)
        self.icon_label.pack(side="left")
        self.icon_label.bind("<Button-1>", self._toggle)

        self.title_label = ctk.CTkLabel(header, text=title, font=("Inter", 11, "bold"),
                                        text_color=Theme.TEXT_PRIMARY)
        self.title_label.pack(side="left", padx=(2, 0))
        self.title_label.bind("<Button-1>", self._toggle)

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=10, pady=(6, 10))

    def _toggle(self, event=None):
        self.expanded = not self.expanded
        self.icon_label.configure(text=self.icon_open if self.expanded else self.icon_closed)
        if self.expanded:
            self.content.pack(fill="x", padx=10, pady=(6, 10))
        else:
            self.content.pack_forget()


class DroneCard(ctk.CTkFrame):
    """Compact drone card."""

    def __init__(self, parent, drone: 'Drone', index: int, on_update=None):
        super().__init__(parent, fg_color=Theme.BG_CARD, corner_radius=4,
                         border_width=1, border_color=Theme.BORDER)
        self.drone = drone
        self.on_update = on_update

        r1 = ctk.CTkFrame(self, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=4)

        self.led = ctk.CTkLabel(r1, text="●", font=("Inter", 12), width=14,
                                text_color=Theme.TEXT_MUTED)
        self.led.pack(side="left")
        self.after(100, self._update_led)

        self.active_var = ctk.BooleanVar(value=drone.active)
        ctk.CTkCheckBox(r1, text=f"D{index + 1}", variable=self.active_var, font=("Inter", 10, "bold"),
                        width=40, checkbox_width=16, checkbox_height=16,
                        command=self._on_change).pack(side="left")

        self.mode_var = ctk.StringVar(value=drone.mode)
        ctk.CTkSegmentedButton(r1, values=["Loop", "Hold"], variable=self.mode_var, width=70, height=20,
                               font=("Inter", 9), command=self._on_change).pack(side="right")

        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=(0, 4))

        self.note_var = ctk.StringVar(value=drone.note)
        ctk.CTkOptionMenu(r2, values=NOTE_NAMES, variable=self.note_var, width=42, height=18,
                          font=("Inter", 9), command=self._on_change).pack(side="left", padx=(0, 2))

        self.oct_var = ctk.StringVar(value=str(drone.octave))
        ctk.CTkOptionMenu(r2, values=[str(i) for i in range(9)], variable=self.oct_var, width=32,
                          height=18, font=("Inter", 9), command=self._on_change).pack(side="left", padx=(0, 4))

        ctk.CTkLabel(r2, text="v", font=("Inter", 8), text_color=Theme.TEXT_MUTED).pack(side="left")
        self.vel_var = ctk.StringVar(value=str(drone.velocity))
        ctk.CTkEntry(r2, textvariable=self.vel_var, width=28, height=18, font=("Inter", 9),
                     justify="center").pack(side="left", padx=2)
        self.vel_var.trace("w", self._on_change)

        ctk.CTkLabel(r2, text="ch", font=("Inter", 8), text_color=Theme.TEXT_MUTED).pack(side="left", padx=(4, 0))
        self.ch_var = ctk.StringVar(value=str(drone.channel))
        ctk.CTkOptionMenu(r2, values=[str(i) for i in range(1, 17)], variable=self.ch_var, width=36,
                          height=18, font=("Inter", 9), command=self._on_change).pack(side="left", padx=2)

        self.r3 = ctk.CTkFrame(self, fg_color="transparent")
        self.r3.pack(fill="x", padx=6, pady=(0, 4))

        ctk.CTkLabel(self.r3, text="dur", font=("Inter", 8), text_color=Theme.TEXT_MUTED).pack(side="left")
        self.dur_var = ctk.StringVar(value=str(drone.duration))
        ctk.CTkEntry(self.r3, textvariable=self.dur_var, width=32, height=18, font=("Inter", 9)).pack(side="left", padx=2)
        self.dur_var.trace("w", self._on_change)

        ctk.CTkLabel(self.r3, text="pause", font=("Inter", 8), text_color=Theme.TEXT_MUTED).pack(side="left", padx=(6, 0))
        self.pause_var = ctk.StringVar(value=str(drone.pause))
        ctk.CTkEntry(self.r3, textvariable=self.pause_var, width=32, height=18, font=("Inter", 9)).pack(side="left", padx=2)
        self.pause_var.trace("w", self._on_change)

        self._update_visibility()

    def _update_visibility(self):
        if self.mode_var.get() == "Hold":
            self.r3.pack_forget()
        else:
            self.r3.pack(fill="x", padx=6, pady=(0, 4))

    def _update_led(self):
        if self.drone.active and self.drone.state == "PLAYING":
            self.led.configure(text_color=Theme.SUCCESS)
        else:
            self.led.configure(text_color=Theme.TEXT_MUTED)
        self.after(200, self._update_led)

    def _on_change(self, *args):
        self._update_visibility()
        try:
            self.drone.active = self.active_var.get()
            self.drone.note = self.note_var.get()
            self.drone.octave = int(self.oct_var.get())
            self.drone.velocity = int(self.vel_var.get() or 0)
            self.drone.channel = int(self.ch_var.get())
            self.drone.mode = self.mode_var.get()
            self.drone.duration = float(self.dur_var.get() or 0)
            self.drone.pause = float(self.pause_var.get() or 0)
        except ValueError:
            pass


class CollapsibleZoneCard(ctk.CTkFrame):
    """Zone card with embedded settings and Set Cells button."""

    def __init__(self, parent, zone: 'Zone', on_delete=None, on_update=None,
                 on_pick=None, on_set_cells=None):
        super().__init__(parent, fg_color=Theme.BG_CARD, corner_radius=6,
                         border_width=1, border_color=Theme.BORDER)
        self.zone = zone
        self.on_delete = on_delete
        self.on_update = on_update
        self.on_pick = on_pick
        self.on_set_cells = on_set_cells
        self.expanded = False

        header = ctk.CTkFrame(self, fg_color="transparent", height=30, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        self.color_bar = ctk.CTkButton(header, text="", width=6, height=30, fg_color=zone.color_hex,
                                       hover_color=zone.color_hex, corner_radius=0, command=self._pick_color)
        self.color_bar.pack(side="left", fill="y")

        self.arrow = ctk.CTkLabel(header, text="▸", font=("Inter", 10), text_color=Theme.TEXT_MUTED, width=20)
        self.arrow.pack(side="left")
        self.arrow.bind("<Button-1>", self._toggle)

        self.mode_btn = ctk.CTkButton(header, text=zone.zone_type, width=32, height=18,
                                      font=("Inter", 8, "bold"), fg_color=Theme.BG_INPUT,
                                      hover_color=Theme.ACCENT, command=self._toggle_mode)
        self.mode_btn.pack(side="left", padx=2)

        self.name_var = ctk.StringVar(value=zone.name)
        self.name_var.trace("w", self._on_name_change)
        self.name_entry = ctk.CTkEntry(header, textvariable=self.name_var, height=22, width=70,
                                       font=("Inter", 11, "bold"), fg_color="transparent",
                                       border_width=0, text_color=Theme.TEXT_PRIMARY)
        self.name_entry.pack(side="left", fill="x", expand=True)

        self.cell_count = ctk.CTkLabel(header, text=f"{len(zone.cells)}c",
                                       font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.cell_count.pack(side="right", padx=4)

        self.mute_btn = ctk.CTkButton(header, text="M", width=22, height=18, font=("Inter", 9, "bold"),
                                      fg_color=Theme.DANGER if zone.mute else Theme.BG_INPUT,
                                      hover_color=Theme.DANGER, command=self._toggle_mute)
        self.mute_btn.pack(side="right", padx=4)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.tabs = ctk.CTkTabview(self.body, height=360, fg_color=Theme.BG_INPUT,
                                   corner_radius=4, border_width=1, border_color=Theme.BORDER_LIGHT,
                                   segmented_button_fg_color=Theme.BG_DARK,
                                   segmented_button_selected_color=Theme.ACCENT)
        self.tabs.pack(fill="x", padx=4, pady=4)

        self.tab_music = self.tabs.add("Music")
        self.tab_dyn = self.tabs.add("Dynamics")
        self.tab_track = self.tabs.add("Track")

        self._rebuild_tabs()

        btn_row = ctk.CTkFrame(self.body, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkButton(btn_row, text="⬚ Set Cells", height=22, width=80, font=("Inter", 10),
                      fg_color=Theme.BG_HOVER, hover_color=Theme.ACCENT,
                      command=self._set_cells).pack(side="left", padx=(0, 4))

        ctk.CTkButton(btn_row, text="✕ Delete", height=22, fg_color=Theme.BG_INPUT,
                      hover_color=Theme.DANGER, text_color="#ff8888", font=("Inter", 10),
                      command=self._delete).pack(side="right")

    def _set_cells(self):
        if self.on_set_cells:
            self.on_set_cells(self.zone, self)

    def update_cell_count(self):
        self.cell_count.configure(text=f"{len(self.zone.cells)}c")

    def _toggle_mode(self):
        self.zone.zone_type = "CC" if self.zone.zone_type == "Note" else "Note"
        self.mode_btn.configure(text=self.zone.zone_type)
        self._rebuild_tabs()
        if self.on_update:
            self.on_update()

    def _rebuild_tabs(self):
        for w in self.tab_music.winfo_children():
            w.destroy()
        for w in self.tab_dyn.winfo_children():
            w.destroy()

        if self.zone.zone_type == "Note":
            self._build_music_tab_note(self.tab_music)
            self._build_dynamics_tab_note(self.tab_dyn)
        else:
            self._build_music_tab_cc(self.tab_music)
            self._build_dynamics_tab_cc(self.tab_dyn)

        for w in self.tab_track.winfo_children():
            w.destroy()
        self._build_tracking_tab(self.tab_track)

    def _build_music_tab_note(self, parent):
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(row1, text="Channel", font=("Inter", 10)).pack(side="left")
        self.ch_menu = ctk.CTkOptionMenu(row1, values=[str(i) for i in range(1, 17)], width=50, height=20,
                                         font=("Inter", 10), command=self._on_channel_change)
        self.ch_menu.set(str(self.zone.channel))
        self.ch_menu.pack(side="right")

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 4))
        self.use_custom_root = ctk.BooleanVar(value=self.zone.root_override is not None)
        ctk.CTkCheckBox(row2, text="Root", variable=self.use_custom_root, width=50, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_overrides).pack(side="left")
        self.root_var = ctk.StringVar(value=self.zone.root_override or "C")
        self.root_menu = ctk.CTkOptionMenu(row2, values=NOTE_NAMES, variable=self.root_var, width=50, height=20,
                                           state="disabled", command=self._update_overrides)
        self.root_menu.pack(side="right")

        row3 = ctk.CTkFrame(parent, fg_color="transparent")
        row3.pack(fill="x", pady=(0, 4))
        self.use_custom_octave = ctk.BooleanVar(value=self.zone.octave_override is not None)
        ctk.CTkCheckBox(row3, text="Octave", variable=self.use_custom_octave, width=50, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_overrides).pack(side="left")
        self.octave_var = ctk.StringVar(value=str(self.zone.octave_override if self.zone.octave_override is not None else 3))
        self.octave_menu = ctk.CTkOptionMenu(row3, values=OCTAVES, variable=self.octave_var, width=50, height=20,
                                             state="disabled", command=self._update_overrides)
        self.octave_menu.pack(side="right")

        row_scale = ctk.CTkFrame(parent, fg_color="transparent")
        row_scale.pack(fill="x", pady=(0, 4))
        self.use_custom_scale = ctk.BooleanVar(value=self.zone.scale_override is not None)
        ctk.CTkCheckBox(row_scale, text="Scale", variable=self.use_custom_scale, width=50, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_overrides).pack(side="left")
        self.scale_var = ctk.StringVar(value=self.zone.scale_override or "Minor Pentatonic")
        self.scale_menu = ctk.CTkOptionMenu(row_scale, values=list(SCALES.keys()), variable=self.scale_var,
                                            width=100, height=20, state="disabled", command=self._update_overrides)
        self.scale_menu.pack(side="right")

        row_map = ctk.CTkFrame(parent, fg_color="transparent")
        row_map.pack(fill="x", pady=(0, 4))
        self.use_custom_map = ctk.BooleanVar(value=self.zone.mapping_override is not None)
        ctk.CTkCheckBox(row_map, text="Map", variable=self.use_custom_map, width=50, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_overrides).pack(side="left")
        self.map_var = ctk.StringVar(value=self.zone.mapping_override or "Left→Right")
        self.map_menu = ctk.CTkOptionMenu(row_map, values=DIRECTIONS, variable=self.map_var, width=100, height=20,
                                          state="disabled", command=self._update_overrides)
        self.map_menu.pack(side="right")

        self._toggle_overrides()

        ctk.CTkLabel(parent, text="Note Range", font=("Inter", 10, "bold"),
                     text_color=Theme.TEXT_MUTED).pack(anchor="w", pady=(4, 2))
        range_frame = ctk.CTkFrame(parent, fg_color="transparent")
        range_frame.pack(fill="x")

        r_low = ctk.CTkFrame(range_frame, fg_color="transparent")
        r_low.pack(fill="x")
        ctk.CTkLabel(r_low, text="Low", width=28, font=("Inter", 9)).pack(side="left")
        self.low_note_var = ctk.StringVar(value=NOTE_NAMES[self.zone.note_low % 12])
        ctk.CTkOptionMenu(r_low, values=NOTE_NAMES, variable=self.low_note_var, width=45, height=18,
                          command=self._update_range).pack(side="left", padx=2)
        self.low_oct_var = ctk.StringVar(value=str(self.zone.note_low // 12 - 1))
        ctk.CTkOptionMenu(r_low, values=[str(i) for i in range(-1, 10)], variable=self.low_oct_var,
                          width=35, height=18, command=self._update_range).pack(side="left")

        r_high = ctk.CTkFrame(range_frame, fg_color="transparent")
        r_high.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(r_high, text="High", width=28, font=("Inter", 9)).pack(side="left")
        self.high_note_var = ctk.StringVar(value=NOTE_NAMES[self.zone.note_high % 12])
        ctk.CTkOptionMenu(r_high, values=NOTE_NAMES, variable=self.high_note_var, width=45, height=18,
                          command=self._update_range).pack(side="left", padx=2)
        self.high_oct_var = ctk.StringVar(value=str(self.zone.note_high // 12 - 1))
        ctk.CTkOptionMenu(r_high, values=[str(i) for i in range(-1, 10)], variable=self.high_oct_var,
                          width=35, height=18, command=self._update_range).pack(side="left")

    def _build_music_tab_cc(self, parent):
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(row1, text="Channel", font=("Inter", 10)).pack(side="left")
        self.ch_menu = ctk.CTkOptionMenu(row1, values=[str(i) for i in range(1, 17)], width=50, height=20,
                                         font=("Inter", 10), command=self._on_channel_change)
        self.ch_menu.set(str(self.zone.channel))
        self.ch_menu.pack(side="right")

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(row2, text="CC #", font=("Inter", 10)).pack(side="left")
        self.cc_menu = ctk.CTkOptionMenu(row2, values=[str(i) for i in range(128)], width=50, height=20,
                                         command=self._on_cc_change)
        self.cc_menu.set(str(self.zone.cc_number))
        self.cc_menu.pack(side="right")

    def _build_dynamics_tab_note(self, parent):
        self.use_custom_vel = ctk.BooleanVar(value=self.zone.vel_min is not None)
        ctk.CTkCheckBox(parent, text="Custom Velocity", variable=self.use_custom_vel, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_dyn).pack(anchor="w", pady=(0, 4))

        self.vel_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.v_min_slider = ModernSlider(self.vel_frame, "Min", 0, 127,
                                         self.zone.vel_min or DEFAULT_MIN_VEL, command=self._update_dyn)
        self.v_min_slider.pack(fill="x")
        self.v_max_slider = ModernSlider(self.vel_frame, "Max", 0, 127,
                                         self.zone.vel_max or DEFAULT_MAX_VEL, command=self._update_dyn)
        self.v_max_slider.pack(fill="x")

        self.use_random_vel = ctk.BooleanVar(value=self.zone.use_random_velocity)
        self.random_vel_cb = ctk.CTkCheckBox(self.vel_frame, text="Random velocity",
                                             variable=self.use_random_vel, font=("Inter", 10),
                                             checkbox_width=14, checkbox_height=14,
                                             command=self._update_dyn)
        self.random_vel_cb.pack(anchor="w", pady=(4, 0))

        self.vel_frame.pack(fill="x", padx=8)

        ctk.CTkFrame(parent, height=1, fg_color=Theme.BORDER).pack(fill="x", pady=8)

        self.use_prob = ctk.BooleanVar(value=self.zone.use_probability)
        ctk.CTkCheckBox(parent, text="Random Chance", variable=self.use_prob, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_dyn).pack(anchor="w", pady=(0, 4))
        self.prob_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.prob_slider = ModernSlider(self.prob_frame, "Chance %", 0, 100,
                                        self.zone.probability, command=self._update_dyn)
        self.prob_slider.pack(fill="x")
        self.prob_frame.pack(fill="x", padx=8)

        ctk.CTkFrame(parent, height=1, fg_color=Theme.BORDER).pack(fill="x", pady=8)

        self.use_smooth = ctk.BooleanVar(value=self.zone.use_smoothing)
        ctk.CTkCheckBox(parent, text="Smoothening", variable=self.use_smooth, font=("Inter", 10),
                        checkbox_width=14, checkbox_height=14, command=self._toggle_dyn).pack(anchor="w", pady=(0, 4))
        self.smooth_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.smooth_slider = ModernSlider(self.smooth_frame, "Smoothness", 0, 100,
                                          self.zone.smoothing, command=self._update_dyn)
        self.smooth_slider.pack(fill="x")
        self.smooth_frame.pack(fill="x", padx=8)

        ctk.CTkFrame(parent, height=1, fg_color=Theme.BORDER).pack(fill="x", pady=8)

        self.dur_slider = ModernSlider(parent, "Duration (ms)", 50, 10000,
                                       self.zone.duration, command=self._update_dyn)
        self.dur_slider.pack(fill="x", padx=0)

        self._toggle_dyn()

    def _build_dynamics_tab_cc(self, parent):
        ctk.CTkLabel(parent, text="CC Value Range", font=("Inter", 10, "bold"),
                     text_color=Theme.ACCENT).pack(anchor="w", pady=(4, 6))
        self.vel_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.vel_frame.pack(fill="x", padx=8)
        self.v_min_slider = ModernSlider(self.vel_frame, "Min Value", 0, 127,
                                         self.zone.vel_min if self.zone.vel_min is not None else 0,
                                         command=self._update_dyn)
        self.v_min_slider.pack(fill="x")
        self.v_max_slider = ModernSlider(self.vel_frame, "Max Value", 0, 127,
                                         self.zone.vel_max if self.zone.vel_max is not None else 127,
                                         command=self._update_dyn)
        self.v_max_slider.pack(fill="x")

        ctk.CTkFrame(parent, height=1, fg_color=Theme.BORDER).pack(fill="x", pady=8)
        ctk.CTkLabel(parent, text="Signal Smoothing", font=("Inter", 10, "bold"),
                     text_color=Theme.ACCENT).pack(anchor="w", pady=(0, 6))
        self.smooth_slider = ModernSlider(parent, "Slew Rate", 0, 100,
                                          self.zone.smoothing, command=self._update_dyn)
        self.smooth_slider.pack(fill="x", padx=8)
        self.zone.use_smoothing = True

    def _build_tracking_tab(self, parent):
        ctk.CTkLabel(parent, text="Color Tracking", font=("Inter", 10, "bold"),
                     text_color=Theme.ACCENT).pack(anchor="w", pady=(4, 6))
        ctk.CTkLabel(parent, text="Track specific color in this zone",
                     font=("Inter", 9), text_color=Theme.TEXT_MUTED).pack(anchor="w", pady=(0, 8))

        self.use_color_track = ctk.BooleanVar(value=self.zone.use_custom_color)
        ctk.CTkSwitch(parent, text="Enable", variable=self.use_color_track, font=("Inter", 10),
                      command=self._toggle_track).pack(anchor="w")

        self.track_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.track_frame.pack(fill="x", pady=8)

        self.track_color_preview = ctk.CTkLabel(self.track_frame, text="", width=36, height=36,
                                                corner_radius=4, fg_color=self._hsv_to_hex(self.zone.track_hsv))
        self.track_color_preview.pack(side="left", padx=(8, 8))

        self.pick_btn = ctk.CTkButton(self.track_frame, text="Pick", command=self._start_pick,
                                      fg_color=Theme.BG_HOVER, hover_color=Theme.ACCENT, width=50, height=24)
        self.pick_btn.pack(side="left")

        self.track_info = ctk.CTkLabel(self.track_frame, text=f"HSV: {self.zone.track_hsv}",
                                       font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.track_info.pack(side="left", padx=8)

        self._toggle_track()

    def _on_cc_change(self, value):
        self.zone.cc_number = int(value)

    def _toggle_track(self):
        self.zone.use_custom_color = self.use_color_track.get()
        if self.use_color_track.get():
            self.track_frame.pack(fill="x", pady=8)
        else:
            self.track_frame.pack_forget()

    def _start_pick(self):
        if self.on_pick:
            self.on_pick(self.zone)

    def update_picked_color(self):
        hex_col = self._hsv_to_hex(self.zone.track_hsv)
        self.track_color_preview.configure(fg_color=hex_col)
        self.track_info.configure(text=f"HSV: {self.zone.track_hsv}")

    def _hsv_to_hex(self, hsv):
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hsv[0] / 179.0, hsv[1] / 255.0, hsv[2] / 255.0)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

    def _toggle(self, event=None):
        self.expanded = not self.expanded
        self.arrow.configure(text="▾" if self.expanded else "▸")
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()

    def _on_name_change(self, *args):
        self.zone.name = self.name_var.get()
        if self.on_update:
            self.on_update()

    def _on_channel_change(self, value):
        self.zone.channel = int(value)
        if self.on_update:
            self.on_update()

    def _toggle_mute(self):
        self.zone.mute = not self.zone.mute
        self.mute_btn.configure(fg_color=Theme.DANGER if self.zone.mute else Theme.BG_INPUT)
        if self.on_update:
            self.on_update()

    def _pick_color(self):
        color = colorchooser.askcolor(title="Zone Color", color=self.zone.color_hex)[1]
        if color:
            self.zone.update_color(color)
            self.color_bar.configure(fg_color=color)
            if self.on_update:
                self.on_update()

    def _delete(self):
        if self.on_delete:
            self.on_delete(self.zone, self)

    def _toggle_overrides(self):
        if self.use_custom_root.get():
            self.root_menu.configure(state="normal")
            self.zone.root_override = self.root_var.get()
        else:
            self.root_menu.configure(state="disabled")
            self.zone.root_override = None

        if self.use_custom_octave.get():
            self.octave_menu.configure(state="normal")
            self.zone.octave_override = int(self.octave_var.get())
        else:
            self.octave_menu.configure(state="disabled")
            self.zone.octave_override = None

        if self.use_custom_scale.get():
            self.scale_menu.configure(state="normal")
            self.zone.scale_override = self.scale_var.get()
        else:
            self.scale_menu.configure(state="disabled")
            self.zone.scale_override = None

        if self.use_custom_map.get():
            self.map_menu.configure(state="normal")
            self.zone.mapping_override = self.map_var.get()
        else:
            self.map_menu.configure(state="disabled")
            self.zone.mapping_override = None

    def _update_overrides(self, *args):
        if self.use_custom_root.get():
            self.zone.root_override = self.root_var.get()
        if self.use_custom_octave.get():
            self.zone.octave_override = int(self.octave_var.get())
        if self.use_custom_scale.get():
            self.zone.scale_override = self.scale_var.get()
        if self.use_custom_map.get():
            self.zone.mapping_override = self.map_var.get()

    def _calc_midi(self, name, oct_str):
        idx = NOTE_NAMES.index(name) if name in NOTE_NAMES else 0
        return (int(oct_str) + 1) * 12 + idx

    def _update_range(self, *args):
        low = self._calc_midi(self.low_note_var.get(), self.low_oct_var.get())
        high = self._calc_midi(self.high_note_var.get(), self.high_oct_var.get())
        self.zone.note_low = max(0, min(127, low))
        self.zone.note_high = max(0, min(127, high))

    def _toggle_dyn(self):
        if self.use_custom_vel.get():
            self.vel_frame.pack(fill="x", padx=8)
            self.zone.vel_min = int(self.v_min_slider.get())
            self.zone.vel_max = int(self.v_max_slider.get())
        else:
            self.vel_frame.pack_forget()
            self.zone.vel_min = None
            self.zone.vel_max = None

        if hasattr(self, 'use_prob'):
            if self.use_prob.get():
                self.prob_frame.pack(fill="x", padx=8)
                self.zone.use_probability = True
                self.zone.probability = int(self.prob_slider.get())
            else:
                self.prob_frame.pack_forget()
                self.zone.use_probability = False

        if hasattr(self, 'use_smooth'):
            if self.use_smooth.get():
                self.smooth_frame.pack(fill="x", padx=8)
                self.zone.use_smoothing = True
                self.zone.smoothing = int(self.smooth_slider.get())
            else:
                self.smooth_frame.pack_forget()
                self.zone.use_smoothing = False

    def _update_dyn(self, *args):
        if hasattr(self, 'use_custom_vel') and self.use_custom_vel.get() or self.zone.zone_type == "CC":
            self.zone.vel_min = int(self.v_min_slider.get())
            self.zone.vel_max = int(self.v_max_slider.get())

        if hasattr(self, 'use_random_vel'):
            self.zone.use_random_velocity = self.use_random_vel.get()

        if hasattr(self, 'use_prob') and self.use_prob.get() and hasattr(self, 'prob_slider'):
            self.zone.probability = int(self.prob_slider.get())
        if (hasattr(self, 'use_smooth') and self.use_smooth.get() or self.zone.zone_type == "CC") and hasattr(self, 'smooth_slider'):
            self.zone.smoothing = int(self.smooth_slider.get())
        if hasattr(self, 'dur_slider'):
            self.zone.duration = int(self.dur_slider.get())


class StatusBar(ctk.CTkFrame):
    """Bottom status bar with performance info."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=Theme.BG_PANEL, height=24, corner_radius=0)
        self.pack_propagate(False)

        self.midi_indicator = ctk.CTkLabel(self, text="●", font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.midi_indicator.pack(side="left", padx=(10, 3))

        self.midi_label = ctk.CTkLabel(self, text="MIDI: --", font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.midi_label.pack(side="left")

        ctk.CTkFrame(self, width=1, height=14, fg_color=Theme.BORDER).pack(side="left", padx=10)

        self.fps_label = ctk.CTkLabel(self, text="-- FPS", font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.fps_label.pack(side="left")

        # OPTIMIZATION: Show thread status
        self.thread_label = ctk.CTkLabel(self, text="⚡ Threaded", font=("Inter", 9), 
                                         text_color=Theme.SUCCESS)
        self.thread_label.pack(side="left", padx=(10, 0))

        # OPTIMIZATION: Show numba status
        numba_status = "JIT ✓" if NUMBA_AVAILABLE else "JIT ✗"
        numba_color = Theme.SUCCESS if NUMBA_AVAILABLE else Theme.WARNING
        self.numba_label = ctk.CTkLabel(self, text=numba_status, font=("Inter", 9), 
                                        text_color=numba_color)
        self.numba_label.pack(side="left", padx=(10, 0))

        self.grid_label = ctk.CTkLabel(self, text="8×8", font=("Inter", 9), text_color=Theme.TEXT_MUTED)
        self.grid_label.pack(side="right", padx=10)

        ctk.CTkLabel(self, text="Grid:", font=("Inter", 9), text_color=Theme.TEXT_MUTED).pack(side="right")

    def set_midi_status(self, connected: bool, active: bool = False):
        if active:
            self.midi_indicator.configure(text_color=Theme.SUCCESS)
            self.midi_label.configure(text="MIDI: ▶", text_color=Theme.SUCCESS)
        elif connected:
            self.midi_indicator.configure(text_color=Theme.WARNING)
            self.midi_label.configure(text="MIDI: Ready", text_color=Theme.TEXT_SECONDARY)
        else:
            self.midi_indicator.configure(text_color=Theme.TEXT_MUTED)
            self.midi_label.configure(text="MIDI: --", text_color=Theme.TEXT_MUTED)

    def set_fps(self, fps: float):
        self.fps_label.configure(text=f"{fps:.0f} FPS")

    def set_grid(self, w: int, h: int):
        self.grid_label.configure(text=f"{w}×{h}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class LavaMIDIApp(ctk.CTk):
    """Main application with threaded video processing."""

    def __init__(self):
        super().__init__()

        self.title("Lava MIDI")
        self.geometry("1400x900")
        self.minsize(1000, 650)
        self.configure(fg_color=Theme.BG_DARK)

        # State
        self.app_state = AppState()
        self.zones: List[Zone] = []
        self.cell_map: Dict[Tuple[int, int], Zone] = {}

        # Selection & Picking
        self.is_selecting = False
        self.sel_start = (0, 0)
        self.sel_end = (0, 0)
        self.drag_cells: Set[Tuple[int, int]] = set()
        self.selected_cells: Set[Tuple[int, int]] = set()
        self.picking_zone: Optional[Zone] = None

        # Processing buffers
        self._init_buffers()

        # Components
        self.analyzer = MotionAnalyzer(
            self.app_state.grid_w, 
            self.app_state.grid_h,
            self.app_state.analysis_width
        )
        self.pitch_calc = PitchCalculator(self.app_state.grid_w, self.app_state.grid_h)

        # OPTIMIZATION: Threaded video processing
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.video_processor: Optional[VideoProcessor] = None

        # Video state
        self.last_frame_time = time.time()
        self.fps_smooth = 30.0
        self.last_frame_bgr: Optional[np.ndarray] = None

        # Early logs
        self._early_logs: List[Tuple[str, str]] = []

        # MIDI
        self.midi = MIDIManager(self.app_state.midi_port_name, self._log)

        # Build UI
        self._build_ui()

        # Flush Early Logs
        for msg, level in self._early_logs:
            self._log(msg, level)
        self._early_logs.clear()

        # OPTIMIZATION: Start video processor thread
        self._init_video_thread()

        if os.path.exists(DEFAULT_SETTINGS_FILE):
            self._load_settings(DEFAULT_SETTINGS_FILE)

        self.status_bar.set_midi_status(self.midi.output is not None)
        self.status_bar.set_grid(self.app_state.grid_w, self.app_state.grid_h)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._restore_layout)
        
        # Start UI update loop (now only handles display, not processing)
        self._update_loop()

    def _init_buffers(self):
        """Initialize processing buffers."""
        self.grid_timers = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_end_times = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_velocities = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_current_vals = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_target_vals = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.cooldown_matrix = np.full((self.app_state.grid_h, self.app_state.grid_w), COOLDOWN_TIME)
        
        # Easing animation buffers
        self.vis_anim_start_times = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_anim_from = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        self.vis_anim_to = np.zeros((self.app_state.grid_h, self.app_state.grid_w))
        
        # Initialize zone cells with base alpha
        for (x, y), zone in self.cell_map.items():
            if y < self.app_state.grid_h and x < self.app_state.grid_w:
                self.vis_current_vals[y, x] = ZONE_ALPHA
                self.vis_anim_to[y, x] = ZONE_ALPHA

    def _init_video_thread(self):
        """OPTIMIZATION: Initialize the video processing thread (but don't start capture yet)."""
        self.video_processor = VideoProcessor(
            self.app_state, 
            self.frame_queue, 
            self.result_queue,
            self._log
        )
        self.video_processor.set_analyzer(self.analyzer)
        # Don't auto-start video capture - wait for user to click camera/file button
        self.video_processor.start()
        self._log("Ready - click 📷 or 📁 to start video", "info")
        if NUMBA_AVAILABLE:
            self._log("Numba JIT enabled", "success")
        else:
            self._log("Numba not available - install for better performance", "warning")

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.h_paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=3,
                                      bg=Theme.BORDER, sashrelief=tk.FLAT)
        self.h_paned.grid(row=0, column=0, sticky="nsew")

        # Layout ratio: 1/6 left sidebar, 4/6 video, 2/6 right panel
        sidebar_container = ctk.CTkFrame(self.h_paned, fg_color=Theme.BG_PANEL, corner_radius=0)
        self.h_paned.add(sidebar_container, minsize=150, width=233)  # 1/6 of 1400
        sidebar_container.grid_rowconfigure(0, weight=1)
        sidebar_container.grid_columnconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(sidebar_container, fg_color=Theme.BG_PANEL, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.midi_section = ctk.CTkFrame(sidebar_container, fg_color=Theme.BG_CARD, corner_radius=0)
        self.midi_section.grid(row=1, column=0, sticky="ew")

        self.video_container = ctk.CTkFrame(self.h_paned, fg_color=Theme.BG_DARK, corner_radius=0)
        self.h_paned.add(self.video_container, minsize=400, width=933)  # 4/6 of 1400

        right_container = ctk.CTkFrame(self.h_paned, fg_color=Theme.BG_PANEL, corner_radius=0)
        self.h_paned.add(right_container, minsize=200, width=234)  # 2/6 of 1400

        self.v_paned = tk.PanedWindow(right_container, orient=tk.VERTICAL, sashwidth=3,
                                      bg=Theme.BORDER, sashrelief=tk.FLAT)
        self.v_paned.pack(fill="both", expand=True)

        self.tabs_right = ctk.CTkTabview(self.v_paned, fg_color=Theme.BG_PANEL,
                                         segmented_button_fg_color=Theme.BG_INPUT,
                                         segmented_button_unselected_color=Theme.BG_INPUT,
                                         segmented_button_selected_color=Theme.ACCENT)
        self.v_paned.add(self.tabs_right, minsize=200, height=650)

        self.tab_zones = self.tabs_right.add("Zones")
        self.tab_drones = self.tabs_right.add("Drones")

        self.log_container = ctk.CTkFrame(self.v_paned, fg_color=Theme.BG_CARD, corner_radius=0)
        self.v_paned.add(self.log_container, minsize=80, height=180)

        self._build_sidebar_content()
        self._build_midi_section()
        self._build_video_area()
        self._build_zone_manager(self.tab_zones)
        self._build_drone_manager(self.tab_drones)
        self._build_log_area(self.log_container)
        self._build_status_bar()

    def _build_sidebar_content(self):
        sidebar = self.sidebar

        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(14, 10))
        ctk.CTkLabel(header, text="🌋", font=("Inter", 22)).pack(side="left")
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=(6, 0))
        ctk.CTkLabel(title_frame, text="Lava MIDI", font=("Inter", 14, "bold"),
                     text_color=Theme.TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="v61 ⚡", font=("Inter", 9),
                     text_color=Theme.SUCCESS).pack(anchor="w")

        # PROJECT SECTION
        project = CollapsibleSection(sidebar, "Project", expanded=True)
        project.pack(fill="x", padx=8, pady=(0, 6))

        ctk.CTkButton(project.content, text="💾 Save", fg_color=Theme.ACCENT,
                      hover_color=Theme.ACCENT_HOVER, height=28, font=("Inter", 11, "bold"),
                      command=lambda: self._save_settings(DEFAULT_SETTINGS_FILE)).pack(fill="x", pady=(0, 4))

        io_row = ctk.CTkFrame(project.content, fg_color="transparent")
        io_row.pack(fill="x")
        ctk.CTkButton(io_row, text="Export", fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER,
                      height=24, font=("Inter", 10),
                      command=lambda: self._save_settings(None)).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(io_row, text="Import", fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER,
                      height=24, font=("Inter", 10),
                      command=lambda: self._load_settings(None)).pack(side="left", expand=True, fill="x", padx=(2, 0))

        # VIDEO SECTION - expanded by default, right after Project
        video = CollapsibleSection(sidebar, "Video", expanded=True)
        video.pack(fill="x", padx=8, pady=(0, 6))

        src_row = ctk.CTkFrame(video.content, fg_color="transparent")
        src_row.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(src_row, text="📷 Webcam", fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER,
                      height=28, font=("Inter", 10), command=self._set_camera).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(src_row, text="📁 File", fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER,
                      height=28, font=("Inter", 10), command=self._set_file).pack(side="left", expand=True, fill="x", padx=(2, 0))

        self.bright_slider = ModernSlider(video.content, "Brightness", -100, 100, 0,
                                          lambda v: setattr(self.app_state, 'brightness', v))
        self.bright_slider.pack(fill="x", pady=(0, 6))
        self.contrast_slider = ModernSlider(video.content, "Contrast", 0.5, 3.0, 1.0,
                                            lambda v: setattr(self.app_state, 'contrast', v), format_str="{:.1f}")
        self.contrast_slider.pack(fill="x", pady=(0, 6))
        self.blur_slider = ModernSlider(video.content, "Blur", 0, 50, self.app_state.video_blur_amount,
                                        lambda v: setattr(self.app_state, 'video_blur_amount', int(v)))
        self.blur_slider.pack(fill="x", pady=(0, 6))

        toggle_row = ctk.CTkFrame(video.content, fg_color="transparent")
        toggle_row.pack(fill="x")
        self.grid_switch = ctk.CTkSwitch(toggle_row, text="Grid", font=("Inter", 10), width=40,
                                         command=lambda: setattr(self.app_state, 'show_grid', self.grid_switch.get()))
        self.grid_switch.select()
        self.grid_switch.pack(side="left")
        self.notes_switch = ctk.CTkSwitch(toggle_row, text="Notes", font=("Inter", 10), width=40,
                                          command=lambda: setattr(self.app_state, 'show_notes', self.notes_switch.get()))
        self.notes_switch.select()
        self.notes_switch.pack(side="left", padx=(8, 0))

        toggle_row2 = ctk.CTkFrame(video.content, fg_color="transparent")
        toggle_row2.pack(fill="x", pady=(4, 0))
        self.pixelate_switch = ctk.CTkSwitch(toggle_row2, text="Pixelate", font=("Inter", 10), width=40,
                                             command=lambda: setattr(self.app_state, 'pixelate_view', self.pixelate_switch.get()))
        self.pixelate_switch.pack(side="left")
        self.flow_switch = ctk.CTkSwitch(toggle_row2, text="Flow", font=("Inter", 10), width=40,
                                         command=lambda: setattr(self.app_state, 'show_flow_vectors', self.flow_switch.get()))
        self.flow_switch.pack(side="left", padx=(8, 0))

        # Music
        music = CollapsibleSection(sidebar, "Music", expanded=False)
        music.pack(fill="x", padx=8, pady=(0, 6))
        self.scale_dropdown = ModernDropdown(music.content, "Scale", list(SCALES.keys()),
                                             self.app_state.scale_name,
                                             command=lambda v: setattr(self.app_state, 'scale_name', v))
        self.scale_dropdown.pack(fill="x", pady=(0, 6))

        row = ctk.CTkFrame(music.content, fg_color="transparent")
        row.pack(fill="x", pady=(0, 6))
        self.root_var = ctk.StringVar(value=self.app_state.root_key)
        ctk.CTkOptionMenu(row, values=NOTE_NAMES, variable=self.root_var, fg_color=Theme.BG_INPUT,
                          width=50, height=24,
                          command=lambda v: setattr(self.app_state, 'root_key', v)).pack(side="left", padx=(0, 3))
        self.oct_var = ctk.StringVar(value=str(self.app_state.octave))
        ctk.CTkOptionMenu(row, values=OCTAVES, variable=self.oct_var, fg_color=Theme.BG_INPUT,
                          width=50, height=24,
                          command=lambda v: setattr(self.app_state, 'octave', int(v))).pack(side="left")

        self.dir_dropdown = ModernDropdown(music.content, "Mapping", DIRECTIONS,
                                           self.app_state.mapping_direction,
                                           command=lambda v: setattr(self.app_state, 'mapping_direction', v))
        self.dir_dropdown.pack(fill="x")

        # Grid
        grid_sec = CollapsibleSection(sidebar, "Grid", expanded=False)
        grid_sec.pack(fill="x", padx=8, pady=(0, 6))
        self.grid_w_slider = ModernSlider(grid_sec.content, "Columns", MIN_GRID, MAX_GRID,
                                          self.app_state.grid_w, self._on_grid_w_change,
                                          number_of_steps=MAX_GRID - MIN_GRID)
        self.grid_w_slider.pack(fill="x", pady=(0, 6))
        self.grid_h_slider = ModernSlider(grid_sec.content, "Rows", MIN_GRID, MAX_GRID,
                                          self.app_state.grid_h, self._on_grid_h_change,
                                          number_of_steps=MAX_GRID - MIN_GRID)
        self.grid_h_slider.pack(fill="x")

        # Motion
        motion = CollapsibleSection(sidebar, "Motion", expanded=False)
        motion.pack(fill="x", padx=8, pady=(0, 6))
        self.motion_dropdown = ModernDropdown(motion.content, "Detection", [m.value for m in MotionMode],
                                              self.app_state.motion_mode,
                                              command=lambda v: setattr(self.app_state, 'motion_mode', v))
        self.motion_dropdown.pack(fill="x", pady=(0, 6))
        
        # Extended sensitivity range (1-200) for finer control at high values
        self.sens_slider = ModernSlider(motion.content, "Sensitivity", 1, 200,
                                        self.app_state.sensitivity, self._on_sensitivity_change)
        self.sens_slider.pack(fill="x", pady=(0, 6))
        
        # NEW: Motion gain/amplification control
        self.gain_slider = ModernSlider(motion.content, "Motion Gain", 0.5, 5.0,
                                        self.app_state.motion_gain,
                                        lambda v: setattr(self.app_state, 'motion_gain', v),
                                        format_str="{:.1f}x")
        self.gain_slider.pack(fill="x", pady=(0, 6))
        
        self.smooth_slider = ModernSlider(motion.content, "Smoothing", 0, 0.9,
                                          self.app_state.motion_smoothing,
                                          lambda v: setattr(self.app_state, 'motion_smoothing', v),
                                          format_str="{:.1f}")
        self.smooth_slider.pack(fill="x", pady=(0, 6))
        
        # OPTIMIZATION: Analysis resolution control
        self.analysis_slider = ModernSlider(motion.content, "Analysis Res", 
                                            ANALYSIS_WIDTH_MIN, ANALYSIS_WIDTH_MAX,
                                            self.app_state.analysis_width, 
                                            self._on_analysis_width_change)
        self.analysis_slider.pack(fill="x")

        # Velocity
        vel = CollapsibleSection(sidebar, "Velocity", expanded=False)
        vel.pack(fill="x", padx=8, pady=(0, 6))
        self.vel_min_slider = ModernSlider(vel.content, "Min", 0, 127,
                                           self.app_state.min_velocity, self._on_min_vel_change)
        self.vel_min_slider.pack(fill="x", pady=(0, 6))
        self.vel_max_slider = ModernSlider(vel.content, "Max", 0, 127,
                                           self.app_state.max_velocity, self._on_max_vel_change)
        self.vel_max_slider.pack(fill="x", pady=(0, 6))
        self.curve_dropdown = ModernDropdown(vel.content, "Curve", [c.value for c in VelocityCurve],
                                             self.app_state.velocity_curve,
                                             command=lambda v: setattr(self.app_state, 'velocity_curve', v))
        self.curve_dropdown.pack(fill="x")

    def _build_midi_section(self):
        section = self.midi_section

        ctk.CTkFrame(section, height=1, fg_color=Theme.BORDER).pack(fill="x")

        dev_row = ctk.CTkFrame(section, fg_color="transparent")
        dev_row.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkLabel(dev_row, text="Output", font=("Inter", 10), text_color=Theme.TEXT_SECONDARY).pack(anchor="w")

        sel_row = ctk.CTkFrame(dev_row, fg_color="transparent")
        sel_row.pack(fill="x", pady=(2, 0))

        self.port_var = ctk.StringVar(value=self.app_state.midi_port_name)
        self.port_menu = ctk.CTkOptionMenu(sel_row, variable=self.port_var,
                                           values=self.midi.get_available_ports(),
                                           command=self._on_port_change, fg_color=Theme.BG_INPUT,
                                           button_color=Theme.BG_HOVER, height=24)
        self.port_menu.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(sel_row, text="⟳", width=24, height=24, fg_color=Theme.BG_INPUT,
                      hover_color=Theme.BG_HOVER, command=self._refresh_ports).pack(side="right", padx=(3, 0))

        trigger_row = ctk.CTkFrame(section, fg_color="transparent")
        trigger_row.pack(fill="x", padx=10, pady=(8, 0))

        self.manual_trigger_switch = ctk.CTkSwitch(trigger_row, text="Manual", font=("Inter", 10),
                                                   command=self._toggle_manual_trigger)
        self.manual_trigger_switch.pack(side="left")

        self.midi_btn = ctk.CTkButton(section, text="▶  Start", height=40, font=("Inter", 13, "bold"),
                                      fg_color=Theme.ACCENT, hover_color=Theme.ACCENT_HOVER,
                                      command=self._toggle_midi)
        self.midi_btn.pack(fill="x", padx=10, pady=(8, 10))

    def _refresh_ports(self):
        ports = self.midi.get_available_ports()
        self.port_menu.configure(values=ports)
        if self.app_state.midi_port_name not in ports and ports:
            self.port_var.set(ports[0])
            self._on_port_change(ports[0])
        elif not ports:
            self.port_var.set("No Devices")

    def _on_port_change(self, port_name):
        self.app_state.midi_port_name = port_name
        self.midi.set_port(port_name)
        self.status_bar.set_midi_status(self.midi.output is not None, self.app_state.midi_active)

    def _build_video_area(self):
        self.video_label = tk.Label(self.video_container, bg=Theme.BG_DARK)
        self.video_label.pack(expand=True, fill="both")
        self.video_label.bind("<Button-1>", self._on_mouse_down)
        self.video_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.video_label.bind("<Button-3>", self._on_right_click)

    def _build_zone_manager(self, parent):
        tools = CollapsibleSection(parent, "Zone Tools", expanded=True)
        tools.pack(fill="x", padx=6, pady=(6, 0))

        t_frame = tools.content
        t_frame.grid_columnconfigure(0, weight=1)
        t_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(t_frame, text="+ From Selection", command=self._create_zone_from_selection,
                      fg_color=Theme.ACCENT, hover_color=Theme.ACCENT_HOVER, height=26,
                      font=("Inter", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        ctk.CTkButton(t_frame, text="Quadrants", command=self._create_quadrants,
                      fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER, height=22,
                      font=("Inter", 9)).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ctk.CTkButton(t_frame, text="Rows", command=self._create_rows,
                      fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER, height=22,
                      font=("Inter", 9)).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ctk.CTkButton(t_frame, text="Halves", command=self._create_halves,
                      fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER, height=22,
                      font=("Inter", 9)).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ctk.CTkButton(t_frame, text="Clear All", command=self._clear_zones,
                      fg_color=Theme.BG_INPUT, hover_color=Theme.DANGER, height=22,
                      text_color=Theme.DANGER, font=("Inter", 9)).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        self.selection_label = ctk.CTkLabel(parent, text="Selection: 0 cells", font=("Inter", 9),
                                            text_color=Theme.TEXT_MUTED)
        self.selection_label.pack(anchor="w", padx=10, pady=(6, 2))

        list_head = ctk.CTkFrame(parent, fg_color="transparent", height=26)
        list_head.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(list_head, text="Active Zones", font=("Inter", 11, "bold"),
                     text_color=Theme.TEXT_PRIMARY).pack(side="left")

        self.zones_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.zones_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 6))

    def _build_drone_manager(self, parent):
        self.drones_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.drones_scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self._refresh_drone_widgets()

    def _refresh_drone_widgets(self):
        for widget in self.drones_scroll.winfo_children():
            widget.destroy()
        for i, drone in enumerate(self.app_state.drones):
            DroneCard(self.drones_scroll, drone, i).pack(fill="x", pady=(0, 4))

    def _build_log_area(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent", height=24)
        header.pack(fill="x", padx=6, pady=(3, 0))
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="Log", font=("Inter", 10, "bold"),
                     text_color=Theme.TEXT_MUTED).pack(side="left")

        ctk.CTkButton(header, text="Clear", width=36, height=18, font=("Inter", 8),
                      fg_color=Theme.BG_INPUT, hover_color=Theme.BG_HOVER,
                      command=self._clear_log).pack(side="right")

        self.log_midi_switch = ctk.CTkSwitch(header, text="MIDI", font=("Inter", 9), width=36,
                                             command=lambda: setattr(self.app_state, 'log_midi_signals',
                                                                     self.log_midi_switch.get()))
        self.log_midi_switch.pack(side="right", padx=6)

        self.log_box = ctk.CTkTextbox(parent, font=("Consolas", 9), fg_color=Theme.BG_DARK,
                                      text_color=Theme.TEXT_SECONDARY)
        self.log_box.pack(fill="both", expand=True, padx=6, pady=(2, 6))

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _build_status_bar(self):
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, sticky="ew")

    # ══════════════════════════════════════════════════════════════════════════
    # LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _reset_buffers(self):
        self.midi.panic()
        self._init_buffers()
        self.analyzer.resize(self.app_state.grid_w, self.app_state.grid_h)
        self.pitch_calc.resize(self.app_state.grid_w, self.app_state.grid_h)
        self.status_bar.set_grid(self.app_state.grid_w, self.app_state.grid_h)

    def _set_camera(self):
        self.app_state.video_source_type = "camera"
        if self.video_processor:
            self.video_processor.init_video("camera")

    def _set_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.app_state.video_source_type = "file"
            self.app_state.video_file_path = path
            if self.video_processor:
                self.video_processor.init_video("file", path)

    def _on_grid_w_change(self, val):
        if int(val) != self.app_state.grid_w:
            self.app_state.grid_w = int(val)
            self._reset_buffers()

    def _on_grid_h_change(self, val):
        if int(val) != self.app_state.grid_h:
            self.app_state.grid_h = int(val)
            self._reset_buffers()

    def _on_sensitivity_change(self, val):
        self.app_state.sensitivity = int(val)

    def _on_analysis_width_change(self, val):
        """OPTIMIZATION: Update analysis resolution."""
        self.app_state.analysis_width = int(val)
        self.analyzer.set_analysis_width(int(val))

    def _on_min_vel_change(self, val):
        self.app_state.min_velocity = int(val)
        self.vel_max_slider.set(max(self.app_state.min_velocity, self.app_state.max_velocity))

    def _on_max_vel_change(self, val):
        self.app_state.max_velocity = int(val)
        self.vel_min_slider.set(min(self.app_state.min_velocity, self.app_state.max_velocity))

    def _toggle_midi(self):
        self.app_state.midi_active = not self.app_state.midi_active
        if self.app_state.midi_active:
            self.midi_btn.configure(text="⏹  Stop", fg_color=Theme.DANGER, hover_color="#dc2626")
            self.status_bar.set_midi_status(True, True)
            self._log("MIDI started", "success")
        else:
            self.midi_btn.configure(text="▶  Start", fg_color=Theme.ACCENT, hover_color=Theme.ACCENT_HOVER)
            self.midi.panic()
            self.status_bar.set_midi_status(self.midi.output is not None, False)
            self._log("MIDI stopped", "info")

    def _toggle_manual_trigger(self):
        self.app_state.manual_trigger_mode = self.manual_trigger_switch.get()
        self._log(f"Manual trigger {'ON' if self.app_state.manual_trigger_mode else 'OFF'}", "info")

    def _grid_from_mouse(self, x: int, y: int) -> Tuple[int, int]:
        w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if w < 10 or h < 10:
            return 0, 0
        gx = int((x / w) * self.app_state.grid_w)
        gy = int((y / h) * self.app_state.grid_h)
        return (max(0, min(gx, self.app_state.grid_w - 1)), max(0, min(gy, self.app_state.grid_h - 1)))

    def _calc_selection(self):
        x1, x2 = sorted([self.sel_start[0], self.sel_end[0]])
        y1, y2 = sorted([self.sel_start[1], self.sel_end[1]])
        return {(x, y) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)}

    def _enter_pick_mode(self, zone):
        self.picking_zone = zone
        self.video_label.config(cursor="crosshair")
        self._log(f"Pick color for {zone.name}...", "info")

    def _on_mouse_down(self, event):
        if self.picking_zone:
            if self.last_frame_bgr is not None:
                vw, vh = self.video_label.winfo_width(), self.video_label.winfo_height()
                ih, iw = self.last_frame_bgr.shape[:2]
                px = int(event.x / vw * iw)
                py = int(event.y / vh * ih)
                r = 4
                y1, y2 = max(0, py - r), min(ih, py + r + 1)
                x1, x2 = max(0, px - r), min(iw, px + r + 1)
                roi = self.last_frame_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mean_hsv = cv2.mean(hsv_roi)[:3]
                    self.picking_zone.track_hsv = tuple(map(int, mean_hsv))
                    self._log(f"Picked HSV: {self.picking_zone.track_hsv}", "success")
                    for child in self.zones_scroll.winfo_children():
                        if isinstance(child, CollapsibleZoneCard) and child.zone == self.picking_zone:
                            child.update_picked_color()
            self.picking_zone = None
            self.video_label.config(cursor="")
            return

        gx, gy = self._grid_from_mouse(event.x, event.y)
        if self.app_state.manual_trigger_mode:
            zone = self.cell_map.get((gx, gy))
            if zone and not zone.mute:
                pitch = self.pitch_calc.calculate(gx, gy, self._get_root_midi(zone),
                                                  self._get_scale(zone), self.app_state.mapping_direction, 0)
                pitch = self._wrap_pitch_to_range(pitch, zone)
                self._start_cell_anim(gx, gy, 1.0, time.time())
                self.midi.send_note_on(pitch, self.app_state.manual_trigger_velocity, zone.channel - 1)
                if self.app_state.log_midi_signals:
                    self._log(f"⚡ {zone.name} Note {pitch}", "midi")
                self.after(200, lambda: self._manual_reset(gx, gy))
            return

        self.is_selecting = True
        self.sel_start = self.sel_end = (gx, gy)
        self.drag_cells = self._calc_selection()

    def _manual_reset(self, x, y):
        self._start_cell_anim(x, y, ZONE_ALPHA if (x, y) in self.cell_map else 0.0, time.time())

    def _on_mouse_drag(self, event):
        if self.picking_zone or self.app_state.manual_trigger_mode:
            return
        if self.is_selecting:
            self.sel_end = self._grid_from_mouse(event.x, event.y)
            self.drag_cells = self._calc_selection()

    def _on_mouse_up(self, event):
        if self.picking_zone or self.app_state.manual_trigger_mode:
            return
        self.is_selecting = False
        self.selected_cells = self.drag_cells.copy()
        self.drag_cells = set()
        self._update_selection_label()

    def _on_right_click(self, event):
        self.selected_cells = set()
        self.drag_cells = set()
        self._update_selection_label()

    def _update_selection_label(self):
        if hasattr(self, 'selection_label'):
            count = len(self.selected_cells)
            self.selection_label.configure(text=f"Selection: {count} cell{'s' if count != 1 else ''}")

    def _rebuild_cell_map(self):
        self.cell_map = {cell: z for z in self.zones for cell in z.cells}
        # Update visual values for zone cells
        now = time.time()
        for (x, y), zone in self.cell_map.items():
            if y < self.app_state.grid_h and x < self.app_state.grid_w:
                # Start animation to ZONE_ALPHA for new zone cells
                if self.vis_current_vals[y, x] < ZONE_ALPHA:
                    self._start_cell_anim(x, y, ZONE_ALPHA, now)

    def _set_zone_cells(self, zone: Zone, widget: CollapsibleZoneCard):
        if not self.selected_cells:
            self._log("Select cells first", "warning")
            return

        for cell in zone.cells:
            if cell in self.cell_map and self.cell_map[cell] == zone:
                del self.cell_map[cell]

        zone.cells = self.selected_cells.copy()

        for cell in zone.cells:
            self.cell_map[cell] = zone

        widget.update_cell_count()

        self.selected_cells = set()
        self._update_selection_label()

        self._log(f"Updated {zone.name} ({len(zone.cells)} cells)", "success")

    def _add_zone_widget(self, zone: Zone):
        card = CollapsibleZoneCard(self.zones_scroll, zone, self._delete_zone,
                                   self._rebuild_cell_map, self._enter_pick_mode, self._set_zone_cells)
        card.pack(fill="x", pady=(0, 4))

    def _create_zone_from_selection(self):
        if not self.selected_cells:
            self._log("Select cells first", "warning")
            return
        idx = len(self.zones)
        zone = Zone(f"Zone {idx + 1}", (idx % 16) + 1,
                    Theme.ZONE_COLORS[idx % len(Theme.ZONE_COLORS)], self.selected_cells.copy())
        self.zones.append(zone)
        self._rebuild_cell_map()
        self._add_zone_widget(zone)
        self.selected_cells = set()
        self._update_selection_label()
        self._log(f"Created {zone.name}", "success")

    def _create_quadrants(self):
        self._clear_zones()
        hw, hh = self.app_state.grid_w // 2, self.app_state.grid_h // 2
        quads = [
            ("Top-Left", {(x, y) for x in range(hw) for y in range(hh)}),
            ("Top-Right", {(x, y) for x in range(hw, self.app_state.grid_w) for y in range(hh)}),
            ("Bot-Left", {(x, y) for x in range(hw) for y in range(hh, self.app_state.grid_h)}),
            ("Bot-Right", {(x, y) for x in range(hw, self.app_state.grid_w) for y in range(hh, self.app_state.grid_h)})
        ]
        for i, (name, cells) in enumerate(quads):
            zone = Zone(name, i + 1, Theme.ZONE_COLORS[i], cells)
            self.zones.append(zone)
            self._add_zone_widget(zone)
        self._rebuild_cell_map()
        self._log("Created quadrants", "success")

    def _create_rows(self):
        self._clear_zones()
        for y in range(self.app_state.grid_h):
            cells = {(x, y) for x in range(self.app_state.grid_w)}
            zone = Zone(f"Row {y + 1}", (y % 16) + 1,
                        Theme.ZONE_COLORS[y % len(Theme.ZONE_COLORS)], cells)
            self.zones.append(zone)
            self._add_zone_widget(zone)
        self._rebuild_cell_map()
        self._log("Created rows", "success")

    def _create_halves(self):
        self._clear_zones()
        hw = self.app_state.grid_w // 2
        halves = [
            ("Left", {(x, y) for x in range(hw) for y in range(self.app_state.grid_h)}, Theme.ZONE_COLORS[0]),
            ("Right", {(x, y) for x in range(hw, self.app_state.grid_w) for y in range(self.app_state.grid_h)}, Theme.ZONE_COLORS[2])
        ]
        for i, (name, cells, col) in enumerate(halves):
            zone = Zone(name, i + 1, col, cells)
            self.zones.append(zone)
            self._add_zone_widget(zone)
        self._rebuild_cell_map()
        self._log("Created halves", "success")

    def _delete_zone(self, zone: Zone, widget):
        if zone in self.zones:
            self.zones.remove(zone)
        widget.destroy()
        self._rebuild_cell_map()

    def _clear_zones(self):
        self.zones.clear()
        self.cell_map.clear()
        for w in self.zones_scroll.winfo_children():
            w.destroy()

    def _save_settings(self, path):
        if path is None:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.app_state.layout_sash_h1 = self.h_paned.sash_coord(0)[0]
            self.app_state.layout_sash_h2 = self.h_paned.sash_coord(1)[0]
            self.app_state.layout_sash_v = self.v_paned.sash_coord(0)[1]
        except:
            pass
        data = {"state": self.app_state.to_dict(), "zones": [z.to_dict() for z in self.zones]}
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self._log(f"Saved {os.path.basename(path)}", "success")
        except Exception as e:
            self._log(f"Save failed: {e}", "error")

    def _load_settings(self, path):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.app_state = AppState.from_dict(data.get("state", data))

            self.port_var.set(self.app_state.midi_port_name)
            self.midi.set_port(self.app_state.midi_port_name)

            # UI Updates
            self.scale_dropdown.set(self.app_state.scale_name)
            self.root_var.set(self.app_state.root_key)
            self.oct_var.set(str(self.app_state.octave))
            self.dir_dropdown.set(self.app_state.mapping_direction)
            self.grid_w_slider.set(self.app_state.grid_w)
            self.grid_h_slider.set(self.app_state.grid_h)
            self.motion_dropdown.set(self.app_state.motion_mode)
            self.sens_slider.set(self.app_state.sensitivity)
            self.gain_slider.set(self.app_state.motion_gain)
            self.smooth_slider.set(self.app_state.motion_smoothing)
            self.analysis_slider.set(self.app_state.analysis_width)
            self.vel_min_slider.set(self.app_state.min_velocity)
            self.vel_max_slider.set(self.app_state.max_velocity)
            self.curve_dropdown.set(self.app_state.velocity_curve)
            self.bright_slider.set(self.app_state.brightness)
            self.contrast_slider.set(self.app_state.contrast)
            self.blur_slider.set(self.app_state.video_blur_amount)

            if self.app_state.show_grid:
                self.grid_switch.select()
            else:
                self.grid_switch.deselect()
            if self.app_state.show_notes:
                self.notes_switch.select()
            else:
                self.notes_switch.deselect()
            if self.app_state.pixelate_view:
                self.pixelate_switch.select()
            else:
                self.pixelate_switch.deselect()
            if self.app_state.show_flow_vectors:
                self.flow_switch.select()
            else:
                self.flow_switch.deselect()
            if self.app_state.log_midi_signals:
                self.log_midi_switch.select()
            else:
                self.log_midi_switch.deselect()
            if self.app_state.manual_trigger_mode:
                self.manual_trigger_switch.select()
            else:
                self.manual_trigger_switch.deselect()

            self._clear_zones()
            for z in data.get("zones", []):
                zone = Zone.from_dict(z)
                self.zones.append(zone)
                self._add_zone_widget(zone)
            self._rebuild_cell_map()
            self._refresh_drone_widgets()
            
            # Update analyzer
            self.analyzer.set_analysis_width(self.app_state.analysis_width)
            
            self.after(200, self._restore_layout)
            self._reset_buffers()
            self._log(f"Loaded {os.path.basename(path)}", "success")
        except Exception as e:
            self._log(f"Load failed: {e}", "error")

    def _restore_layout(self):
        try:
            if self.app_state.layout_sash_h1 > 0:
                self.h_paned.sash_place(0, self.app_state.layout_sash_h1, 0)
            if self.app_state.layout_sash_h2 > 0:
                self.h_paned.sash_place(1, self.app_state.layout_sash_h2, 0)
            if self.app_state.layout_sash_v > 0:
                self.v_paned.sash_place(0, 0, self.app_state.layout_sash_v)
        except:
            pass

    def _log(self, msg, level="info"):
        if not hasattr(self, 'log_box'):
            self._early_logs.append((msg, level))
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        print(f"[{level.upper()}] {msg}")

    def _get_root_midi(self, zone=None):
        root = zone.root_override if zone and zone.root_override else self.app_state.root_key
        octave = zone.octave_override if zone and zone.octave_override is not None else self.app_state.octave
        return (octave + 1) * 12 + (NOTE_NAMES.index(root) if root in NOTE_NAMES else 0)

    def _wrap_pitch_to_range(self, pitch, zone):
        low, high = zone.note_low, zone.note_high
        rng = high - low + 1
        if pitch < low:
            return high - (low - pitch) % rng + 1 if (low - pitch) % rng > 0 else low
        elif pitch > high:
            return low + (pitch - high - 1) % rng
        return pitch

    def _get_scale(self, zone=None):
        s_name = zone.scale_override if zone and zone.scale_override else self.app_state.scale_name
        return SCALES.get(s_name, SCALES["Minor Pentatonic"])

    def _get_threshold(self):
        # Extended range: 1-200, with finer control at high sensitivity
        # At sens=100: threshold=0.08, at sens=200: threshold≈0
        return (201 - self.app_state.sensitivity) / 200.0 * 8.0

    def _start_cell_anim(self, x: int, y: int, target: float, now: float):
        """Start an eased animation for a cell if target changed."""
        current_target = self.vis_anim_to[y, x]
        # Only start new animation if target actually changed (with small epsilon)
        if abs(target - current_target) > 0.01:
            self.vis_anim_from[y, x] = self.vis_current_vals[y, x]
            self.vis_anim_to[y, x] = target
            # Start slightly in the past so first frame shows ~20% progress (immediate feedback)
            self.vis_anim_start_times[y, x] = now - (VISUAL_RAMP_TIME * 0.2)

    def _update_cell_anim(self, x: int, y: int, now: float):
        """Update cell animation with ease-in-out interpolation."""
        start_time = self.vis_anim_start_times[y, x]
        
        # No animation running
        if start_time <= 0:
            return
        
        elapsed = now - start_time
        
        if elapsed >= VISUAL_RAMP_TIME:
            # Animation complete - snap to target
            self.vis_current_vals[y, x] = self.vis_anim_to[y, x]
            # Clear start time to indicate animation done
            self.vis_anim_start_times[y, x] = 0
        elif elapsed > 0:
            # Calculate eased progress (quadratic ease-in-out)
            progress = min(1.0, elapsed / VISUAL_RAMP_TIME)
            if progress < 0.5:
                eased = 2.0 * progress * progress
            else:
                eased = 1.0 - ((-2.0 * progress + 2.0) ** 2) / 2.0
            
            from_val = self.vis_anim_from[y, x]
            to_val = self.vis_anim_to[y, x]
            self.vis_current_vals[y, x] = from_val + (to_val - from_val) * eased

    def _process_drones(self, now):
        if not self.app_state.midi_active:
            for drone in self.app_state.drones:
                if drone.state == "PLAYING" and drone.current_pitch != -1:
                    self.midi.send_raw_note(drone.current_pitch, 0, drone.channel - 1, False)
                    drone.state = "IDLE"
                    drone.current_pitch = -1
            return

        for drone in self.app_state.drones:
            target_pitch = self._calc_drone_pitch(drone)

            if not drone.active:
                if drone.state == "PLAYING" and drone.current_pitch != -1:
                    self.midi.send_raw_note(drone.current_pitch, 0, drone.channel - 1, False)
                    drone.state = "IDLE"
                    drone.current_pitch = -1
                continue

            if drone.state == "PLAYING" and drone.current_pitch != -1 and drone.current_pitch != target_pitch:
                self.midi.send_raw_note(drone.current_pitch, 0, drone.channel - 1, False)
                self.midi.send_raw_note(target_pitch, drone.velocity, drone.channel - 1, True)
                drone.current_pitch = target_pitch

            if drone.state == "IDLE":
                self.midi.send_raw_note(target_pitch, drone.velocity, drone.channel - 1, True)
                drone.current_pitch = target_pitch
                drone.state = "PLAYING"
                drone.last_action_time = now

            elif drone.state == "PLAYING":
                if drone.mode == "Loop":
                    if now - drone.last_action_time > drone.duration:
                        self.midi.send_raw_note(drone.current_pitch, 0, drone.channel - 1, False)
                        drone.current_pitch = -1
                        drone.state = "WAITING"
                        drone.last_action_time = now

            elif drone.state == "WAITING":
                if now - drone.last_action_time > drone.pause:
                    self.midi.send_raw_note(target_pitch, drone.velocity, drone.channel - 1, True)
                    drone.current_pitch = target_pitch
                    drone.state = "PLAYING"
                    drone.last_action_time = now

    def _calc_drone_pitch(self, drone):
        idx = NOTE_NAMES.index(drone.note) if drone.note in NOTE_NAMES else 0
        return (drone.octave + 1) * 12 + idx

    def _update_loop(self):
        """
        OPTIMIZATION: Main UI loop now only handles display.
        Heavy processing is done in the video processor thread.
        """
        self.midi.process_note_offs(log_enabled=self.app_state.log_midi_signals)

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * (1.0 / max(dt, 0.001))
        self.status_bar.set_fps(self.fps_smooth)

        self._process_drones(now)

        # OPTIMIZATION: Get processed frame from thread queue
        result = None
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        if result is None:
            self.after(16, self._update_loop)
            return

        frame = result['frame']
        global_speeds = result['speeds']
        
        # Apply motion gain/amplification
        if global_speeds is not None:
            global_speeds = global_speeds * self.app_state.motion_gain
        
        self.last_frame_bgr = frame.copy()
        tw, th = max(10, self.video_label.winfo_width()), max(10, self.video_label.winfo_height())
        frame_disp = cv2.resize(frame, (tw, th))
        fh, fw = frame_disp.shape[:2]

        background = frame_disp.copy()
        if self.app_state.pixelate_view:
            tiny = cv2.resize(background, (self.app_state.grid_w, self.app_state.grid_h), interpolation=cv2.INTER_AREA)
            background = cv2.resize(tiny, (fw, fh), interpolation=cv2.INTER_NEAREST)

        if self.app_state.video_blur_amount > 0:
            k = self.app_state.video_blur_amount * 2 + 1
            background = cv2.GaussianBlur(background, (k, k), 0)

        overlay = background.copy()

        threshold = self._get_threshold()
        cw, ch = fw // self.app_state.grid_w, fh // self.app_state.grid_h

        # Ensure buffers match grid size
        if self.vis_end_times.shape != (self.app_state.grid_h, self.app_state.grid_w):
            self._reset_buffers()

        # Process grid cells
        for y in range(self.app_state.grid_h):
            for x in range(self.app_state.grid_w):
                zone = self.cell_map.get((x, y))

                speed = 0.0
                if global_speeds is not None and global_speeds.shape == (self.app_state.grid_h, self.app_state.grid_w):
                    if zone and zone.use_custom_color:
                        speed = self.analyzer.calculate_cell_color_motion(frame_disp, x, y, zone.track_hsv)
                    else:
                        speed = global_speeds[y, x]

                if zone and zone.zone_type == "CC" and not zone.mute and self.app_state.midi_active:
                    raw_val = min(127, (speed / 10.0) * 127 * 3)
                    alpha = 1.0 - (zone.smoothing / 100.0 * 0.95)
                    if zone.last_cc_value < 0:
                        zone.last_cc_value = 0
                    current_cc = zone.last_cc_value * (1 - alpha) + raw_val * alpha
                    zone.last_cc_value = current_cc
                    min_v = zone.vel_min if zone.vel_min is not None else 0
                    max_v = zone.vel_max if zone.vel_max is not None else 127
                    final_cc = int(min_v + (max_v - min_v) * (current_cc / 127.0))
                    final_cc = max(0, min(127, final_cc))
                    self.midi.send_cc(zone.cc_number, final_cc, zone.channel - 1)
                    new_target = (final_cc / 127.0) * 0.8
                    self._start_cell_anim(x, y, new_target, now)
                else:
                    smooth_cd = 0.1 + (zone.smoothing / 100.0) * 1.9 if zone and zone.use_smoothing else COOLDOWN_TIME
                    duration_sec = (zone.duration / 1000.0) if zone else (DEFAULT_NOTE_DURATION_MS / 1000.0)
                    effective_cd = max(smooth_cd, duration_sec)

                    if speed > threshold and (now - self.grid_timers[y, x] > effective_cd):
                        self.grid_timers[y, x] = now
                        if zone and not zone.mute and self.app_state.midi_active:
                            if not zone.use_probability or random.randint(1, 100) <= zone.probability:
                                map_override = zone.mapping_override if zone.mapping_override else self.app_state.mapping_direction
                                pitch = self.pitch_calc.calculate(x, y, self._get_root_midi(zone),
                                                                  self._get_scale(zone), map_override, 0)
                                pitch = self._wrap_pitch_to_range(pitch, zone)

                                min_v = zone.vel_min or self.app_state.min_velocity
                                max_v = zone.vel_max or self.app_state.max_velocity
                                if min_v > max_v:
                                    min_v, max_v = max_v, min_v

                                if zone.use_random_velocity:
                                    vel = VelocityMapper.random_between(min_v, max_v)
                                else:
                                    vel = VelocityMapper.map(speed, threshold, min_v, max_v,
                                                             VelocityCurve(self.app_state.velocity_curve))

                                self.midi.send_note_on(pitch, vel, zone.channel - 1, duration=duration_sec)
                                self.vis_end_times[y, x] = now + duration_sec
                                new_target = 0.3 + (0.6 * (vel / 127.0))
                                self._start_cell_anim(x, y, new_target, now)

                                if self.app_state.log_midi_signals:
                                    self._log(f"♪ {zone.name} N{pitch} v{vel}", "midi")

                    if zone:
                        if now >= self.vis_end_times[y, x]:
                            self._start_cell_anim(x, y, ZONE_ALPHA, now)
                    else:
                        if speed > threshold:
                            self._start_cell_anim(x, y, 0.5, now)
                        else:
                            self._start_cell_anim(x, y, 0.0, now)

                # Eased visual interpolation
                self._update_cell_anim(x, y, now)

                # Draw cell
                x1, y1 = int(x * cw), int(y * ch)
                x2, y2 = int((x + 1) * cw - 1), int((y + 1) * ch - 1)
                current = self.vis_current_vals[y, x]

                if zone or current > 0.01:
                    roi = overlay[y1:y2 + 1, x1:x2 + 1]
                    if roi.size > 0:
                        color = zone.color_bgr if zone else (255, 255, 255)
                        colored = np.full_like(roi, color)
                        cv2.addWeighted(colored, current, roi, 1 - current, 0, roi)

                if (x, y) in self.selected_cells or (x, y) in self.drag_cells:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

                if zone and self.app_state.show_notes and self.app_state.grid_w <= 12 and ch > 20:
                    if zone.zone_type == "Note":
                        map_override = zone.mapping_override if zone.mapping_override else self.app_state.mapping_direction
                        p = self._wrap_pitch_to_range(
                            self.pitch_calc.calculate(x, y, self._get_root_midi(zone),
                                                      self._get_scale(zone), map_override), zone)
                        label = f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"
                    else:
                        label = f"CC{zone.cc_number}"
                    cv2.putText(overlay, label, (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.32, (255, 255, 255), 1, cv2.LINE_AA)

        if self.app_state.show_flow_vectors:
            self.analyzer.draw_flow_vectors(overlay, scale=3.0 * self.app_state.motion_gain)

        if self.app_state.show_grid and not self.app_state.pixelate_view:
            for i in range(1, self.app_state.grid_w):
                cv2.line(overlay, (i * cw, 0), (i * cw, fh), (40, 40, 40), 1)
            for i in range(1, self.app_state.grid_h):
                cv2.line(overlay, (0, i * ch), (fw, i * ch), (40, 40, 40), 1)

        img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.after(16, self._update_loop)

    def _on_close(self):
        # Stop video processor thread
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor.join(timeout=1.0)
        self.midi.close()
        self.destroy()


if __name__ == "__main__":
    app = LavaMIDIApp()
    app.mainloop()
