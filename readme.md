# 🌋 Lava MIDI Architect

**Transform motion into music.** Lava MIDI analyzes video input (webcam or file) and converts detected movement into MIDI signals, letting you control synthesizers, DAWs, and any MIDI-compatible software with gestures, dance, or any visual motion.

![Version](https://img.shields.io/badge/version-v61-purple)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **Multiple Motion Detection Modes**
  - Optical Flow (precise movement tracking)
  - Frame Difference (fast, CPU-friendly)
  - Color Tracking (follow specific colors)

- **Flexible Zone System**
  - Create custom trigger zones on the video grid
  - Per-zone MIDI channels, scales, velocity ranges
  - Note mode (triggers) or CC mode (continuous control)
  - Probability and smoothing per zone

- **Musical Intelligence**
  - 12 built-in scales (Pentatonic, Blues, Dorian, etc.)
  - Configurable root note and octave
  - Multiple pitch mapping directions
  - Note range clamping with octave wrapping

- **Drone Generator**
  - 6 independent drone slots
  - Loop or Hold modes
  - Perfect for ambient pads and backgrounds

- **Performance Optimized** ⚡
  - Multithreaded video processing
  - Numba JIT compilation support
  - Configurable analysis resolution

---

## 📋 Prerequisites

### Python Requirements

- Python 3.9 or higher
- pip (Python package manager)

### Virtual MIDI Port Setup

Lava MIDI needs a virtual MIDI port to send signals to your DAW or synth software.

#### 🪟 Windows: loopMIDI

1. Download **loopMIDI** from [Tobias Erichsen's website](https://www.tobias-erichsen.de/software/loopmidi.html)
2. Install and run loopMIDI
3. Click the **+** button to create a new virtual port
4. Name it `LavaPort` (or any name you prefer)
5. The port will appear in Lava MIDI's output dropdown

> 💡 loopMIDI runs in the system tray. Make sure it's running before starting Lava MIDI.

#### 🍎 macOS: IAC Driver (Built-in)

macOS has a built-in virtual MIDI driver called **IAC Driver**:

1. Open **Audio MIDI Setup** (search in Spotlight or find in `/Applications/Utilities/`)
2. Press `Cmd + 2` or go to **Window → Show MIDI Studio**
3. Double-click the **IAC Driver** icon
4. Check **"Device is online"**
5. Click the **+** button under "Ports" to add a port
6. Name it `LavaPort` (or any name you prefer)
7. Click **Apply**

The IAC port will now appear in Lava MIDI and your DAW.

#### 🐧 Linux: ALSA or JACK

**Option A: ALSA Virtual MIDI**
```bash
sudo modprobe snd-virmidi
```
This creates virtual MIDI ports (`VirMIDI 0-0`, etc.)

**Option B: JACK MIDI**
If you're using JACK:
```bash
# Install a2jmidid for ALSA-JACK MIDI bridging
sudo apt install a2jmidid
a2jmidid -e &
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lava-midi.git
cd lava-midi
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy mido python-rtmidi customtkinter pillow
```

**Optional but recommended** (for better performance):
```bash
pip install numba
```

#### Full requirements:
```
opencv-python>=4.5
numpy>=1.20
mido>=1.2
python-rtmidi>=1.4
customtkinter>=5.0
pillow>=9.0
numba>=0.56  # optional, for JIT acceleration
```

### 3. Run Lava MIDI

```bash
python lava_midi_v61_optimized.py
```

---

## 🎮 Quick Start

1. **Set up your virtual MIDI port** (see Prerequisites above)
2. **Launch Lava MIDI**
3. **Select MIDI Output** — Choose your virtual port from the dropdown in the left sidebar
4. **Create Zones** — Click and drag on the video to select cells, then click "+ From Selection"
5. **Press ▶ Start** — MIDI output is now active
6. **Move!** — Motion in the zones will trigger MIDI notes

### Connecting to a DAW

1. In your DAW (Ableton, FL Studio, Logic, Reaper, etc.), create a new MIDI track
2. Set the track's MIDI input to your virtual port (`LavaPort`)
3. Arm the track for recording
4. Load a software instrument
5. Start making music with motion!

---

## 🎛️ Interface Overview

```
┌─────────────┬────────────────────────────┬──────────────────┐
│   SIDEBAR   │                            │   ZONE MANAGER   │
│             │                            │                  │
│  • Project  │                            │  • Zone Tools    │
│  • Music    │        VIDEO FEED          │  • Zone List     │
│  • Grid     │       (Motion Grid)        │                  │
│  • Motion   │                            ├──────────────────┤
│  • Velocity │                            │   DRONE PANEL    │
│  • Video    │                            │                  │
│             │                            ├──────────────────┤
│  [MIDI OUT] │                            │      LOG         │
│  [▶ START]  │                            │                  │
└─────────────┴────────────────────────────┴──────────────────┘
```

### Left Sidebar
- **Project** — Save/Load/Export settings
- **Music** — Scale, root note, octave, mapping direction
- **Grid** — Rows and columns (2-16)
- **Motion** — Detection mode, sensitivity, gain, smoothing
- **Velocity** — Min/max velocity and curve
- **Video** — Source, brightness, contrast, blur, overlays

### Zone Manager (Right Panel)
- Create zones from grid selection
- Quick presets: Quadrants, Rows, Halves
- Per-zone settings: channel, scale override, velocity, probability

### Drones Tab
- 6 independent drone generators
- Loop (rhythmic) or Hold (sustain) modes
- Individual note, octave, velocity, channel settings

---

## ⚙️ Configuration Guide

### Motion Detection Modes

| Mode | Best For | CPU Usage |
|------|----------|-----------|
| **Optical Flow** | Precise directional tracking, flow visualization | High |
| **Frame Diff** | General motion, fast response | Low |
| **Color Track** | Following specific colored objects | Medium |

### Sensitivity & Gain

- **Sensitivity (1-200)**: Threshold for triggering notes. Higher = more sensitive.
- **Motion Gain (0.5x-5.0x)**: Amplifies detected motion. Use for subtle movements.

### Velocity Curves

| Curve | Character |
|-------|-----------|
| Linear | Direct 1:1 mapping |
| Exponential | Soft at low motion, strong at high |
| Logarithmic | Strong at low motion, compressed at high |
| S-Curve | Smooth, natural feel |

### Zone Types

- **Note Mode**: Triggers discrete MIDI notes on motion
- **CC Mode**: Sends continuous controller values based on motion intensity

---

## 💾 Saving & Loading

- **💾 Save**: Saves to `settings.json` in the current directory (auto-loads on startup)
- **Export**: Save to a custom location
- **Import**: Load from a custom file

Settings include: all parameters, zones, drone configurations, and window layout.

---

## 🔧 Troubleshooting

### No MIDI ports showing up
- **Windows**: Make sure loopMIDI is running
- **macOS**: Check that IAC Driver is online in Audio MIDI Setup
- **Linux**: Load the `snd-virmidi` kernel module

### High CPU usage
- Switch from "Optical Flow" to "Frame Diff" mode
- Lower the "Analysis Res" slider (60-100 is usually sufficient)
- Reduce grid size

### Notes not triggering
- Check that MIDI is started (button shows "⏹ Stop")
- Lower the Sensitivity or increase Motion Gain
- Ensure zones are not muted (M button)
- Verify the MIDI output is selected

### Video not loading
- Supported formats: MP4, AVI, MOV, MKV
- Try a different video codec (H.264 recommended)
- Check that OpenCV is properly installed

### Numba warnings on startup
- This is normal if Numba isn't installed
- Install with: `pip install numba`
- The app works fine without it, just slightly slower

---

## 🎵 Creative Ideas

- **Dance Performance**: Map full-body movements to drums and synths
- **Hand Tracking**: Use zones for left/right hand to play different instruments
- **VJ Integration**: Process video loops to generate synchronized MIDI
- **Installation Art**: Create interactive sound sculptures
- **Live Coding Visuals**: Feed generative visuals to trigger evolving soundscapes
- **Conducting**: Control orchestral samples with gesture dynamics

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built with [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern UI
- MIDI handling via [mido](https://github.com/mido/mido) and [python-rtmidi](https://github.com/SpotlightKid/python-rtmidi)
- Motion analysis powered by [OpenCV](https://opencv.org/)

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  <b>Made with 🔥 for musicians, performers, and creative coders</b>
</p>
