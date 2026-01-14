# fnuidesktop-vr

Virtual Desktop alternative for Linux/Wayland. Mirrors your desktop to a SteamVR overlay with controller-based mouse input, scrolling, and grab-to-move/pinch-to-resize interaction.

## Features

- Screen capture via XDG Desktop Portal + PipeWire (Wayland native)
- 60fps desktop mirroring to SteamVR overlay
- Controller interaction:
  - **Trigger**: Left click
  - **Thumbstick click**: Right click
  - **Thumbstick up/down**: Scroll (speed follows tilt amount)
  - **Grip (single)**: Drag overlay (follows controller rotation)
  - **Grip (both)**: Pinch to resize
  - **B button (right controller)**: Toggle overlay visibility

## Requirements

- SteamVR
- Python 3.10+
- Wayland compositor with XDG Desktop Portal support

### Python packages

```
pip install openvr numpy PyOpenGL glfw pynput pycairo PyGObject
```

### System packages (Arch)

```
pacman -S python-gobject gst-plugins-base gst-plugin-pipewire
```

## Usage

```
python main.py
```

A screen picker dialog will appear. Select the screen/window to capture, then put on your headset.

Press Ctrl+C to exit.

## Files

- `main.py` - Launcher (runs capture and display as separate processes)
- `test_wayland_capture.py` - Wayland screen capture via Portal/PipeWire
- `vr_display.py` - SteamVR overlay with controller interaction
