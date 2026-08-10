# Changelog - RetroLens Pro

All notable changes to this project will be documented in this file.

## [2.0.0 Pro Architecture] - 2026-08-10

### 🚀 Major Breakthroughs & Upgrades
- **MediaPipe 0.10+ Tasks Engine Integration**: Replaced legacy `mp.solutions.hands` API with high-performance `HandLandmarker` Tasks API featuring automatic model provisioning (`hand_landmarker.task`).
- **Uncapped 120 FPS Rendering**: Optimized camera capture pipeline and frame loop for maximum throughput up to 120 FPS.
- **Smart Standing Finger Filtering**: Added `is_finger_extended` detection algorithm. Folded fingers are automatically excluded from portal calculations.
- **Counter-Clockwise (CCW) Polar Angle Polygon Sorting**: Polygon vertices are sorted counter-clockwise around the centroid `(cx, cy)`, mathematically guaranteeing zero self-intersection or folded geometry.
- **Exponential Moving Average (EMA) Landmark Smoothing**: Smooths hand landmark positions across frames to eliminate pixel jitter.
- **Distance-Invariant Normalized Pinch Detection**: Pinch gesture threshold scales dynamically with relative hand size.
- **16 Premium Visual Filters**: Included Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon, Dual-Tone, Thermal, Sketch, Pixelate, Glitch, Sepia, Invert, Red Channel, Edge Canny, and Gaussian Blur.
- **Futuristic Glassmorphism HUD & Toast System**: Added real-time floating status badges and toast notifications.
- **Real-time MP4 Video Recording**: Added video recording with live duration display (`R` key).
- **One-Click Desktop Launcher**: Added automated PowerShell Desktop shortcut generator (`.lnk`) and `.bat` launcher.
- **Automated Unit Test Suite**: Added 8 comprehensive test cases in `test_retrolens.py`.
