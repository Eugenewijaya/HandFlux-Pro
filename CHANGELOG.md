# Changelog - HandFlux Pro

All notable changes to this project will be documented in this file.

## [3.0.0 Pro Y2K & Indie Sleaze Edition] - 2026-08-11

### 🚀 Major Upgrades & New Features
- **Y2K Pop-Art & Indie Sleaze Special Effects Theme**: Introduced 5 high-fashion filters (`red-halftone`, `indie-flash`, `y2k-lime-doodle`, `pink-starburst`, `pink-halo-dots`) inspired by Y2K magazine halftone print and indie sleaze flash photography aesthetics.
- **1-to-1 Un-Distorted Alignment (No Face Warping/Stretching)**: Removed perspective warping from portal rendering so the image inside the portal aligns 100% 1-to-1 with the background without face deformation or pixel stretching.
- **Adaptive Flexible Quad Portal (`make_flexible_quad`)**: Dynamic segment-intersection algorithm that prevents bowtie folding and keeps the 4-corner quad frame flexible when hands twist, slant, or rotate ("saat dipelintir").
- **Touch-to-Activate Skeleton Portal**: Portal box activates when the two hand skeletons touch or come close (< 160px), then stretches and expands as hands are pulled apart.
- **Ultra-Sensitive 1.0x Full-Scale Hand Tracking**: Upgraded to 1.0x full-resolution MediaPipe detection with zero frame-skipping (`_detect_every = 1`) and tuned confidence thresholds (`0.30`) to reliably detect hands whether 3 fingers are curled or extended.
- **Streamlined Filter Engine**: Purged slow/legacy filters into 4 high-speed SIMD/Vectorized categories (`Y2K POP-ART`, `SPECIAL FX`, `CYBER`, `TACTICAL`), removing default-cam from portal rotation.

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
- **Automated Unit Test Suite**: Added 8 comprehensive test cases in `test_handflux.py`.
