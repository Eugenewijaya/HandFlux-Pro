# HandFlux Pro 🖐️⚡

**HandFlux Pro** adalah aplikasi *Computer Vision* real-time berbasis Python, OpenCV, dan MediaPipe Tasks API yang mengubah gerakan tangan Anda menjadi portal visual interaktif berkecepatan tinggi hingga **120 FPS**.

Aplikasi ini menggunakan algoritma pemrosesan spasial canggih untuk mendeteksi jari-jari tangan yang berdiri tegak, menyusun geometri poligon tanpa garis menyilang (*non-self-intersecting CCW polygon*), serta mengaplikasikan 16 filter efek visual secara real-time di dalam area portal.

---

## 🇮🇩 Fitur Utama

- **⚡ Uncapped 120 FPS Rendering**: Pengolahan bingkai video ultra-cepat tanpa batas framerate buatan.
- **🖐️ Deteksi Jari Berdiri (Smart Extension Filter)**: Hanya mendeteksi dan menggunakan jari yang sedang berdiri tegak. Jari yang dilipat ke telapak tangan otomatis diabaikan.
- **📐 Geometri CCW Polar Sorting**: Pengurutan titik poligon portal berbasis *Counter-Clockwise Polar Angle* relatif terhadap titik pusat (*centroid*). Dijamin **100% portal tidak pernah terlipat atau menyilang**.
- **✨ Exponential Moving Average (EMA) Smoothing**: Menyaring getaran halus piksel (*landmark jitter*) dari kamera untuk pergerakan portal yang tenang dan mulus.
- **🎯 Pinch Gesture Berjarak Relatif**: Pengenalan gestur cubit (*pinch*) yang dinamis sesuai skala perbandingan ukuran tangan.
- **🎨 16 Filter Visual Premium**: Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon, Thermal, Sketch, Glitch, Dual-Tone, Pixelate, Sepia, dan lainnya.
- **🖥️ Desktop Launcher Ready**: Siap dijalankan langsung via klik 2x shortcut Desktop.
- **📹 Perekaman Video MP4 & Capture Screenshot**: Dukungan mengambil gambar dan merekam video secara langsung.

---

## 🚀 Panduan Memulai

### 1. Prasyarat System
- Python 3.8 hingga Python 3.14+
- Webcam internal atau USB

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Cara Menjalankan

#### A. Menggunakan Shortcut Desktop (Paling Praktis)
Klik 2x ikon **HandFlux Pro** di Desktop Windows Anda.

#### B. Menggunakan Command Line
```bash
python Retrolens.py
```
*Opsi Kustomisasi:*
```bash
python Retrolens.py --fps 120 --width 960 --height 540 --fingers 5
```

#### C. Menjalankan Automated Test
```bash
python test_retrolens.py
```

---

## 🎮 Tabel Kontrol & Gesture

| Tombol / Gesture | Fungsi & Deskripsi |
| --- | --- |
| **Double Click Shortcut** | Jalankan aplikasi langsung dari Desktop |
| **Pinch (Jempol + Telunjuk)** | Berganti filter visual secara real-time |
| **Tombol `1` - `5`** | Pengaturan jumlah jari aktif (1=Lingkaran, 2=Kapsul, 3=Segitiga, 4=Quad, 5=Poligon 5 Sudut) |
| **Tombol `F`** | Melakukan cycle pilihan jumlah jari aktif secara berurutan |
| **Tombol `S`** | Mengambil screenshot manual (disimpan di folder `captures/`) |
| **Tombol `G`** | Toggle gesture screenshot Peace Sign `✌️` (Off secara default) |
| **Tombol `R`** | Memulai / Menghentikan perekaman video MP4 |
| **Tombol `N` / `P`** | Berpindah ke filter berikutnya / sebelumnya |
| **Tombol `C`** | Toggle mode portal 2D Quad vs 3D Mesh |
| **Tombol `M`** | Toggle mode Mirror (cermin) |
| **Tombol `H`** | Menyembunyikan / Menampilkan HUD overlay |
| **Tombol `Q` / `Esc`** | Keluar dari aplikasi |

---

## 🇬🇧 English Overview

**HandFlux Pro** is a state-of-the-art real-time computer vision application powered by Python, OpenCV, and MediaPipe Tasks API. It transforms hand gestures into interactive visual filter portals running at up to **120 FPS**.

### Key Highlights
- **120 FPS Uncapped Throughput**: High-performance camera pipeline.
- **Smart Standing Finger Filter**: Ignores folded fingers into the palm.
- **CCW Polar Angle Sorting**: Guarantees clean, non-crossing polygon portal shapes.
- **EMA Landmark Smoothing**: Eliminates pixel jitter for rock-solid tracking.
- **16 Premium Visual Filters**: Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon, etc.
- **One-Click Desktop Launcher**: Immediate execution via Windows Desktop shortcut.

---

## 📄 Lisensi

Proyek ini dirilis di bawah [MIT License](LICENSE).
