<div align="center">

# 🖐️⚡ HANDFLUX PRO v3.0.0
### *Interactive Real-Time Hand-Gesture Visual Portal Engine*
#### *Y2K Pop-Art & Indie Sleaze High-Fashion Edition*

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3ZtYnhuZmpubWxkcTNpaTR5ZnptZnJndmtwNDdxcWRlMGdncXZmdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/13l74g92vuYuUU/giphy.gif" alt="HandFlux Pro Anime Gesture" width="480"/>
</p>

[![Version](https://img.shields.io/badge/Version-3.0.0-FF6B9D?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Eugenewijaya/HandFlux-Pro)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.14%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20v0.10%2B-00979D?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-F7B500?style=for-the-badge)](LICENSE)

---

**[ 🇮🇩 Bahasa Indonesia ](#-bahasa-indonesia)** &nbsp;|&nbsp; **[ 🇬🇧 English ](#-english)** &nbsp;|&nbsp; **[ 🇨🇳 中文 ](#-中文-chinese)**

---
</div>

<br/>

## 🇮🇩 Bahasa Indonesia

### 📖 Tentang HandFlux Pro
**HandFlux Pro** adalah aplikasi *Computer Vision* interaktif berbasis Python, OpenCV, dan MediaPipe Tasks API. Engine ini mentransformasi gerakan tangan Anda menjadi portal filter visual real-time dengan **18 efek visual premium** dalam 4 tema.

**`HandFlux.py` — Real-Time Visual Filter Portal Engine**:
Portal visual dengan **18 filter premium** (Y2K Pop-Art, Special FX, Cyber, Tactical), deteksi kerangka tangan ultra-presisi, gestur *pinch-to-activate portal*, algoritma `make_flexible_quad` anti-bowtie, dan performa tinggi.

---

### 🎨 18 Special Effects Filter & Kategori Tema (`HandFlux.py`)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXA3YzQzM3E1aTV1cGZocm0xOGdydnJyMXY1aDZscWhsN3g3eHNtbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/C21GGDOpKT6Z4VuXyn/giphy.gif" alt="Anime Domain Expansion Portal" width="480"/>
</p>

1. **📸 Tema Y2K POP-ART (5 Filter)**:
   - `red-halftone`: Cetak koran *duotone halftone dot matrix* merah marun & rose white.
   - `indie-flash`: Kamera malam *Indie Sleaze* dengan *motion streak* & *warm amber flash light leak*.
   - `y2k-lime-doodle`: Majalah Y2K dengan kontur *electric lime green doodle* & gambar petir.
   - `pink-starburst`: B&W kontras tinggi dengan aksen grafis *Neon Magenta Starburst*.
   - `pink-halo-dots`: *Halftone newsprint monokrom* dengan lingkaran *glowing neon pink halo*.
2. **⚡ Tema SPECIAL FX (10 Filter)**:
   - `glitch-cyber`, `broken-tv`, `vhs-retro`, `pixel-8bit`, `chromatic`, `film-grain`, `emboss`, `solarize`, `comic`, `negative`.

3. **💻 Tema CYBER (5 Filter)**:
   - `matrix-rain`, `hologram-cyan`, `edge-neon`, `duotone-cyber`, `cross-process`.

4. **🎯 Tema TACTICAL (3 Filter)**:
   - `thermal-vision`, `night-vision`, `xray-scan`.

---

### 🔥 Keunggulan Utama HandFlux Pro v3.0.0

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzR0MHg3aW5jYWhwbmpzb2RtbHRocmlrdmpjcDdsOHo3Nm8yaWRscSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4lu5FuhtrbaFi/giphy.gif" alt="Anime Hand Sign Gesture Tracking" width="480"/>
</p>

| Fitur & Keunggulan | Kenapa Ini Lebih Keren? 🚀 |
| --- | --- |
| **✨ 1-to-1 Un-Distorted Alignment** | Posisi wajah & tubuh di dalam portal **100% presisi 1-to-1 dengan latar luar** tanpa peyot, tanpa tertarik, dan tanpa distorsi piksel! |
| **🖐️ Touch-to-Activate Portal** | Portal hanya aktif saat **kedua kerangka tangan disentuhkan/didekatkan (<160px)**, lalu ditarik melebar sesuai posisi 4 ujung jari. |
| **🔄 Adaptive Flexible Quad** | Algoritma `make_flexible_quad` mendeteksi posisi persilangan jari secara otomatis. **Portal TIDAK AKAN PERNAH MENYILANG (anti-bowtie)** dan tetap fleksibel saat tangan dipelintir/diputar! |
| **🎯 High-Precision 1.0x Detection** | Pemrosesan deteksi MediaPipe pada **resolusi penuh 1.0x di setiap frame (`_detect_every = 1`)** untuk deteksi ultra-stabil tanpa glitch. |
| **🤏 Pinch to Cycle Filter** | Gestur Pinch (cubit jari telunjuk-jempol) untuk berpindah ke efek visual spesial berikutnya secara *real-time*. |
| **🖥️ Apple Clean Glass HUD** | Antarmuka futuristik glassmorphism bergaya San Francisco Typography dengan *floating toast notifications*. |

---

---

### 🎮 Tabel Perintah & Eksekusi

| Perintah / Tombol | Fungsi & Cara Pakai |
| --- | --- |
| 🖼️ **`python HandFlux.py`** | Jalankan HandFlux Pro v3.0.0 Engine |
| 🏷️ **Tombol `T`** | Berganti Kategori Tema Filter (`ALL` ➔ `Y2K POP-ART` ➔ `SPECIAL FX` ➔ `CYBER` ➔ `TACTICAL`) |
| 🤏 **Gestur Pinch** | Ganti efek filter visual |
| 📸 **Tombol `S`** | Ambil Foto / Screenshot (tersimpan ke `captures/`) |
| 🎥 **Tombol `R`** | Mulai / Hentikan Rekam Video MP4 |
| 🚪 **Tombol `Q` / `Esc`** | Keluar dari Aplikasi |

---

### 🚀 Panduan Instalasi & Memulai

1. **Clone Repositori:**
   ```bash
   git clone https://github.com/Eugenewijaya/HandFlux-Pro.git
   cd HandFlux-Pro
   ```

2. **Pasang Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan HandFlux Pro:**
   ```bash
   python HandFlux.py
   ```

4. **Uji Coba Automated Test:**
   ```bash
   python test_handflux.py
   ```

---

<br/>

## 🇬🇧 English

### 📖 About HandFlux Pro
**HandFlux Pro** is an interactive Computer Vision application powered by Python, OpenCV, and MediaPipe Tasks API. Transform your hand gestures into real-time visual filter portals with **18 premium effects** across 4 curated themes.

---

### 🎨 18 Themed Visual Filters

- **📸 Y2K POP-ART (5)**: `red-halftone`, `indie-flash`, `y2k-lime-doodle`, `pink-starburst`, `pink-halo-dots`.
- **⚡ SPECIAL FX (5)**: `glitch-cyber`, `broken-tv`, `vhs-retro`, `pixel-8bit`, `chromatic`, `film-grain`, `emboss`, `solarize`, `comic`, `negative`.
- **💻 CYBER (5)**: `matrix-rain`, `hologram-cyan`, `edge-neon`, `duotone-cyber`, `cross-process`.
- **🎯 TACTICAL (3)**: `thermal-vision`, `night-vision`, `xray-scan`.

---

### 🎮 Controls & Shortcuts

| Key / Gesture | Function |
| --- | --- |
| 🏷️ **Key `T`** | **Switch Filter Theme** (`ALL` ➔ `Y2K POP-ART` ➔ `SPECIAL FX` ➔ `CYBER` ➔ `TACTICAL`) |
| 🤏 **Pinch (Thumb + Index)** | Cycle visual filters in real-time |
| 🖐️ **Touch both hand skeletons** | Activate the visual portal (then drag to resize) |
| 📸 **Key `S`** | Take manual screenshot (saved to `captures/`) |
| 🎥 **Key `R`** | Start / stop MP4 video recording |
| 🚪 **Key `Q` / `Esc`** | Quit application |

---

<br/>

## 🇨🇳 中文 (Chinese)

### 📖 关于 HandFlux Pro
**HandFlux Pro** 是一款基于 Python、OpenCV 和 MediaPipe Tasks API 开发的互动式计算机视觉应用。它包含 **18 种主题滤镜**（Y2K 流行艺术、特效、赛博、战术）。

---

### 🎨 18 种主题滤镜

- **📸 Y2K 流行艺术 (Y2K POP-ART)**: 红色半色调、独立闪光、Y2K 酸橙涂鸦、粉红星爆、粉红光晕点。
- **⚡ 特效 (SPECIAL FX)**: 故障赛博、破损电视、VHS 复古、8位像素、色差分离、胶片颗粒、浮雕、曝光、漫画、负片。
- **💻 赛博 (CYBER)**: 黑客帝国雨、全息青色、霓虹边缘、双色调赛博、交叉冲洗。
- **🎯 战术 (TACTICAL)**: 热成像视觉、夜视仪、X 射线扫描。

---

<br/>

## 👤 Author & Socials

<div align="left">

Developed with passion by **Evid Wijaya** 🚀

- 📷 **Instagram**: [@epidoey](https://instagram.com/epidoey)
- 💼 **LinkedIn**: [Evid Wijaya](https://www.linkedin.com/in/evid-wijaya/)

</div>

---

## 📜 Credits & Acknowledgments

- **Base Project Idea**: Original credit goes to [`github.com/syahdanfx/Retrolens`](https://github.com/syahdanfx/Retrolens)
- **Engine Evolution & Upgrade**: Completely rebuilt and upgraded using **Claude Sonnet 4.5 / Opus** via **Antigravity** engine by Google DeepMind.

---

<div align="center">

Released under the [MIT License](LICENSE).

</div>
