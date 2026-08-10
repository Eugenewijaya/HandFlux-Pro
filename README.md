<div align="center">

# 🖐️⚡ HANDFLUX PRO
### *Interactive Real-Time Hand-Gesture Visual Portal Engine*

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3ZtYnhuZmpubWxkcTNpaTR5ZnptZnJndmtwNDdxcWRlMGdncXZmdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/13l74g92vuYuUU/giphy.gif" alt="HandFlux Pro Anime Gesture" width="480"/>
</p>

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
**HandFlux Pro** adalah ekosistem aplikasi *Computer Vision* interaktif berbasis Python, OpenCV, dan MediaPipe. Repositori ini memiliki 3 aplikasi utama yang terpisah secara spesifik:

1. **`HandFlux.py` — Real-Time Visual Filter Portal Engine (Fokus Utama Filter)**:
   Aplikasi portal visual murni dengan **40 filter visual premium** yang dikategorikan ke dalam 5 tema (*Cinematic, Anime, Cyberpunk, Artistic, Exotic*), **Auto-Cycle 2s Mode**, deteksi jari berdiri, dan performa tinggi 120 FPS.
2. **`naruto_jutsu.py` — Standalone Naruto Ninjutsu Camera Engine (Fokus Jurus Anime)**:
   Aplikasi khusus jurus Ninjutsu interaktif yang memanfaatkan integrasi Kaggle Model (`menhari/naruto-hand-guesture-rasengan`). Rilis **Rasengan 🌀** dengan pita chakra 3D & ledakan Blast 💥 via **Tepuk Tangan (CLAP 👏)**, **Katon Fireball 🔥**, **Chidori ⚡**, dan **Shadow Clone 👥**!
3. **`foto_kita_blur.py` — Standalone Foto Kita Blur Camera Engine (Fokus Romantis V-Sign)**:
   Aplikasi khusus pendeteksi gestur tangan "V" (Peace Sign ✌️) yang mengaktifkan **Efek Screen Blur 40%** dan animasi **Balon Hati / Love Emoticons** melayang ke atas, serta menyimpan foto secara privat di folder lokal `foto kita blurr/`.

---

### 🎨 40 Filter Visual & Kategori Tema (`HandFlux.py`)

1. **🎬 Tema CINEMATIC & FILM (8 Filter)**:
   `Teal-Orange`, `Kodachrome`, `Technicolor`, `Noir-Film`, `Cinematic-Warm`, `Vignette-Cinema`, `Sepia-Vintage`, `Detail-Enhance (HDR)`.
2. **⛩️ Tema ANIME & CARTOON (8 Filter)**:
   `Anime-Cel`, `Manga-Ink`, `Cartoon-Classic`, `Pop-Art`, `Pencil-Sketch`, `Pencil-Color`, `Stylized-Water`, `Posterize`.
3. **⚡ Tema CYBER & SCI-FI (8 Filter)**:
   `Cyberpunk`, `Matrix`, `Thermal`, `Night-Vision`, `Hologram`, `Glitch-RGB`, `Anaglyph-3D`, `Emboss-3D`.
4. **🎨 Tema ARTISTIC & EFX (8 Filter)**:
   `Oil-Paint`, `Rainbow-Wave`, `Edge-Neon`, `Pixelate`, `VHS-Tape`, `Solarize`, `Duotone-Cyan`, `Cross-Process`.
5. **🌀 Tema EXOTIC (8 Filter)**:
   `Pixel-Sort`, `Kaleidoscope`, `Water-Ripple`, `Frosted-Glass`, `CRT-Screen`, `Aurora-Gradient`, `Diamond-Mosaic`, `Dream-Glow`.

---

### 🔥 Kenapa HandFlux Pro Jauh Lebih Keren?

| Fitur & Keunggulan | Kenapa Ini Lebih Keren? 🚀 |
| --- | --- |
| **🎨 40 Filter Visual Premium** | Koleksi 40 filter spektakuler dari gaya film sinematik, anime cel shading, cyberpunk neon, pixel sort glitch art, hingga efek 3D stereo. |
| **🔄 Auto-Cycle Mode 2 Detik (`A`)** | Filter otomatis berganti setiap 2 detik secara acak tanpa perlu melakukan gestur cubit! |
| **🏷️ Theme Category Switcher (`T`)** | Pilih kategori tema khusus (Cinematic, Anime, Cyber, Artistic, Exotic) atau jalankan seluruh filter secara acak. |
| **🍃 Standalone Naruto Ninjutsu Engine** | Aplikasi terpisah `naruto_jutsu.py` khusus jurus **Rasengan 🌀**, **Fire Style 🔥**, **Chidori ⚡**, dan **Shadow Clone 👥**! |
| **❤️ Standalone Foto Kita Blur Engine** | Aplikasi terpisah `foto_kita_blur.py` khusus **Love Blur 40%** & **Balon Hati 🎈** via gestur V (✌️)! |
| **🖐️ Smart Standing Finger Filter** | Hanya mendeteksi dan menggunakan jari yang **berdiri tegak**. Jari yang dilipat ke telapak tangan otomatis diabaikan! |
| **📐 CCW Polar Angle Sorting** | Poligon portal diurutkan berlawanan arah jarum jam secara matematis. **Portal 100% TIDAK AKAN KELIPET ATAU MENYILANG**! |
| **✨ Velocity-Predicting Tracker** | Pelacak gerakan tangan berbasis *Velocity Prediction & Ghost Frames (12 frames)* untuk pergerakan portal mulus tanpa flicker. |
| **🖥️ Glassmorphic HUD & Toasts** | Tampilan antarmuka transparan futuristik dilengkapi sistem notifikasi mengambang (*floating toast system*). |
| **🖱️ One-Click Desktop Launcher** | Dilengkapi file batch & shortcut instan untuk ketiga aplikasi di Windows. |
| **⚡ High Frame-Rate Throughput** | Didukung optimasi buffer kamera real-time yang mampu membuka performa hingga 120 FPS. |

---

### 🎮 Tabel Perintah & Eksekusi

| Aplikasi / Perintah | Fungsi & Cara Pakai |
| --- | --- |
| 🖼️ **Filter Portal (`HandFlux.py`)** | `python HandFlux.py` (atau `python HandFlux.py --auto-cycle`) |
| 🍃 **Naruto Ninjutsu (`naruto_jutsu.py`)** | `python naruto_jutsu.py` (atau klik `Run Naruto Jutsu.bat`) |
| ❤️ **Foto Kita Blur (`foto_kita_blur.py`)** | `python foto_kita_blur.py` (atau klik `Run Foto Kita Blur.bat`) |
| 🔄 **Tombol `A`** | Toggle Auto-Cycle Mode (Filter ganti otomatis per 2s) |
| 🏷️ **Tombol `T`** | Switch Kategori Tema Filter |
| 🔢 **Tombol `1` - `5`** | Jumlah Jari Aktif Portal (1=Lingkaran, ..., 5=Hand Polygon) |
| 👏 **CLAP Hands (Tepuk Tangan)** | Rilis **Rasengan 🌀** pada `naruto_jutsu.py` |

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

3. **Jalankan Aplikasi Filter Portal:**
   ```bash
   python HandFlux.py
   ```

4. **Jalankan Aplikasi Naruto Ninjutsu Camera (Terpisah):**
   ```bash
   python naruto_jutsu.py
   ```

4. **Uji Coba Automated Test:**
   ```bash
   python test_handflux.py
   ```

---

<br/>

## 🇬🇧 English

### 📖 About HandFlux Pro
**HandFlux Pro** is an interactive Computer Vision application powered by Python, OpenCV, and MediaPipe Tasks API. It dynamically transforms your hand gestures into interactive visual filter portals in real-time.

Featuring **32 premium visual filters** categorized into 4 themes (*Cinematic, Anime, Cyberpunk, Artistic*) and an **Auto-Cycle 2s Mode** that automatically rotates filters every 2 seconds without requiring gestures.

---

### 🎨 32 Themed Visual Filters

- **🎬 CINEMATIC**: `Teal-Orange`, `Kodachrome`, `Technicolor`, `Noir-Film`, `Cinematic-Warm`, `Vignette-Cinema`, `Sepia-Vintage`, `Detail-Enhance (HDR)`.
- **⛩️ ANIME & CARTOON**: `Anime-Cel`, `Manga-Ink`, `Cartoon-Classic`, `Pop-Art`, `Pencil-Sketch`, `Pencil-Color`, `Stylized-Water`, `Posterize`.
- **⚡ CYBER & SCI-FI**: `Cyberpunk`, `Matrix`, `Thermal`, `Night-Vision`, `Hologram`, `Glitch-RGB`, `Anaglyph-3D`, `Emboss-3D`.
- **🎨 ARTISTIC & EFX**: `Oil-Paint`, `Rainbow-Wave`, `Edge-Neon`, `Pixelate`, `VHS-Tape`, `Solarize`, `Duotone-Cyan`, `Cross-Process`.

---

### 🎮 Controls & Shortcuts

| Key / Gesture | Function |
| --- | --- |
| 🔄 **Key `A`** | **Toggle Auto-Cycle 2s Mode** (Auto switch filter every 2 seconds) |
| 🏷️ **Key `T`** | **Switch Filter Theme** (`ALL` ➔ `CINEMATIC` ➔ `ANIME` ➔ `CYBER` ➔ `ARTISTIC`) |
| 🤏 **Pinch (Thumb + Index)** | Manually cycle visual filters |
| 🔢 **Keys `1` to `5`** | Set active finger count (1=Circle, 2=Pill, 3=Triangle, 4=Quad, 5=Polygon) |
| 📸 **Key `S`** | Take manual screenshot (saved to `captures/`) |
| 🎥 **Key `R`** | Start / stop MP4 video recording |
| 🚪 **Key `Q` / `Esc`** | Quit application |

---

<br/>

## 🇨🇳 中文 (Chinese)

### 📖 关于 HandFlux Pro
**HandFlux Pro** 是一款基于 Python、OpenCV 和 MediaPipe Tasks API 开发的互动式计算机视觉应用。它包含 **32 种主题滤镜**（电影感、动漫风、赛博朋克、艺术涂鸦）以及 **2 秒自动循环模式**。

---

### 🎨 32 种主题滤镜

- **🎬 电影感 (CINEMATIC)**: 好莱坞橙青 (Teal-Orange)、柯达胶卷 (Kodachrome)、彩色印相 (Technicolor)、黑白电影 (Noir-Film)、暖阳 (Warm)、暗角 (Vignette)、复古褐色 (Sepia)、HDR 细节增强 (Detail-Enhance)。
- **⛩️ 动漫与卡通 (ANIME)**: 动漫赛璐珞 (Anime-Cel)、日漫水墨 (Manga-Ink)、经典卡通 (Cartoon)、波普艺术 (Pop-Art)、铅笔素描 (Pencil-Sketch)、彩色铅笔 (Pencil-Color)、水彩画 (Stylized-Water)、海报化 (Posterize)。
- **⚡ 赛博与科影 (CYBER)**: 赛博朋克 (Cyberpunk)、黑客帝国 (Matrix)、红外热成像 (Thermal)、夜视仪 (Night-Vision)、全息投影 (Hologram)、RGB 故障 (Glitch-RGB)、3D 红蓝立体 (Anaglyph-3D)、3D 浮雕 (Emboss-3D)。
- **🎨 艺术与特效 (ARTISTIC)**: 油画 (Oil-Paint)、彩虹波浪 (Rainbow-Wave)、霓虹边缘 (Edge-Neon)、像素化 (Pixelate)、VHS 录像带 (VHS-Tape)、曝光过度 (Solarize)、双色调青紫 (Duotone-Cyan)、LOMO 交叉冲洗 (Cross-Process)。

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
- **Engine Evolution & Upgrade**: Completely rebuilt and upgraded using **Gemini 3.6 Pro** via **Antigravity** engine.

---

<div align="center">

Released under the [MIT License](LICENSE).

</div>
