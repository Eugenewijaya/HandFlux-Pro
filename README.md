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
**HandFlux Pro** adalah aplikasi *Computer Vision* interaktif yang mengubah gerakan tangan Anda menjadi portal efek visual real-time. Cukup bentangkan dua tangan di depan webcam untuk memproyeksikan portal efek futuristik dari **25+ filter visual premium** yang dikategorikan ke dalam 4 tema utama (*Cinematic, Anime, Cyberpunk, Artistic*).

Aplikasi ini dilengkapi dengan mode **Auto-Cycle 2 Detik** (otomatis berganti filter tanpa perlu gesture cubit), pemilihan tema filter dinamis, deteksi jari berdiri yang presisi, serta pengurutan poligon anti-kelipet.

---

### 🎨 25 Filter Visual Berdasarkan Tema

1. **🎬 Tema CINEMATIC & FILM**: `Teal-Orange`, `Kodachrome`, `Technicolor`, `Noir-Film`, `Cinematic-Warm`, `Vignette-Cinema`, `Sepia`.
2. **⛩️ Tema ANIME & CARTOON**: `Anime-Cel`, `Manga-Ink`, `Cartoon-Classic`, `Pop-Art`, `Pencil-Sketch`, `Posterize`.
3. **⚡ Tema CYBER & SCI-FI**: `Cyberpunk`, `Matrix`, `Thermal`, `Night-Vision`, `Hologram`, `Glitch-RGB`.
4. **🎨 Tema ARTISTIC & EFX**: `Oil-Paint`, `Rainbow-Wave`, `Edge-Neon`, `Pixelate`, `VHS-Tape`, `Solarize`.

---

### 🔥 Kenapa HandFlux Pro Jauh Lebih Keren?

| Fitur & Keunggulan | Kenapa Ini Lebih Keren? 🚀 |
| --- | --- |
| **🔄 Auto-Cycle Mode 2 Detik (`A`)** | Filter otomatis berganti setiap 2 detik secara otomatis tanpa perlu melakukan gestur cubit! |
| **🏷️ Theme Category Switcher (`T`)** | Pilih kategori tema khusus (Cinematic, Anime, Cyber, Artistic) atau jalankan seluruh filter secara acak. |
| **🖐️ Smart Standing Finger Filter** | Hanya mendeteksi dan menggunakan jari yang **berdiri tegak**. Jari yang dilipat ke telapak tangan otomatis diabaikan! |
| **📐 CCW Polar Angle Sorting** | Poligon portal diurutkan berlawanan arah jarum jam secara matematis. **Portal 100% TIDAK AKAN KELIPET ATAU MENYILANG**! |
| **🎨 25 Filter Visual Premium** | Koleksi 25 filter spektakuler dari gaya film sinematik, anime cel shading, cyberpunk neon, hingga efek lukisan minyak. |
| **✨ EMA Landmark Smoothing** | Diterapkan *Exponential Moving Average (alpha=0.5)* untuk menyaring getaran piksel (*jitter*). |
| **🖥️ Glassmorphic HUD & Toasts** | Tampilan antarmuka transparan futuristik dilengkapi sistem notifikasi mengambang (*floating toast system*). |
| **🖱️ One-Click Desktop Launcher** | Dilengkapi file batch & shortcut instan di Desktop Windows. |
| **⚡ High Frame-Rate Throughput** | Didukung optimasi buffer kamera real-time yang mampu membuka performa hingga 120 FPS. |

---

### 🎮 Tabel Kontrol & Gesture

| Tombol / Gesture | Aksi & Fungsi |
| --- | --- |
| 🖱️ **Double Click Shortcut** | Jalankan aplikasi langsung dari Desktop Windows |
| 🔄 **Tombol `A`** | **Toggle Auto-Cycle Mode** (Filter otomatis berganti tiap 2 detik) |
| 🏷️ **Tombol `T`** | **Ganti Tema Filter** (`ALL` ➔ `CINEMATIC` ➔ `ANIME` ➔ `CYBER` ➔ `ARTISTIC`) |
| 🤏 **Pinch (Jempol + Telunjuk)** | Mengganti filter visual secara manual |
| 🔢 **Tombol `1` - `5`** | Mengatur jumlah jari aktif (1=Lingkaran, 2=Kapsul, 3=Segitiga, 4=Quad, 5=Poligon 5 Sudut) |
| 📸 **Tombol `S`** | Mengambil screenshot manual (tersimpan di `captures/`) |
| ✌️ **Tombol `G`** | Toggle auto-screenshot gesture Peace Sign (`Off` secara default) |
| 🎥 **Tombol `R`** | Memulai / menghentikan perekaman video MP4 |
| ◀️▶️ **Tombol `N` / `P`** | Filter berikutnya / sebelumnya |
| 🧊 **Tombol `C`** | Toggle mode portal 2D Quad vs 3D Mesh |
| 🪞 **Tombol `M`** | Toggle mode Mirror (cermin) |
| 👁️ **Tombol `H`** | Menyembunyikan / menampilkan HUD overlay |
| 🚪 **Tombol `Q` / `Esc`** | Keluar dari aplikasi |

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

3. **Jalankan Aplikasi:**
   ```bash
   python Retrolens.py
   ```
   *Atau dengan Auto-Cycle langsung aktif:*
   ```bash
   python Retrolens.py --auto-cycle
   ```

4. **Uji Coba Automated Test:**
   ```bash
   python test_retrolens.py
   ```

---

<br/>

## 🇬🇧 English

### 📖 About HandFlux Pro
**HandFlux Pro** is an interactive Computer Vision application powered by Python, OpenCV, and MediaPipe Tasks API. It dynamically transforms your hand gestures into interactive visual filter portals in real-time.

Featuring **25 premium visual filters** categorized into 4 themes (*Cinematic, Anime, Cyberpunk, Artistic*) and an **Auto-Cycle 2s Mode** that automatically rotates filters every 2 seconds without requiring gestures.

---

### 🎨 25 Themed Visual Filters

- **🎬 CINEMATIC**: `Teal-Orange`, `Kodachrome`, `Technicolor`, `Noir-Film`, `Cinematic-Warm`, `Vignette-Cinema`, `Sepia`.
- **⛩️ ANIME & CARTOON**: `Anime-Cel`, `Manga-Ink`, `Cartoon-Classic`, `Pop-Art`, `Pencil-Sketch`, `Posterize`.
- **⚡ CYBER & SCI-FI**: `Cyberpunk`, `Matrix`, `Thermal`, `Night-Vision`, `Hologram`, `Glitch-RGB`.
- **🎨 ARTISTIC & EFX**: `Oil-Paint`, `Rainbow-Wave`, `Edge-Neon`, `Pixelate`, `VHS-Tape`, `Solarize`.

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
**HandFlux Pro** 是一款基于 Python、OpenCV 和 MediaPipe Tasks API 开发的互动式计算机视觉应用。它包含 **25 种主题滤镜**（电影感、动漫风、赛博朋克、艺术涂鸦）以及 **2 秒自动循环模式**。

---

### 🎨 25 种主题滤镜

- **🎬 电影感 (CINEMATIC)**: 好莱坞橙青 (Teal-Orange)、柯达胶卷 (Kodachrome)、彩色印相 (Technicolor)、黑白电影 (Noir-Film)、暖阳 (Warm)、暗角 (Vignette)、复古褐色 (Sepia)。
- **⛩️ 动漫与卡通 (ANIME)**: 动漫赛璐珞 (Anime-Cel)、日漫水墨 (Manga-Ink)、经典卡通 (Cartoon)、波普艺术 (Pop-Art)、铅笔素描 (Pencil-Sketch)、海报化 (Posterize)。
- **⚡ 赛博与科影 (CYBER)**: 赛博朋克 (Cyberpunk)、黑客帝国 (Matrix)、红外热成像 (Thermal)、夜视仪 (Night-Vision)、全息投影 (Hologram)、RGB 故障 (Glitch-RGB)。
- **🎨 艺术与特效 (ARTISTIC)**: 油画 (Oil-Paint)、彩虹波浪 (Rainbow-Wave)、霓虹边缘 (Edge-Neon)、像素化 (Pixelate)、VHS 录像带 (VHS-Tape)、曝光过度 (Solarize)。

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
