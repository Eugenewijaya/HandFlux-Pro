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
**HandFlux Pro** adalah aplikasi *Computer Vision* interaktif yang mengubah gerakan tangan Anda menjadi portal efek visual real-time. Cukup bentangkan atau gerakkan jari tangan Anda di depan webcam untuk memproyeksikan portal efek futuristik (*Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon*, dan 10+ efek visual spektakuler lainnya).

Aplikasi ini dilengkapi dengan pengenalan bentuk portal otomatis (1 hingga 5 jari), deteksi jari tegak yang presisi, serta penyaringan gerakan cubit (*pinch*) untuk mengganti filter visual secara alami.

---

### 🔥 Kenapa HandFlux Pro Jauh Lebih Keren?

| Fitur & Keunggulan | Kenapa Ini Lebih Keren? 🚀 |
| --- | --- |
| **🖐️ Smart Standing Finger Filter** | Hanya mendeteksi dan menggunakan jari yang **berdiri tegak**. Jari yang dilipat ke telapak tangan otomatis diabaikan, membuat pembuatan portal jauh lebih presisi. |
| **📐 CCW Polar Angle Sorting** | Poligon portal diurutkan berlawanan arah jarum jam secara matematis. **Portal 100% TIDAK AKAN KELIPET ATAU MENYILANG** meski tangan diputar! |
| **🎨 16 Filter Visual Premium** | Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon, Thermal, Sketch, Glitch, Dual-Tone, Pixelate, Sepia, dan banyak lagi. |
| **✨ EMA Landmark Smoothing** | Diterapkan *Exponential Moving Average (alpha=0.5)* untuk menyaring getaran piksel (*jitter*). Rangka jari dan garis portal terasa tenang dan solid. |
| **🎯 Distance-Invariant Pinch Ratio** | Gestur cubit (*pinch*) dihitung relatif terhadap ukuran tangan. Respon konsisten baik saat tangan 30 cm maupun 2 meter dari webcam. |
| **🖥️ Glassmorphic HUD & Toasts** | Tampilan antarmuka transparan futuristik dilengkapi sistem notifikasi mengambang (*floating toast system*). |
| **🖱️ One-Click Desktop Launcher** | Dilengkapi file batch & shortcut instan di Desktop. Tinggal *double click* tanpa perlu buka terminal! |
| **📹 Perekam MP4 & Screenshot** | Dukungan perekaman video MP4 real-time (`R`) dan jepret foto manual (`S`). |
| **⚡ High Frame-Rate Throughput** | Didukung optimasi buffer kamera real-time yang mampu membuka performa hingga 120 FPS tanpa batas buatan. |

---

### 🎮 Tabel Kontrol & Gesture

| Tombol / Gesture | Aksi & Fungsi |
| --- | --- |
| 🖱️ **Double Click Shortcut** | Jalankan aplikasi langsung dari Desktop Windows |
| 🤏 **Pinch (Jempol + Telunjuk)** | Mengganti filter visual secara real-time |
| 🔢 **Tombol `1` - `5`** | Mengatur jumlah jari aktif (1=Lingkaran, 2=Kapsul, 3=Segitiga, 4=Quad, 5=Poligon 5 Sudut) |
| 🔄 **Tombol `F`** | Berpindah jumlah jari aktif secara berurutan (`1` ➔ `5`) |
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
   *Atau jalankan via file shortcut di Desktop Anda!*

4. **Uji Coba Automated Test:**
   ```bash
   python test_retrolens.py
   ```

---

<br/>

## 🇬🇧 English

### 📖 About HandFlux Pro
**HandFlux Pro** is an interactive Computer Vision application powered by Python, OpenCV, and MediaPipe Tasks API. It dynamically transforms your hand gestures into interactive visual filter portals in real-time.

Simply move your hands in front of the webcam to project futuristic visual effects (*Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon*, and 10+ more).

---

### 🔥 Why HandFlux Pro Rocks?

- **🖐️ Smart Standing Finger Detection**: Detects and uses ONLY extended (standing) fingers. Folded fingers are automatically excluded.
- **📐 Non-Crossing CCW Polygon Geometry**: Vertices are sorted counter-clockwise around the centroid, mathematically preventing self-intersecting or folded shapes.
- **🎨 16 Premium Visual Filters**: Cyberpunk, VHS, Matrix, Pop-Art, Rainbow Wave, Cartoon, Thermal, Sketch, Glitch, Dual-Tone, Pixelate, Sepia, and more.
- **✨ EMA Landmark Smoothing**: Uses Exponential Moving Average filtering to eliminate pixel jitter.
- **🎯 Distance-Invariant Pinch Recognition**: Dynamic scale-invariant pinch detection that works consistently at any distance.
- **⚡ Uncapped Frame-Rate Performance**: High-throughput rendering pipeline supporting up to 120 FPS.

---

### 🎮 Controls & Shortcuts

| Key / Gesture | Function |
| --- | --- |
| 🤏 **Pinch (Thumb + Index)** | Cycle visual filters in real-time |
| 🔢 **Keys `1` to `5`** | Set active finger count (1=Circle, 2=Pill, 3=Triangle, 4=Quad, 5=Polygon) |
| 🔄 **Key `F`** | Cycle active finger count |
| 📸 **Key `S`** | Take manual screenshot (saved to `captures/`) |
| ✌️ **Key `G`** | Toggle Peace Sign gesture screenshot (`Off` by default) |
| 🎥 **Key `R`** | Start / stop MP4 video recording |
| ◀️▶️ **Keys `N` / `P`** | Next / previous filter |
| 🧊 **Key `C`** | Toggle 2D Quad vs 3D Mesh portal mode |
| 🪞 **Key `M`** | Toggle mirror mode |
| 👁️ **Key `H`** | Toggle HUD overlay |
| 🚪 **Key `Q` / `Esc`** | Quit application |

---

<br/>

## 🇨🇳 中文 (Chinese)

### 📖 关于 HandFlux Pro
**HandFlux Pro** 是一款基于 Python、OpenCV 和 MediaPipe Tasks API 开发的互动式计算机视觉应用。它能将您的手势实时转换为动态视觉滤镜传送门。

只需要在摄像头前展示手势，即可投射赛博朋克 (Cyberpunk)、VHS 录像带、黑客帝国 (Matrix)、波普艺术 (Pop-Art)、彩虹波浪 (Rainbow Wave) 等 16 种炫酷特效。

---

### 🔥 为什么 HandFlux Pro 更强大？

- **🖐️ 智能立手指过滤**: 仅识别并使用**直立伸展的手指**，弯曲收起的手指会自动被忽略。
- **📐 CCW 极角排序防折叠几何**: 采用逆时针极角排序算法，从数学上保证传送门多边形**绝不交叉或折叠**。
- **🎨 16 种炫酷视觉特效**: 包含赛博朋克、VHS、黑客帝国、波普艺术、彩虹波浪等 16 种丰富特效。
- **✨ EMA 手势降噪平滑**: 引入指数移动平均 (EMA) 算法，彻底消除骨骼及线条抖动。
- **⚡ 高帧率性能支持**: 支持无锁帧高吞吐量渲染，最高可达 120 FPS。

---

### 🎮 快捷键与手势指南

| 按键 / 手势 | 功能描述 |
| --- | --- |
| 🤏 **捏合 (拇指 + 食指)** | 实时切换视觉滤镜 |
| 🔢 **按键 `1` - `5`** | 设置有效手指数量 (1=圆形, 2=胶囊形, 3=三角形, 4=四边形, 5=多边形) |
| 🔄 **按键 `F`** | 循环切换有效手指数量 |
| 📸 **按键 `S`** | 手动截图 (保存至 `captures/`) |
| 🎥 **按键 `R`** | 开始 / 停止 MP4 视频录制 |
| ◀️▶️ **按键 `N` / `P`** | 切换上一个 / 下一个滤镜 |
| 🧊 **按键 `C`** | 切换 2D 四边形 / 3D 网格传送门模式 |
| 🚪 **按键 `Q` / `Esc`** | 退出程序 |

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
