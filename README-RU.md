# Apex Legends Enemy Detector

> [English version](README.md)

Демо-приложение для просмотра результатов работы YOLOv8-моделей обнаружения врагов на видео геймплея Apex Legends. Загрузи видео, выбери модель и получи обработанный результат с bounding box-ами и опциональным исправлением растянутого разрешения.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-required-76B900?logo=nvidia&logoColor=white)

> Финальный проект курса [Deep Learning School](https://stepik.org/250969) на Stepik.

![Detection Demo](docs/demo.gif)

---

> **Дисклеймер:** Проект является исключительно учебным инструментом, созданным как финальное задание курса по Deep Learning. Он обрабатывает **заранее записанные видеофайлы** и не взаимодействует с процессом игры, памятью или сетью. Не может использоваться как чит или аимбот в реальном времени. Не нарушает системы античита. Цель — продемонстрировать техники обнаружения объектов с помощью YOLOv8, а не получить игровое преимущество.

---

## Интерфейс настроек

![interface](docs/interface.png)

---

## Возможности

- **Умная детекция:** Центрированная область поиска для лучшей производительности.
- **Поддержка растянутого разрешения:** Исправляет stretched-видео до пропорций 1:1 для точного инференса.
- **Синхронизация звука:** Обрабатывает переменный FPS и пропущенные кадры для сохранения синхронизации аудио.
- **Очередь задач:** Асинхронная обработка через FastAPI и Worker.

## Модели

Две кастомные модели YOLOv8, размещённые на HuggingFace: [PSImera/apex_enemy_detect](https://huggingface.co/PSImera/apex_enemy_detect).

| | **YOLOv8n (nano)** | **YOLOv8m (medium)** |
|---|---|---|
| Файл | `apex_detect_v8n_v2.1.pt` | `apex_detect_v8m_v2.1.pt` |
| Параметры | ~3.2M | ~25.9M |
| mAP@50 | 0.930 | 0.938 |
| mAP@50-95 | 0.756 | 0.796 |
| Лучше для | Скорость / слабый GPU | Точность |

Подробности об обучении, метрики и кривые — на [странице модели](https://huggingface.co/PSImera/apex_enemy_detect).

---

## Требования

- **GPU с поддержкой CUDA** — обязательно для инференса (рекомендуется NVIDIA)
- **CUDA 12.8** (или измени URL установки torch под свою версию)
- **Python 3.10+**
- **ffmpeg** установлен и доступен в `PATH`

---

## Установка

### 1. Клонировать репозиторий с подмодулями:
```bash
git clone --recurse-submodules https://github.com/PSImera/apex_enemy_detect_demo.git
cd apex_enemy_detect_demo
```

### 2. Создать и активировать виртуальное окружение

```bash
python -m venv .venv
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> Измени URL torch, если используешь другую версию CUDA.

**Linux (Debian/Ubuntu) — системные библиотеки:**
```bash
sudo apt update
sudo apt install ffmpeg freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev
```

**Windows — ffmpeg:**

Скачай с [ffmpeg.org/download.html](https://ffmpeg.org/download.html) и добавь `ffmpeg/bin` в `PATH`.

---

## Запуск

Запустить бэкенд:
```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Запустить фронтенд:
```bash
streamlit run frontend/app.py
```

Страница `http://localhost:8501` откроется в браузере автоматически.
