# 🎙️ Stutter Detection System

A **multimodal end-to-end machine learning system** that detects stuttering in speech from video input.  
It fine-tunes **Wav2Vec2** on the SEP-28k dataset and fuses audio embeddings with **MediaPipe facial landmark** features.

---

## Architecture

```
Video Upload
    │
    ├─► FFmpeg ──────────────► Mono 16-kHz WAV
    │           Audio Branch:
    │           Wav2Vec2 encoder → mean-pool → Linear → 256-D embedding
    │
    └─► OpenCV frames
         MediaPipe FaceMesh
         10 facial features / frame → BiGRU + attention → 128-D embedding
    
    Audio (256-D) ⊕ Video (128-D) → Fusion MLP → 5-class classifier
    
    Output: label | confidence | stutter % | timeline
```

---

## Detected Stutter Types

| Label          | Description                                      |
|----------------|--------------------------------------------------|
| **Fluent**     | Normal, uninterrupted speech                     |
| **Repetition** | Sound / syllable / word repetitions (e-e-eight)  |
| **Prolongation**| Extended sound durations (ssssix)               |
| **Block**      | Silent pauses mid-word (no airflow)              |
| **Interjection**| Filler words (um, uh, like)                     |

---------
