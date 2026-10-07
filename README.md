# Tetris ML

A learning project exploring a Tetris-playing machine learning agent. The planned
work includes a neural controller optimized with a genetic algorithm and a later
vision stage. The project is currently validating the game environment; no learned
controller has been implemented yet.

## Current contents

- `Controller.py`: runs one seeded random-action episode in Tetris Gymnasium,
  reports episode metrics, and saves synchronized RGB images and numerical board
  arrays at steps 0 and 5.

The script uses environment seed 42, action seed 123, and a 500-step cap.

## Local use

The existing development environment is a Python virtual environment in `.venv`
with Tetris Gymnasium, Gymnasium, NumPy, and OpenCV installed. From the project
folder in PowerShell, run:

```powershell
New-Item -ItemType Directory -Force captures
.\.venv\Scripts\python.exe .\Controller.py
```

The script currently saves captures to the absolute path `C:\MLproject\captures`.
Running it from another location requires updating those output paths. Each run
overwrites the same step-0 and step-5 files.

Virtual environments, editor settings, local credentials, and generated captures
are excluded from version control.

## Next milestone

Complete the additional development-seed checks and repeatability validation
before building a learned controller.
