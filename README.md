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

Create a virtual environment and install the required packages from `requirements.txt`.
From the project folder in PowerShell, run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\Controller.py
```

Captures are saved to the automatically created `captures/` folder next to `Controller.py`, regardless of working directory. Each run overwrites the same step-0 and step-5 files.

Virtual environments, editor settings, local credentials, and generated captures
are excluded from version control.

## Next milestone

Complete the additional development-seed checks and repeatability validation
before building a learned controller.
