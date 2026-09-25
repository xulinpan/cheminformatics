@echo off
REM Scaffold-cluster bootstrap for the tuned-baseline contrasts (paper Sect. 6).
REM Four independent slices of 500 draws, run in parallel, then merged.
REM Each slice checkpoints every 100 draws, so rerunning this file resumes
REM whatever was interrupted rather than starting over.
cd /d D:\research2026\kaggle
start "boot1" /min .venv\Scripts\python.exe cheminformatics\scripts\tuned_baseline_intervals.py --root . --dataset open --part p1 --seed 1 --draws 500
start "boot2" /min .venv\Scripts\python.exe cheminformatics\scripts\tuned_baseline_intervals.py --root . --dataset open --part p2 --seed 2 --draws 500
start "boot3" /min .venv\Scripts\python.exe cheminformatics\scripts\tuned_baseline_intervals.py --root . --dataset open --part p3 --seed 3 --draws 500
start "boot4" /min .venv\Scripts\python.exe cheminformatics\scripts\tuned_baseline_intervals.py --root . --dataset open --part p4 --seed 4 --draws 500
echo Four slices started. When all four windows have closed, run:
echo   .venv\Scripts\python.exe cheminformatics\scripts\tuned_baseline_intervals.py --root . --dataset open --merge
