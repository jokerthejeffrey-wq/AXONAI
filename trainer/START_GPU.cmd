@echo off
cd /d "%~dp0.."
py -3.12 trainer\start.py --gpu
pause
