@echo off
title SPDS Pothole Detection
color 0A

echo ========================================================
echo         SPDS Pothole Detection Pipeline (Terminal)
echo ========================================================
echo.

:: Prompt for video path
set /p videoPath="Enter the absolute path to the video or image: "

:: Prompt for showing rejected detections
set /p showRejected="Show rejected detections (red boxes) [y/n]? "

echo.
echo ========================================================
echo Starting inference... Please wait.
echo ========================================================
echo.

:: Run the Python script directly in this terminal window
if /I "%showRejected%"=="y" (
    python integration\spds_pidnet_pipeline.py -i "%videoPath%" --show-rejected
) else (
    python integration\spds_pidnet_pipeline.py -i "%videoPath%"
)

echo.
echo ========================================================
echo Inference Complete!
echo Check the C:\Projects\SPDS\integration\outputs folder for results.
echo ========================================================
pause
