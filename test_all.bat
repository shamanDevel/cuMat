for %%f in (build/tests/Debug/*.exe) do (
	start "" /B /W "build/tests/Debug/%%~f"
)
pause