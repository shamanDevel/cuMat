for %%f in (build/tests/RelWithDebInfo/*.exe) do (
	start "" /B /W "build/tests/RelWithDebInfo/%%~f"
)
pause