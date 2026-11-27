@echo off
chcp 65001 >nul
rem 방식 하나를 합칠 때마다 돌리는 점검. 사용: scripts\check_merge.bat [tsan]
rem 1) Windows: tree_bench 빌드 후 등록된 모든 방식을 검사 (code, chat x 기본, 삭제 포함, 12스레드)
rem 2) WSL: scripts/check_merge_wsl.sh (vLLM 묶음 모듈. tsan 을 주면 ThreadSanitizer 도 실행)
rem WSL 부분은 ~/venv_arctic, ~/arctic_stage 를 만들어 둔 PC 에서만 된다.
rem 종료 코드: 모두 통과 0, 하나라도 실패 1
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set FAILED=0
if not exist results mkdir results
set LOG=results\check_merge_windows.log

echo === [Windows] build
call "%~dp0build.bat"
if errorlevel 1 (
  echo [FAIL] Windows build
  exit /b 1
)
rem build.bat 가 현재 폴더를 바꿀 수 있어서 다시 저장소 루트로
cd /d "%~dp0.."

type nul > "%LOG%"
for %%d in (code chat) do (
  call :run_check %%d "default"
  call :run_check %%d "--evict=2"
  call :run_check %%d "--threads=12"
)

echo.
echo === [WSL] vLLM batch module
where wsl >nul 2>nul
if errorlevel 1 (
  echo [FAIL] wsl not found
  set FAILED=1
) else (
  wsl -- bash scripts/check_merge_wsl.sh %1
  if errorlevel 1 set FAILED=1
)

echo.
if "!FAILED!"=="0" (
  echo === CHECK MERGE: ALL PASSED
) else (
  echo === CHECK MERGE: FAILED
)
echo Windows log: %LOG%
exit /b !FAILED!

rem 인자: 데이터 이름, 추가 옵션 (없으면 "default")
:run_check
set EXTRA=%~2
if "%EXTRA%"=="default" set EXTRA=
set RUN_LOG=results\check_merge_current_run.log
bin\tree_bench data\tokens_%1.bin --quick --mode=all %EXTRA% > "%RUN_LOG%" 2>&1
set CODE=!errorlevel!
echo ===== %1 %~2 exit=!CODE! >> "%LOG%"
type "%RUN_LOG%" >> "%LOG%"
set SUMMARY=
for /f "delims=" %%s in ('findstr /c:"CHECK SUMMARY" "%RUN_LOG%"') do set SUMMARY=%%s
set VERDICT=ok
if not "!CODE!"=="0" set VERDICT=FAIL
findstr /r /c:"CHECK SUMMARY:.*FAIL 0, n/a 0" "%RUN_LOG%" >nul
if errorlevel 1 set VERDICT=FAIL
if "!VERDICT!"=="FAIL" set FAILED=1
echo [!VERDICT!] %1 %~2 exit=!CODE! !SUMMARY!
if "!VERDICT!"=="FAIL" findstr /c:"FAIL" "%RUN_LOG%"
del "%RUN_LOG%"
exit /b 0
