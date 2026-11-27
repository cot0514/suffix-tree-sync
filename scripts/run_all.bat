@echo off
chcp 65001 >nul
rem 전 방식 통합 측정 (lazy 담당). 먼저 scripts\build.bat
rem   scripts\run_all.bat          전체 측정 (방식 4개 약 35분, 6개 약 55분)
rem   scripts\run_all.bat quick    스크립트가 끝까지 도는지 확인하는 작은 설정 (2~3분)
rem 결과 (results\, P = all 또는 quick)
rem   P_info.txt              시작·끝 시각, git 커밋, 등록된 방식
rem   P_check.txt             1) 정확성 검사 (--quick, 모든 방식, 삭제 0·2)
rem   P_scale.csv / .txt      2) 스레드 수별 속도 (phase, 데이터 2개 x 삭제 0·2·8)
rem   P_mixed.csv / .txt      3) 읽기·쓰기가 섞일 때 (mixed, 삭제 0. read_with_update 가 없는 방식은 측정 틀이 건너뜀)
rem 그래프: python scripts\plot.py   (all_*.csv 를 읽어 results\plots\ 에 저장)
rem 방식: 기본은 등록된 전부(--mode=all) 라서 새 방식을 합쳐도 이 파일은 고치지 않는다.
rem       일부만 재려면 먼저  set RUN_MODES=serial,global_lock,lazy
rem 같은 P 로 다시 돌리면 이전 결과는 .bak 으로 옮긴다 (CSV 는 이어 쓰기라 섞이지 않게).
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set H=bin\tree_bench.exe
set R=results
if not exist %H% (
  echo [ERROR] %H% 가 없음. 먼저 scripts\build.bat
  exit /b 1
)
if not exist %R% mkdir %R%

rem ---------------------------------------------------------------- 설정
set P=all
set THREADS=1,2,4,6,8,12
set MIXTH=1,4,12
set B=256
set N=10000
set STEPS=100
set REPEAT=3
set EVICTS=0 2 8
if /i "%~1"=="quick" (
  set P=quick
  set THREADS=1,4
  set MIXTH=1,4
  set B=32
  set N=300
  set STEPS=20
  set REPEAT=1
  set EVICTS=0 2
)
set MODES=all
if defined RUN_MODES set MODES=%RUN_MODES%

rem 이전 결과 보관
for %%f in (%P%_info.txt %P%_check.txt %P%_scale.csv %P%_scale.txt %P%_mixed.csv %P%_mixed.txt) do (
  if exist %R%\%%f move /y %R%\%%f %R%\%%f.bak >nul
)

set INFO=%R%\%P%_info.txt
> %INFO% echo run_all %P%   start %date% %time%
for /f %%v in ('git rev-parse --short HEAD 2^>nul') do >> %INFO% echo git commit %%v
>> %INFO% echo settings: modes=%MODES% threads=%THREADS% mixed_threads=%MIXTH% b=%B% n_cached=%N% steps=%STEPS% repeat=%REPEAT% evict=%EVICTS%
%H% --list >> %INFO%
type %INFO%

rem ---------------------------------------------------------------- 1) 정확성
echo.
echo === 1/3 정확성 검사 (--quick, 삭제 0·2)   %time%
set CHECK_FAIL=0
for %%D in (code chat) do for %%X in ("" "--evict=2") do (
  >> %R%\%P%_check.txt echo ===== %%D %%~X
  %H% data\tokens_%%D.bin --quick --mode=%MODES% %%~X >> %R%\%P%_check.txt 2>&1
  if errorlevel 1 set CHECK_FAIL=1
)
findstr /c:"CHECK SUMMARY" /c:"thr: FAIL" %R%\%P%_check.txt

rem ---------------------------------------------------------------- 2) 스레드 수별 (phase)
echo.
echo === 2/3 스레드 수별 속도 (phase)   %time%
for %%D in (code chat) do for %%E in (%EVICTS%) do (
  echo   %%D evict=%%E   !time!
  %H% data\tokens_%%D.bin --mode=%MODES% --sched=phase --threads=%THREADS% --b=%B% --n_cached=%N% --steps=%STEPS% --repeat=%REPEAT% --evict=%%E --csv=%R%\%P%_scale.csv >> %R%\%P%_scale.txt 2>&1
)

rem ---------------------------------------------------------------- 3) mixed
echo.
echo === 3/3 읽기·쓰기 섞기 (mixed)   %time%
for %%D in (code chat) do (
  echo   %%D   !time!
  %H% data\tokens_%%D.bin --mode=%MODES% --sched=mixed --threads=%MIXTH% --b=%B% --n_cached=%N% --steps=%STEPS% --repeat=%REPEAT% --evict=0 --csv=%R%\%P%_mixed.csv >> %R%\%P%_mixed.txt 2>&1
)

>> %INFO% echo end %date% %time%
echo.
echo done   %time%   결과: %R%\%P%_*.csv, %R%\%P%_*.txt   그래프: python scripts\plot.py
if "!CHECK_FAIL!"=="1" (
  echo [WARN] 정확성 검사에 FAIL 이 있음 → %R%\%P%_check.txt 확인. 그 방식의 속도 숫자는 믿을 수 없음
  exit /b 1
)
exit /b 0
