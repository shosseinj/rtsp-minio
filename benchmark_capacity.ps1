param(
    [int]$DurationSeconds = 60,
    [int]$WarmupSeconds = 10,
    [string]$OutputPath = "benchmark-results.json"
)

$ErrorActionPreference = "Stop"
$stages = @(8, 16, 24, 32, 38, 46, 54, 62, 70, 78, 86, 94)
$modes = @("decode_only", "full")
$results = @()

function Get-SourceFps {
    param([int]$Count)
    $files = @(
        Get-ChildItem -LiteralPath "data" -File -Recurse |
            Where-Object Extension -In @(".mp4", ".mov", ".mkv", ".avi", ".webm") |
            Sort-Object { $_.FullName.Substring((Resolve-Path "data").Path.Length + 1).ToLowerInvariant() }
    )
    $fpsValues = @()
    foreach ($file in $files) {
        $rate = (& ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 -- $file.FullName).Trim()
        $parts = $rate.Split("/")
        $fpsValues += if ($parts.Count -eq 2 -and [double]$parts[1] -ne 0) {
            [double]$parts[0] / [double]$parts[1]
        } else {
            [double]$rate
        }
    }
    $selected = for ($index = 0; $index -lt $Count; $index++) {
        $fpsValues[$index % $fpsValues.Count]
    }
    return [double](($selected | Measure-Object -Sum).Sum)
}

foreach ($mode in $modes) {
    foreach ($stage in $stages) {
        $env:STATIC_CAMERA_EMULATION = "true"
        $env:STATIC_CAMERA_EMULATION_FPS = "source"
        $env:STATIC_CAMERA_SOURCE_COUNT = "$stage"
        $env:BENCHMARK_STATIC_ONLY = "true"
        $env:BENCHMARK_PIPELINE_MODE = $mode

        docker compose up -d --force-recreate --no-build deepstream | Out-Null
        Start-Sleep -Seconds $WarmupSeconds

        $container = (docker compose ps -q deepstream).Trim()
        $expectedFps = Get-SourceFps -Count $stage
        $samples = @()
        $startRuntime = Invoke-RestMethod "http://127.0.0.1:7071/api/runtime"
        $stageStarted = Get-Date

        for ($elapsed = 0; $elapsed -lt $DurationSeconds; $elapsed += 2) {
            Start-Sleep -Seconds 2
            $runtime = Invoke-RestMethod "http://127.0.0.1:7071/api/runtime"
            $gpu = (& nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,memory.used --format=csv,noheader,nounits).Split(",").Trim()
            $statsRaw = docker stats --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.PIDs}}" $container
            $statsParts = $statsRaw.Split("|")
            $memoryMiB = [double](($statsParts[1].Split("/")[0].Trim()) -replace "MiB","" -replace "GiB","000")
            $samples += [pscustomobject]@{
                active = $runtime.sources.Count
                stable = @($runtime.sources | Where-Object online).Count
                decoded_fps = [double](($runtime.sources.fps | Measure-Object -Sum).Sum)
                wall_fps = [double]$runtime.wall_fps
                gpu = [double]$gpu[0]
                nvdec = [double]$gpu[1]
                nvenc = [double]$gpu[2]
                vram_mib = [double]$gpu[3]
                cpu = [double]($statsParts[0] -replace "%","")
                ram_mib = $memoryMiB
                pids = [int]$statsParts[2]
            }
        }

        $endRuntime = Invoke-RestMethod "http://127.0.0.1:7071/api/runtime"
        $previousErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $logs = docker logs --since $stageStarted.ToUniversalTime().ToString("o") $container 2>&1 | Out-String
        $ErrorActionPreference = $previousErrorPreference
        $decodedAverage = [double](($samples.decoded_fps | Measure-Object -Average).Average)
        $dropPercent = [math]::Max(0, 100 * (1 - $decodedAverage / $expectedFps))
        $restarts = ([regex]::Matches($logs, "Resetting source")).Count
        $pipelineErrors = ([regex]::Matches($logs, "GStreamer error|Fatal error|ENCODER INITIALIZATION FAILED|SIGSEGV|out of memory", "IgnoreCase")).Count
        $stableSources = [int](($samples.stable | Measure-Object -Minimum).Minimum)
        $nvdecAverage = [double](($samples.nvdec | Measure-Object -Average).Average)
        $nvdecPeak = [double](($samples.nvdec | Measure-Object -Maximum).Maximum)
        $passed = (
            $stableSources -eq $stage -and
            $nvdecAverage -le 90 -and
            $nvdecPeak -lt 99 -and
            $decodedAverage -ge (0.95 * $expectedFps) -and
            $dropPercent -le 5 -and
            $restarts -eq 0 -and
            $pipelineErrors -eq 0
        )
        $results += [pscustomobject]@{
            mode = $mode
            requested_sources = $stage
            active_sources = [int](($samples.active | Measure-Object -Minimum).Minimum)
            stable_sources = $stableSources
            aggregate_input_fps = [math]::Round($expectedFps, 1)
            aggregate_decoded_fps = [math]::Round($decodedAverage, 1)
            wall_fps = [math]::Round([double](($samples.wall_fps | Measure-Object -Average).Average), 1)
            nvdec_average = [math]::Round($nvdecAverage, 1)
            nvdec_peak = $nvdecPeak
            nvenc_average = [math]::Round([double](($samples.nvenc | Measure-Object -Average).Average), 1)
            nvenc_peak = [double](($samples.nvenc | Measure-Object -Maximum).Maximum)
            gpu_average = [math]::Round([double](($samples.gpu | Measure-Object -Average).Average), 1)
            gpu_peak = [double](($samples.gpu | Measure-Object -Maximum).Maximum)
            vram_average_mib = [math]::Round([double](($samples.vram_mib | Measure-Object -Average).Average), 0)
            vram_peak_mib = [double](($samples.vram_mib | Measure-Object -Maximum).Maximum)
            cpu_average = [math]::Round([double](($samples.cpu | Measure-Object -Average).Average), 1)
            ram_average_mib = [math]::Round([double](($samples.ram_mib | Measure-Object -Average).Average), 0)
            process_thread_count = [int](($samples.pids | Measure-Object -Maximum).Maximum)
            frame_drop_percent = [math]::Round($dropPercent, 1)
            queue_depth = 1
            source_restarts = $restarts
            generation_changes = 0
            pipeline_errors = $pipelineErrors
            oom_sigsegv = [bool]($logs -match "SIGSEGV|out of memory")
            passed = $passed
        }
        $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8
        if (-not $passed) {
            break
        }
    }
}

$results | ConvertTo-Json -Depth 5
