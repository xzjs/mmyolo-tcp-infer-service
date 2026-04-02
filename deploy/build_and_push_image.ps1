param(
    [string]$ImageName = "registry.cn-hangzhou.aliyuncs.com/xzjs/mmyolo-tcp-infer",
    [string]$Dockerfile = "docker/Dockerfile_tcp_infer",
    [string]$ContextDir = ".",
    [string]$TagStateFile = "deploy/.image_tag_state",
    [string]$StartTag = "1.6",
    [bool]$Push = $true,
    [bool]$AlsoLatest = $false,
    [bool]$UpdateCompose = $true,
    [string[]]$BuildArgs = @(),
    [string[]]$ComposeFiles = @(
        "deploy/docker-compose.tcp-infer.runtime.yaml",
        "partner_delivery/docker-compose.tcp-infer.runtime.yaml"
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-SemVerLikeTag {
    param([string]$Tag)
    if ($Tag -notmatch '^\d+(\.\d+){0,2}$') {
        throw "Tag '$Tag' is not supported. Use numeric tags like 1, 1.2 or 1.2.3."
    }
}

function Get-IncrementedTag {
    param([string]$Tag)
    Assert-SemVerLikeTag -Tag $Tag
    $parts = $Tag.Split('.') | ForEach-Object { [int]$_ }
    if ($parts.Count -eq 1) {
        return ([string]($parts[0] + 1))
    }
    if ($parts.Count -eq 2) {
        return "$($parts[0]).$($parts[1] + 1)"
    }
    return "$($parts[0]).$($parts[1]).$($parts[2] + 1)"
}

function Get-ComposeTag {
    param(
        [string]$FilePath,
        [string]$TargetImageName
    )
    if (-not (Test-Path -LiteralPath $FilePath)) {
        return $null
    }
    $lines = Get-Content -LiteralPath $FilePath
    $pattern = "^\s*image:\s*$([regex]::Escape($TargetImageName)):(\S+)\s*$"
    foreach ($line in $lines) {
        if ($line -match $pattern) {
            return $matches[1]
        }
    }
    return $null
}

function Set-ComposeTag {
    param(
        [string]$FilePath,
        [string]$TargetImageName,
        [string]$NewTag
    )
    if (-not (Test-Path -LiteralPath $FilePath)) {
        Write-Host "[WARN] compose file not found, skip: $FilePath" -ForegroundColor Yellow
        return
    }

    $content = Get-Content -LiteralPath $FilePath -Raw
    $pattern = "(?m)^(\s*image:\s*)$([regex]::Escape($TargetImageName)):[^\s#]+(\s*(?:#.*)?)$"
    $replacement = "`$1${TargetImageName}:$NewTag`$2"

    $newContent = [regex]::Replace($content, $pattern, $replacement)
    if ($newContent -ne $content) {
        Set-Content -LiteralPath $FilePath -Value $newContent -Encoding UTF8
        Write-Host "[OK] updated compose image tag -> $FilePath"
    } else {
        Write-Host "[WARN] image line not matched, skip: $FilePath" -ForegroundColor Yellow
    }
}

function Invoke-Docker {
    param([string[]]$Args)
    Write-Host "docker $($Args -join ' ')"
    & docker @Args
    if ($LASTEXITCODE -ne 0) {
        throw "docker command failed with exit code $LASTEXITCODE"
    }
}

$lastTag = $null
if (Test-Path -LiteralPath $TagStateFile) {
    $lastTag = (Get-Content -LiteralPath $TagStateFile -Raw).Trim()
    if ($lastTag) {
        Assert-SemVerLikeTag -Tag $lastTag
    }
}

if (-not $lastTag) {
    foreach ($composeFile in $ComposeFiles) {
        $composeTag = Get-ComposeTag -FilePath $composeFile -TargetImageName $ImageName
        if ($composeTag) {
            $lastTag = $composeTag
            break
        }
    }
}

if ($lastTag) {
    $nextTag = Get-IncrementedTag -Tag $lastTag
} else {
    Assert-SemVerLikeTag -Tag $StartTag
    $nextTag = $StartTag
}

$fullImageTag = "${ImageName}:$nextTag"

Write-Host "=================================================="
Write-Host "Image name : $ImageName"
Write-Host "Last tag   : $lastTag"
Write-Host "Next tag   : $nextTag"
Write-Host "Dockerfile : $Dockerfile"
Write-Host "Context    : $ContextDir"
Write-Host "Push       : $Push"
Write-Host "AlsoLatest : $AlsoLatest"
Write-Host "BuildArgs  : $($BuildArgs -join ',')"
Write-Host "=================================================="

$dockerBuildArgs = @("build", "-f", $Dockerfile, "-t", $fullImageTag)
foreach ($arg in $BuildArgs) {
    if (-not [string]::IsNullOrWhiteSpace($arg)) {
        $dockerBuildArgs += @("--build-arg", $arg)
    }
}
$dockerBuildArgs += $ContextDir

Invoke-Docker -Args $dockerBuildArgs

if ($Push) {
    Invoke-Docker -Args @("push", $fullImageTag)
    if ($AlsoLatest) {
        $latestTag = "${ImageName}:latest"
        Invoke-Docker -Args @("tag", $fullImageTag, $latestTag)
        Invoke-Docker -Args @("push", $latestTag)
    }
}

$stateDir = Split-Path -Parent $TagStateFile
if ($stateDir -and -not (Test-Path -LiteralPath $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir | Out-Null
}
Set-Content -LiteralPath $TagStateFile -Value $nextTag -Encoding UTF8
Write-Host "[OK] tag state updated -> $TagStateFile ($nextTag)"

if ($UpdateCompose) {
    foreach ($composeFile in $ComposeFiles) {
        Set-ComposeTag -FilePath $composeFile -TargetImageName $ImageName -NewTag $nextTag
    }
}

Write-Host "[DONE] build flow completed. New image: $fullImageTag" -ForegroundColor Green
