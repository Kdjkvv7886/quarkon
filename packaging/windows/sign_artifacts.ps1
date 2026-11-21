# Sign Quarkon executables with an Authenticode certificate.
#
# Everything is passed through environment variables so a password never appears
# in a process command line (and therefore never in CI logs):
#
#   QUARKON_SIGN_FILES          semicolon-separated list of files to sign
#   QUARKON_SIGN_PFX            path to a .pfx with the private key (optional if a
#                               thumbprint in the user store is used instead)
#   QUARKON_SIGN_PFX_PASSWORD   password for the .pfx
#   QUARKON_SIGN_CERT_THUMBPRINT thumbprint of a certificate already in
#                                Cert:\CurrentUser\My
#   QUARKON_TIMESTAMP_URL       optional RFC 3161 timestamp server; a self-signed
#                               certificate usually cannot be timestamped, so this
#                               is only set for CA-issued certificates
#
# Exit codes: 0 signed (or nothing to do), 2 no certificate configured,
# 3 signing produced a file without a usable signature.

$ErrorActionPreference = "Stop"

# GitHub Actions exports a PSModulePath aimed at PowerShell 7; Windows PowerShell
# 5.1 then cannot autoload its own Microsoft.PowerShell.Security module. Re-add the
# default module directory before using any cmdlet from it.
$defaultModules = Join-Path $PSHOME "Modules"
if ($env:PSModulePath -notlike "*$defaultModules*") {
    $env:PSModulePath = $defaultModules + ";" + $env:PSModulePath
}
Import-Module Microsoft.PowerShell.Security -Force -ErrorAction Stop

$files = @()
if ($env:QUARKON_SIGN_FILES) {
    $files = @($env:QUARKON_SIGN_FILES -split ";" | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
}
if ($files.Count -eq 0) {
    Write-Host "signing: nothing to sign"
    exit 0
}

$thumbprint = $env:QUARKON_SIGN_CERT_THUMBPRINT
if ($env:QUARKON_SIGN_PFX) {
    if (-not (Test-Path -LiteralPath $env:QUARKON_SIGN_PFX)) {
        Write-Host "signing: PFX not found at $($env:QUARKON_SIGN_PFX)"
        exit 2
    }
    $password = ConvertTo-SecureString $env:QUARKON_SIGN_PFX_PASSWORD -AsPlainText -Force
    $imported = Import-PfxCertificate -FilePath $env:QUARKON_SIGN_PFX -CertStoreLocation "Cert:\CurrentUser\My" -Password $password
    $thumbprint = $imported.Thumbprint
    Write-Host "signing: imported certificate $thumbprint"
}
if (-not $thumbprint) {
    Write-Host "signing: no certificate configured (QUARKON_SIGN_PFX or QUARKON_SIGN_CERT_THUMBPRINT)"
    exit 2
}

$certificate = Get-Item -LiteralPath "Cert:\CurrentUser\My\$thumbprint"
Write-Host "signing: certificate '$($certificate.Subject)' valid until $($certificate.NotAfter.ToString('yyyy-MM-dd'))"

$failed = 0
foreach ($file in $files) {
    $parameters = @{
        FilePath      = $file
        Certificate   = $certificate
        HashAlgorithm = "SHA256"
    }
    if ($env:QUARKON_TIMESTAMP_URL) {
        $parameters["TimestampServer"] = $env:QUARKON_TIMESTAMP_URL
    }
    $result = Set-AuthenticodeSignature @parameters
    $verification = Get-AuthenticodeSignature -FilePath $file
    $name = Split-Path -Path $file -Leaf
    Write-Host ("signing: {0} -> {1}; signer={2}" -f $name, $verification.Status, $verification.SignerCertificate.Subject)
    # UnknownError means "valid signature, chain ends in an untrusted root", which
    # is expected for the project's self-signed certificate.
    if ($verification.Status -ne "Valid" -and $verification.Status -ne "UnknownError") {
        Write-Host ("signing: FAILED for {0}: {1}" -f $name, $verification.StatusMessage)
        $failed++
    }
}

if ($failed -gt 0) { exit 3 }
exit 0
