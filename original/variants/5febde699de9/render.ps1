$ErrorActionPreference='Stop'
$source=(Resolve-Path (Join-Path $PSScriptRoot 'main.docx')).Path
$pdf=Join-Path $PSScriptRoot 'main.pdf'
$app=New-Object -ComObject KWPS.Application
try {
 $app.Visible=$false
 $app.DisplayAlerts=0
 $doc=$app.Documents.Open($source,$false,$true)
 try {$doc.ExportAsFixedFormat($pdf,17)} finally {$doc.Close(0)}
} finally {$app.Quit()}
