$ErrorActionPreference='Stop'
$source=(Resolve-Path (Join-Path $PSScriptRoot 'draft.docx')).Path
$pdf=Join-Path $PSScriptRoot 'draft.pdf'
$app=New-Object -ComObject KWPS.Application
try {
 $app.Visible=$false
 $app.DisplayAlerts=0
 $d=$app.Documents.Open($source,$false,$true)
 try {$d.ExportAsFixedFormat($pdf,17)} finally {$d.Close(0)}
} finally {$app.Quit()}
