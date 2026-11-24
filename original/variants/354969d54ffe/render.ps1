$ErrorActionPreference='Stop'
$source=(Resolve-Path 'output/论文_湿件训练有效性_20261120/main.docx').Path
$qa=Join-Path $PSScriptRoot 'qa'
[IO.Directory]::CreateDirectory($qa) | Out-Null
$pdf=Join-Path $qa 'paper.pdf'
$writer=New-Object -ComObject KWPS.Application
$writer.Visible=$false
$writer.DisplayAlerts=0
try {
 $d=$writer.Documents.Open($source,$false,$true)
 $d.ExportAsFixedFormat($pdf,17)
 $d.Close(0)
} finally { $writer.Quit() }
$rt=(Get-Item 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907*').FullName
& "$rt/dependencies/python/python.exe" -X utf8 (Join-Path $PSScriptRoot 'pages.py')
