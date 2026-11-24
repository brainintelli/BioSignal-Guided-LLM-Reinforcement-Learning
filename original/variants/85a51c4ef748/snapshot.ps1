$ErrorActionPreference='Stop'
$out=(Resolve-Path 'tmp/paper_h5_revision').Path
if ($doc.FullName -ine $expected -or $doc.ReadOnly){throw 'Wrong document or read only'}
$rows=@()
foreach ($pa in $doc.Paragraphs){
 $rr=$pa.Range
 $rows += [ordered]@{s=[int]$rr.Start;e=[int]$rr.End;t=[string]$rr.Text}
}
[IO.File]::WriteAllText((Join-Path $out 'live_before.json'),(ConvertTo-Json -InputObject $rows -Depth 3 -Compress),[Text.UTF8Encoding]::new($false))
try {
 $flat=[string]$doc.WordOpenXML
 if($flat.Length -gt 2048){[IO.File]::WriteAllText((Join-Path $out 'live_before_flat.xml'),$flat,[Text.UTF8Encoding]::new($false))}
 Write-Output ('Flat OPC bytes: '+$flat.Length)
} catch {Write-Output 'No flat OPC backup surface'}
