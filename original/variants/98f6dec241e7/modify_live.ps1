$ErrorActionPreference='Stop'
$out=(Resolve-Path 'tmp/paper_h5_revision').Path
if($null -eq $doc -or $doc.FullName -ine $expected -or $doc.TrackRevisions -or $doc.ReadOnly){throw 'Cannot edit selected document'}
$changes=ConvertFrom-Json ([IO.File]::ReadAllText((Join-Path $out 'edits.json')))
$before=ConvertFrom-Json ([IO.File]::ReadAllText((Join-Path $out 'live_before.json')))
$future=ConvertFrom-Json ([IO.File]::ReadAllText((Join-Path $out 'future.json')))
$ix=0
foreach($pg in $doc.Paragraphs){if([string]$pg.Range.Text -cne [string]$before[$ix].t){throw "Live text changed at paragraph $ix before edits"};$ix++}
if($ix -ne $before.Count -or $changes.Count -ne 16){throw 'Document snapshot mismatch'}
foreach($c in $changes){$rr=$doc.Range([int]$c.s,[int]$c.e);if([string]$rr.Text -cne [string]$c.old){throw "Range mismatch at $($c.s)"}}
foreach($c in ($changes|Sort-Object -Property s -Descending)){
 $rr=$doc.Range([int]$c.s,[int]$c.e)
 if([string]$rr.Text -cne [string]$c.old){throw 'Range moved while performing edit'}
 if($c.mode -eq 'remove'){$null=$rr.Delete()}else{
  $rr.End=[int]$c.e-1
  $rr.Text=[string]$c.new
 }
}
$after=@()
foreach($pg in $doc.Paragraphs){$after+=,[string]$pg.Range.Text}
[IO.File]::WriteAllText((Join-Path $out 'live_after.json'),(ConvertTo-Json -InputObject $after -Depth 3 -Compress),[Text.UTF8Encoding]::new($false))
while($after.Count -gt $future.Count -and $after[-1].Trim().Length -eq 0){$after=@($after[0..($after.Count-2)])}
if($after.Count -ne $future.Count){throw "Paragraph counts after edits $($after.Count) expected $($future.Count)"}
for($ix=0;$ix -lt $future.Count;$ix++){if([string]$after[$ix] -cne [string]$future[$ix]){throw "Unexpected text at paragraph $ix"}}
$doc.Save()
Write-Output ('Verified live text, saved='+$doc.Saved+' current paragraphs='+$doc.Paragraphs.Count)
$doc.ExportAsFixedFormat((Join-Path $out 'final_live.pdf'),17)
Write-Output 'Final live manuscript exported to PDF for layout check'
