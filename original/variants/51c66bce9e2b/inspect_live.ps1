$ErrorActionPreference='Stop'
$expected=(Resolve-Path 'output/论文_湿件训练有效性_20261120/main.docx').Path
foreach ($prog in @('KWPS.Application','Word.Application')){
 try {
  $app=[Runtime.InteropServices.Marshal]::GetActiveObject($prog)
  Write-Output ($prog+' documents='+$app.Documents.Count+' hwnd='+$app.Hwnd)
  for($i=1;$i -le $app.Documents.Count;$i++){
   $doc=$app.Documents.Item($i)
   if($doc.FullName -ieq $expected){Write-Output ('TARGET writable='+(-not $doc.ReadOnly)+' saved='+$doc.Saved+' track='+$doc.TrackRevisions+' paras='+$doc.Paragraphs.Count)}
  }
 } catch {Write-Output ($prog+' '+$_.Exception.GetType().Name+' '+$_.Exception.Message)}
}
