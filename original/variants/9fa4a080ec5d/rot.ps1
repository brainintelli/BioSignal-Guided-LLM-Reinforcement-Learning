$ErrorActionPreference='Stop'
$expected=(Resolve-Path 'output/论文_湿件训练有效性_20261120/main.docx').Path
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
public static class ActivePaper {
 [DllImport("ole32.dll")] static extern int GetRunningObjectTable(int reserved, out IRunningObjectTable rot);
 [DllImport("ole32.dll")] static extern int CreateBindCtx(int reserved, out IBindCtx ctx);
 public static object Find(string target){
  IRunningObjectTable rot;GetRunningObjectTable(0,out rot);IBindCtx ctx;CreateBindCtx(0,out ctx);
  IEnumMoniker en;rot.EnumRunning(out en);IMoniker[] m=new IMoniker[1];
  while(en.Next(1,m,IntPtr.Zero)==0){string name="";
   try{m[0].GetDisplayName(ctx,null,out name);}catch{continue;}
   if(String.Equals(name,target,StringComparison.OrdinalIgnoreCase)){
    object o;rot.GetObject(m[0],out o);return o;
   }
  }
  return null;
 }
}
'@
$doc=[ActivePaper]::Find($expected)
if ($null -eq $doc){Write-Output 'No exact running file moniker';return}
Write-Output ('Exact file: writable='+(-not $doc.ReadOnly)+' saved='+$doc.Saved+' tracking='+$doc.TrackRevisions+' paragraphs='+$doc.Paragraphs.Count)
