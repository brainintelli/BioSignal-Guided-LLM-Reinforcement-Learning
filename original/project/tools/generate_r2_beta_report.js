// Windows entry point for the checked, fixed-tick analysis. Model training runs on the GPU servers.
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.join(__dirname, '..');
const cmd = process.argv.indexOf('--python');
let python = cmd < 0 ? process.env.MEA_REPORT_PYTHON : process.argv[cmd + 1];
if (!python && process.platform === 'win32') {
  const resolved = spawnSync('powershell.exe', ['-NoProfile', '-Command',
    "(Get-Item -Path 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.*/dependencies/python/python.exe' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName"], {encoding:'utf8'});
  if (resolved.status === 0 && fs.existsSync(resolved.stdout.trim())) python = resolved.stdout.trim();
}
python ||= 'python';
for (const name of ['audit_r2_beta_clock.py','report_r2_beta_sweep.py']) {
  const r = spawnSync(python,['-X','utf8','-u',path.join(__dirname,name)],{cwd:root,stdio:'inherit'});
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status ?? 1);
}
