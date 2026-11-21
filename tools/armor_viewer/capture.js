// Render an armor_lab recording to an MP4 (one frame per physics step, played at `fps`).
// usage: node capture.js <recording.json> <out.mp4> [fps=30] [width=960] [height=540]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync } = require('child_process');
(async () => {
  const [recPath, out, fpsArg, wArg, hArg] = process.argv.slice(2);
  const fps = Number(fpsArg || 30), width = Number(wArg || 960), height = Number(hArg || 540);
  const rec = JSON.parse(fs.readFileSync(recPath, 'utf8'));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'armor-'));
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto('file://' + path.join(__dirname, 'viewer.html'));
  await page.addScriptTag({ path: path.join(__dirname, 'node_modules/three/build/three.min.js') });
  await page.evaluate(([r, w, h]) => { window.REC = r; init(w, h); }, [rec, width, height]);
  for (let i = 0; i < rec.frames.length; i++) {
    await page.evaluate(i => show(i), i);
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(5, '0')}.png`) });
  }
  await browser.close();
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(dir, 'f%05d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', out]);
  fs.copyFileSync(path.join(dir, `f${String(rec.frames.length - 1).padStart(5, '0')}.png`), out.replace(/\.mp4$/, '_final.png'));
  fs.rmSync(dir, { recursive: true, force: true });
  console.log(`${rec.frames.length} frames -> ${out}`);
})();
