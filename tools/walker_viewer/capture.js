// Render a walker_lab recording to PNG frames + MP4.
// usage: node capture.js <recording.json> <out_dir> [fps=30] [width=960] [height=540]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

(async () => {
  const [recPath, outDir, fpsArg, wArg, hArg] = process.argv.slice(2);
  const fps = Number(fpsArg || 30), width = Number(wArg || 960), height = Number(hArg || 540);
  const rec = JSON.parse(fs.readFileSync(recPath, 'utf8'));
  const framesDir = path.join(outDir, 'frames');
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto('file://' + path.join(__dirname, 'viewer.html'));
  await page.addScriptTag({ path: path.join(__dirname, 'node_modules/three/build/three.min.js') });
  await page.evaluate(([r, w, h]) => { window.REC = r; init(w, h); }, [rec, width, height]);

  const stride = Math.max(1, Math.round(1 / (rec.dt * fps)));
  let n = 0;
  for (let i = 0; i < rec.frames.length; i += stride) {
    await page.evaluate(i => show(i), i);
    await page.screenshot({ path: path.join(framesDir, `f${String(n).padStart(5, '0')}.png`) });
    n++;
  }
  await browser.close();
  const mp4 = path.join(outDir, `${rec.scenario}${rec.assist > 0 ? '' : '_no_assist'}.mp4`);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(framesDir, 'f%05d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', mp4]);
  console.log(`${n} frames -> ${mp4}`);
})();
