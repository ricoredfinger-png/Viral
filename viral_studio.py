#!/usr/bin/env python3
"""
╔══════════════════════════════════════════╗
║   VIRAL STUDIO PRO — Web Edition         ║
║   Jalankan: python viral_studio.py       ║
║   Buka:     http://localhost:7575        ║
╚══════════════════════════════════════════╝
"""

import os, uuid, logging, threading
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

# ─── KONFIGURASI ENGINE ───────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(CURRENT_DIR, "ffmpeg.exe")
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
logging.getLogger("moviepy").setLevel(logging.ERROR)

OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

jobs = {}  # job_id -> { status, log, files }

app = Flask(__name__)

# ─── HTML (embedded) ──────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Viral Studio Pro</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Oswald:wght@400;700&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{--cyan:#00F2EA;--pink:#FF0050;--bg:#080c12;--surface:#0e1520;--border:#1a2535;--text:#c8d8e8;--dim:#4a6280}
    body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;min-height:100vh;overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,242,234,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,242,234,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
    .container{position:relative;z-index:1;max-width:820px;margin:0 auto;padding:40px 20px 80px}
    header{text-align:center;margin-bottom:48px}
    .badge{display:inline-block;font-size:11px;letter-spacing:3px;color:var(--cyan);border:1px solid var(--cyan);padding:4px 12px;margin-bottom:16px;animation:pulseBorder 2s infinite}
    @keyframes pulseBorder{0%,100%{box-shadow:0 0 0 0 rgba(0,242,234,.3)}50%{box-shadow:0 0 0 6px rgba(0,242,234,0)}}
    h1{font-family:'Oswald',sans-serif;font-size:clamp(28px,6vw,58px);font-weight:700;letter-spacing:2px;line-height:1.1;color:#fff}
    h1 span{color:var(--pink)}
    .sub{margin-top:12px;font-size:12px;color:var(--dim);letter-spacing:1px}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:24px;margin-bottom:20px;position:relative;overflow:hidden}
    .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.6}
    .card-label{font-size:11px;letter-spacing:2px;color:var(--cyan);margin-bottom:12px;display:flex;align-items:center;gap:8px}
    .card-label::after{content:'';flex:1;height:1px;background:var(--border)}
    textarea{width:100%;background:#060a10;border:1px solid var(--border);border-radius:2px;color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:13px;padding:14px;resize:vertical;min-height:130px;outline:none;transition:border-color .2s;line-height:1.7}
    textarea:focus{border-color:var(--cyan)}
    textarea::placeholder{color:var(--dim)}
    .hint{font-size:11px;color:var(--dim);margin-top:8px}
    .btn-execute{display:block;width:100%;padding:18px;background:var(--pink);color:#fff;font-family:'Oswald',sans-serif;font-size:20px;font-weight:700;letter-spacing:3px;border:none;cursor:pointer;transition:background .2s,transform .1s;clip-path:polygon(12px 0%,100% 0%,calc(100% - 12px) 100%,0% 100%)}
    .btn-execute:hover{background:#ff2266;transform:translateY(-1px)}
    .btn-execute:active{transform:translateY(0)}
    .btn-execute:disabled{background:#444;cursor:not-allowed;transform:none}
    .progress-wrap{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:20px}
    .progress-bar{height:100%;width:0%;background:linear-gradient(90deg,var(--cyan),var(--pink));transition:width .4s ease}
    .progress-bar.indeterminate{width:40%!important;animation:slide 1.2s infinite ease-in-out}
    @keyframes slide{0%{transform:translateX(-150%)}100%{transform:translateX(350%)}}
    .terminal{background:#04070d;border:1px solid var(--border);border-radius:2px;padding:16px;min-height:160px;max-height:260px;overflow-y:auto;font-size:12px;line-height:1.8;display:none;margin-top:20px}
    .terminal.visible{display:block}
    .log-line{color:var(--text)}
    .log-line.success{color:var(--cyan)}
    .log-line.error{color:var(--pink)}
    .cursor{display:inline-block;width:8px;height:14px;background:var(--cyan);animation:blink 1s infinite;vertical-align:middle;margin-left:4px}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
    .dl-section{display:none;margin-top:4px}
    .dl-section.visible{display:block}
    .dl-item{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:#04070d;border:1px solid var(--border);border-radius:2px;margin-bottom:8px;animation:fadeIn .4s ease;gap:10px}
    @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    .dl-name{font-size:12px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
    .dl-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;background:transparent;border:1px solid var(--cyan);color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:12px;letter-spacing:1px;text-decoration:none;white-space:nowrap;transition:background .2s,color .2s;flex-shrink:0}
    .dl-btn:hover{background:var(--cyan);color:#000}
    .status-chip{display:none;align-items:center;gap:6px;font-size:12px;padding:4px 10px;margin-top:14px;border-radius:2px}
    .status-chip.visible{display:inline-flex}
    .chip-running{background:rgba(0,242,234,.1);color:var(--cyan);border:1px solid var(--cyan)}
    .chip-done{background:rgba(255,224,102,.1);color:#ffe066;border:1px solid #ffe066}
    .dot{width:8px;height:8px;border-radius:50%;background:currentColor;animation:pulse 1.2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">VIRAL STUDIO PRO · PODCAST ENGINE</div>
    <h1>🎬 PODCAST<br><span>STABLE</span> ENGINE</h1>
    <p class="sub">Auto face-lock · Vertical crop · Background music blend</p>
  </header>

  <div class="card">
    <div class="card-label">INPUT LINKS</div>
    <textarea id="links" placeholder="https://youtube.com/watch?v=...&#10;https://youtube.com/watch?v=...&#10;&#10;// Satu link per baris"></textarea>
    <p class="hint">// Paste link YouTube, satu per baris</p>
  </div>

  <button class="btn-execute" id="startBtn" onclick="startJob()">⚡ MULAI EKSEKUSI STABIL</button>

  <div id="statusChip" class="status-chip">
    <span class="dot"></span><span id="statusText">Processing...</span>
  </div>

  <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>

  <div id="terminal" class="terminal"><span class="cursor"></span></div>

  <div id="dlSection" class="dl-section">
    <div class="card-label" style="margin-top:8px">OUTPUT FILES</div>
    <div id="dlList"></div>
  </div>
</div>

<script>
  let pollInterval=null,jobId=null,lastLog=0;

  async function startJob(){
    const links=document.getElementById('links').value.trim();
    if(!links)return alert('Masukkan minimal satu link YouTube!');
    const btn=document.getElementById('startBtn');
    btn.disabled=true; btn.textContent='⏳ PROCESSING...';
    lastLog=0;
    const t=document.getElementById('terminal');
    t.innerHTML='<span class="cursor"></span>'; t.classList.add('visible');
    document.getElementById('dlSection').classList.remove('visible');
    document.getElementById('dlList').innerHTML='';
    setChip('running','Memproses...');
    setBar('indeterminate');
    try{
      const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({links})});
      const d=await r.json();
      if(d.error){alert(d.error);reset();return;}
      jobId=d.job_id;
      pollInterval=setInterval(poll,1500);
    }catch(e){alert('Gagal: '+e.message);reset();}
  }

  async function poll(){
    if(!jobId)return;
    try{
      const r=await fetch('/status/'+jobId);
      const job=await r.json();
      const t=document.getElementById('terminal');
      const cur=t.querySelector('.cursor');
      job.log.slice(lastLog).forEach(line=>{
        const d=document.createElement('div');
        d.className='log-line'+(line.includes('✅')||line.includes('🎉')?' success':'')+(line.includes('❌')||line.includes('Gagal')?' error':'');
        d.textContent='› '+line;
        t.insertBefore(d,cur);
        t.scrollTop=t.scrollHeight;
      });
      lastLog=job.log.length;
      if(job.status==='done'){
        clearInterval(pollInterval);
        setChip('done','Selesai! ✓');
        setBar('done');
        showDownloads(job.files);
        resetBtn();
      }
    }catch(e){console.error(e);}
  }

  function showDownloads(files){
    if(!files||!files.length)return;
    const list=document.getElementById('dlList');
    files.forEach(f=>{
      const item=document.createElement('div');
      item.className='dl-item';
      item.innerHTML=`<span class="dl-name">📹 ${f}</span><a class="dl-btn" href="/download/${encodeURIComponent(f)}" download>⬇ DOWNLOAD</a>`;
      list.appendChild(item);
    });
    document.getElementById('dlSection').classList.add('visible');
  }

  function setChip(type,text){
    const c=document.getElementById('statusChip');
    c.className='status-chip visible chip-'+type;
    document.getElementById('statusText').textContent=text;
  }

  function setBar(state){
    const b=document.getElementById('progressBar');
    b.classList.remove('indeterminate');
    if(state==='indeterminate')b.classList.add('indeterminate');
    else if(state==='done')b.style.width='100%';
  }

  function reset(){resetBtn();document.getElementById('statusChip').className='status-chip';}
  function resetBtn(){const b=document.getElementById('startBtn');b.disabled=false;b.textContent='⚡ MULAI EKSEKUSI STABIL';}
</script>
</body>
</html>"""


# ─── CORE ENGINE LOGIC (tidak diubah) ────────────────────────────────────────

def find_main_face(clip):
    frames = [clip.get_frame(t) for t in np.linspace(0, min(5, clip.duration), 10)]
    x_positions = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            x_positions.append((x + w / 2) / frame.shape[1])
    return sum(x_positions) / len(x_positions) if x_positions else 0.5


def process_links(job_id, links):
    import yt_dlp
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
    import moviepy.audio.fx as afx

    def log(msg):
        print(f"[{job_id}] {msg}")
        jobs[job_id]["log"].append(msg)

    jobs[job_id]["status"] = "running"
    output_files = []

    for i, url in enumerate(links):
        try:
            temp_video = os.path.join(OUTPUT_DIR, f"temp_{job_id}_{i}.mp4")
            output_name = os.path.join(OUTPUT_DIR, f"PODCAST_STABLE_{job_id}_{i+1}.mp4")

            # 1. Download
            log(f"[{i+1}/{len(links)}] Downloading: {url}...")
            ydl_opts = {
                "outtmpl": temp_video,
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "quiet": True,
                "ffmpeg_location": CURRENT_DIR,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # 2. Re-framing
            clip = VideoFileClip(temp_video).subclipped(10, 40)
            w, h = clip.size
            target_w = h * (9 / 16)

            anchor_x_percent = find_main_face(clip)
            log(f"[{i+1}] Posisi wajah: {anchor_x_percent:.2%}")
            center_x = anchor_x_percent * w

            x1 = int(max(0, min(w - target_w, center_x - (target_w / 2))))
            x2 = int(x1 + target_w)

            log(f"[{i+1}] Proses Crop Vertikal...")
            final_clip = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)

            # 3. Audio
            final_audio = clip.audio
            music_path = os.path.join(CURRENT_DIR, "bg_music.mp3")
            if os.path.exists(music_path):
                bg_music = AudioFileClip(music_path).with_duration(clip.duration)
                bg_music = afx.multiply_volume(bg_music, 0.1)
                final_audio = CompositeAudioClip([clip.audio, bg_music])

            # 4. Render
            log(f"[{i+1}] Rendering: {os.path.basename(output_name)}...")
            final_clip.with_audio(final_audio).write_videofile(
                output_name, codec="libx264", audio_codec="aac",
                fps=24, threads=4, logger=None
            )

            clip.close()
            if os.path.exists(temp_video):
                os.remove(temp_video)

            log(f"[{i+1}] ✅ Selesai: {os.path.basename(output_name)}")
            output_files.append(os.path.basename(output_name))

        except Exception as e:
            log(f"[{i+1}] ❌ Gagal: {str(e)}")

    jobs[job_id]["files"] = output_files
    jobs[job_id]["status"] = "done"
    log("🎉 Semua proses selesai!")


# ─── FLASK ROUTES ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    links = [l.strip() for l in data.get("links", "").split("\n") if l.strip() and "http" in l]
    if not links:
        return jsonify({"error": "Tidak ada link valid"}), 400
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "log": [], "files": []}
    threading.Thread(target=process_links, args=(job_id, links), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job tidak ditemukan"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return "File tidak ditemukan", 404
    return send_file(path, as_attachment=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║   🎬 VIRAL STUDIO PRO — Web Edition      ║
╠══════════════════════════════════════════╣
║   Buka browser:  http://localhost:7575   ║
║   Stop server:   Ctrl+C                  ║
╚══════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=7575, debug=False)
