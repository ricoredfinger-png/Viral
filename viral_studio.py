#!/usr/bin/env python3
import os, uuid, logging, threading
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH  = os.path.join(CURRENT_DIR, "ffmpeg.exe")
COOKIES_PATH = os.path.join(CURRENT_DIR, "cookies.txt")
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH
logging.getLogger("moviepy").setLevel(logging.ERROR)

OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
jobs = {}
app  = Flask(__name__)

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
    :root{--cyan:#00F2EA;--pink:#FF0050;--bg:#080c12;--surface:#0e1520;--border:#1a2535;--text:#c8d8e8;--dim:#4a6280;--warn:#ffb020;--green:#00e676}
    body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;min-height:100vh;overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,242,234,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,242,234,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
    .wrap{position:relative;z-index:1;max-width:820px;margin:0 auto;padding:36px 18px 80px}
    header{text-align:center;margin-bottom:36px}
    .badge{display:inline-block;font-size:10px;letter-spacing:3px;color:var(--cyan);border:1px solid var(--cyan);padding:3px 12px;margin-bottom:14px;animation:pb 2.4s infinite}
    @keyframes pb{0%,100%{box-shadow:0 0 0 0 rgba(0,242,234,.35)}60%{box-shadow:0 0 0 7px rgba(0,242,234,0)}}
    h1{font-family:'Oswald',sans-serif;font-size:clamp(26px,6vw,54px);font-weight:700;letter-spacing:2px;line-height:1.1;color:#fff}
    h1 em{color:var(--pink);font-style:normal}
    .sub{margin-top:9px;font-size:11px;color:var(--dim);letter-spacing:1px}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:20px;margin-bottom:16px;position:relative;overflow:hidden}
    .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.55}
    .card.cw::before{background:linear-gradient(90deg,transparent,var(--warn),transparent)}
    .lbl{font-size:10px;letter-spacing:2px;color:var(--cyan);margin-bottom:11px;display:flex;align-items:center;gap:8px}
    .lbl.w{color:var(--warn)}
    .lbl::after{content:'';flex:1;height:1px;background:var(--border)}
    .ck-b{display:flex;align-items:center;gap:10px;padding:9px 14px;border-radius:2px;font-size:12px;margin-bottom:12px;border:1px solid}
    .ck-b.ok{background:rgba(0,230,118,.07);border-color:var(--green);color:var(--green)}
    .ck-b.no{background:rgba(255,176,32,.07);border-color:var(--warn);color:var(--warn)}
    .ci{font-size:17px;flex-shrink:0}
    .guide{display:none;margin:10px 0 4px}
    .guide ol{counter-reset:s;list-style:none;padding:0}
    .guide li{counter-increment:s;display:flex;gap:10px;padding:7px 0;font-size:12px;color:var(--dim);line-height:1.6;border-bottom:1px solid var(--border)}
    .guide li:last-child{border-bottom:none}
    .guide li::before{content:counter(s);min-width:20px;height:20px;border-radius:50%;background:var(--warn);color:#000;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px}
    .guide a{color:var(--warn);text-decoration:none}
    code{background:#04070d;padding:2px 6px;border-radius:2px;color:var(--cyan);font-size:11px}
    #dz{border:2px dashed var(--border);border-radius:3px;padding:18px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;font-size:13px;color:var(--dim);position:relative;margin-top:12px}
    #dz:hover,#dz.drag{border-color:var(--warn);background:rgba(255,176,32,.04);color:var(--warn)}
    #dz input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
    textarea{width:100%;background:#060a10;border:1px solid var(--border);border-radius:2px;color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:13px;padding:13px;resize:vertical;outline:none;transition:border-color .2s;line-height:1.7}
    textarea:focus{border-color:var(--cyan)}
    textarea::placeholder{color:var(--dim)}
    .hint{font-size:11px;color:var(--dim);margin-top:8px;line-height:1.7}
    .btn-go{display:block;width:100%;padding:17px;background:var(--pink);color:#fff;font-family:'Oswald',sans-serif;font-size:19px;font-weight:700;letter-spacing:3px;border:none;cursor:pointer;transition:background .2s,transform .1s;clip-path:polygon(12px 0%,100% 0%,calc(100% - 12px) 100%,0% 100%)}
    .btn-go:hover{background:#ff2266;transform:translateY(-1px)}
    .btn-go:disabled{background:#2a2a2a;cursor:not-allowed;transform:none;color:#555}
    .btn-s{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;background:transparent;border:1px solid var(--warn);color:var(--warn);font-family:'Share Tech Mono',monospace;font-size:11px;cursor:pointer;transition:background .2s,color .2s;margin-top:10px;letter-spacing:1px}
    .btn-s:hover{background:var(--warn);color:#000}
    .btn-s.red{border-color:var(--pink);color:var(--pink)}
    .btn-s.red:hover{background:var(--pink);color:#fff}
    .pw{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:18px}
    .pb{height:100%;width:0%;background:linear-gradient(90deg,var(--cyan),var(--pink));transition:width .4s}
    .pb.run{width:40%!important;animation:sw 1.2s infinite ease-in-out}
    @keyframes sw{0%{transform:translateX(-150%)}100%{transform:translateX(350%)}}
    .chip{display:none;align-items:center;gap:6px;font-size:11px;padding:4px 10px;margin-top:12px;border-radius:2px}
    .chip.on{display:inline-flex}
    .cr{background:rgba(0,242,234,.1);color:var(--cyan);border:1px solid var(--cyan)}
    .co{background:rgba(0,230,118,.1);color:var(--green);border:1px solid var(--green)}
    .ce{background:rgba(255,0,80,.1);color:var(--pink);border:1px solid var(--pink)}
    .dot{width:7px;height:7px;border-radius:50%;background:currentColor;animation:dp 1.2s infinite}
    @keyframes dp{0%,100%{opacity:1}50%{opacity:.2}}
    .term{background:#04070d;border:1px solid var(--border);border-radius:2px;padding:14px;min-height:140px;max-height:250px;overflow-y:auto;font-size:12px;line-height:1.85;display:none;margin-top:18px}
    .term.on{display:block}
    .ln{color:var(--text)}
    .lok{color:var(--cyan)}
    .ler{color:var(--pink)}
    .lwn{color:var(--warn)}
    .cur{display:inline-block;width:8px;height:13px;background:var(--cyan);animation:bl 1s infinite;vertical-align:middle;margin-left:3px}
    @keyframes bl{0%,100%{opacity:1}50%{opacity:0}}
    .dlw{display:none;margin-top:6px}
    .dlw.on{display:block}
    .dli{display:flex;align-items:center;justify-content:space-between;padding:11px 15px;background:#04070d;border:1px solid var(--border);border-radius:2px;margin-bottom:7px;animation:fi .35s ease;gap:10px}
    @keyframes fi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
    .dln{font-size:12px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
    .dlb{display:inline-flex;align-items:center;gap:5px;padding:7px 15px;background:transparent;border:1px solid var(--cyan);color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:1px;text-decoration:none;white-space:nowrap;transition:background .2s,color .2s;flex-shrink:0}
    .dlb:hover{background:var(--cyan);color:#000}
    .ebox{background:#04070d;border:1px solid var(--pink);border-radius:2px;padding:12px 14px;font-size:11px;color:var(--pink);margin-top:12px;line-height:1.7;display:none}
    .ebox.on{display:block}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">VIRAL STUDIO PRO · PODCAST ENGINE</div>
    <h1>&#x1F3AC; PODCAST<br><em>STABLE</em> ENGINE</h1>
    <p class="sub">Auto face-lock &middot; Vertical 9:16 crop &middot; BG music blend</p>
  </header>

  <div class="card cw">
    <div class="lbl w">&#x1F36A; YOUTUBE COOKIES</div>
    <div id="ckB" class="ck-b no"><span class="ci">&#x26A0;&#xFE0F;</span><span id="ckM">Cookies belum ada &mdash; YouTube akan blokir download dari VPS</span></div>
    <div class="guide" id="guide">
      <ol>
        <li>Install ekstensi <a href="https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc" target="_blank">Get cookies.txt LOCALLY</a> di Chrome/Firefox</li>
        <li>Login ke <a href="https://youtube.com" target="_blank">youtube.com</a></li>
        <li>Klik ikon ekstensi &rarr; <strong style="color:#fff">Export as cookies.txt</strong></li>
        <li>Upload file <code>cookies.txt</code> ke kolom di bawah</li>
      </ol>
    </div>
    <button class="btn-s" id="guideBtn" onclick="tg()">&#x1F4D6; CARA DAPAT COOKIES</button>
    <div id="dz">
      <input type="file" accept=".txt" onchange="upCk(this.files[0])">
      &#x1F4C2; Drop <code>cookies.txt</code> di sini atau klik untuk pilih file
    </div>
    <button class="btn-s red" id="delBtn" style="display:none" onclick="delCk()">&#x1F5D1; HAPUS COOKIES</button>
  </div>

  <div class="card">
    <div class="lbl">INPUT LINKS</div>
    <textarea id="links" rows="5" placeholder="https://youtube.com/watch?v=...&#10;https://youtu.be/xxxxx&#10;&#10;// Satu link per baris"></textarea>
    <p class="hint">&#x26A0;&#xFE0F; Pastikan link ke <strong style="color:#fff">video biasa</strong> (bukan Community Post / gambar).<br>&#x2705; Contoh: <code>youtube.com/watch?v=...</code> atau <code>youtu.be/...</code></p>
  </div>

  <button class="btn-go" id="startBtn" onclick="go()">&#x26A1; MULAI EKSEKUSI STABIL</button>
  <div id="chip" class="chip"><span class="dot"></span><span id="chipT"></span></div>
  <div class="pw"><div class="pb" id="pb"></div></div>
  <div class="term" id="term"><span class="cur"></span></div>
  <div class="ebox" id="ebox"></div>
  <div class="dlw" id="dlw">
    <div class="lbl" style="margin-top:8px">OUTPUT FILES</div>
    <div id="dll"></div>
  </div>
</div>

<script>
let tmr=null,jid=null,n=0;

async function ckSt(){
  const d=await(await fetch('/cookie-status')).json();
  const b=document.getElementById('ckB');
  const m=document.getElementById('ckM');
  const del=document.getElementById('delBtn');
  if(d.exists){b.className='ck-b ok';b.querySelector('.ci').textContent='\u2705';m.textContent='Cookies aktif ('+d.size+') \u2014 siap download YouTube';del.style.display='inline-flex';}
  else{b.className='ck-b no';b.querySelector('.ci').textContent='\u26a0\ufe0f';m.textContent='Cookies belum ada \u2014 YouTube akan blokir download dari VPS';del.style.display='none';}
}
async function upCk(f){
  if(!f)return;
  const t=await f.text();
  if(t.length<50){alert('File tidak valid');return;}
  const fd=new FormData();fd.append('file',f);
  const d=await(await fetch('/upload-cookies',{method:'POST',body:fd})).json();
  d.ok?ckSt():alert('Upload gagal: '+d.error);
}
async function delCk(){if(!confirm('Hapus cookies?'))return;await fetch('/delete-cookies',{method:'POST'});ckSt();}
function tg(){const g=document.getElementById('guide');const b=document.getElementById('guideBtn');const o=g.style.display==='none'||!g.style.display;g.style.display=o?'block':'none';b.textContent=o?'\u1F53C TUTUP PANDUAN':'\u1F4D6 CARA DAPAT COOKIES';}
const dz=document.getElementById('dz');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');const f=e.dataTransfer.files[0];if(f)upCk(f);});

async function go(){
  const raw=document.getElementById('links').value.trim();
  if(!raw)return alert('Masukkan link!');
  const btn=document.getElementById('startBtn');
  btn.disabled=true;btn.textContent='\u23F3 MEMPROSES...';
  n=0;
  const t=document.getElementById('term');t.innerHTML='<span class="cur"></span>';t.classList.add('on');
  document.getElementById('dlw').classList.remove('on');
  document.getElementById('dll').innerHTML='';
  document.getElementById('ebox').classList.remove('on');
  setChip('r','Memproses...');setPb('r');
  try{
    const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({links:raw})});
    const d=await r.json();
    if(d.error){alert(d.error);rst();return;}
    jid=d.job_id;if(tmr)clearInterval(tmr);tmr=setInterval(poll,1800);
  }catch(e){alert('Gagal: '+e.message);rst();}
}

async function poll(){
  if(!jid)return;
  try{
    const job=await(await fetch('/status/'+jid)).json();
    const t=document.getElementById('term');const cur=t.querySelector('.cur');
    job.log.slice(n).forEach(line=>{
      const d=document.createElement('div');
      const l=line.toLowerCase();
      d.className='ln'+(l.includes('\u2705')||l.includes('\u1F389')?' lok':l.includes('\u274C')||l.includes('gagal')?' ler':l.includes('\u26A0')?' lwn':'');
      d.textContent='\u203A '+line;t.insertBefore(d,cur);t.scrollTop=t.scrollHeight;
    });
    n=job.log.length;
    if(job.status==='done'){
      clearInterval(tmr);tmr=null;
      const hasF=job.files&&job.files.length>0;
      const hasE=job.log.some(l=>l.includes('\u274C'));
      if(hasF){setChip('o','\u2705 Selesai! '+job.files.length+' file siap');}
      else if(hasE){setChip('e','\u274C Gagal \u2014 lihat log');showErrTip(job.log);}
      else{setChip('o','Selesai');}
      setPb('d');
      if(hasF)showDL(job.files);
      rstBtn();
    }
  }catch(e){console.error(e);}
}

function showErrTip(logs){
  const box=document.getElementById('ebox');
  const last=logs.filter(l=>l.includes('\u274C')).pop()||'';
  let tip='';
  if(last.includes('Sign in')||last.includes('bot'))tip='\uD83D\uDD11 Cookies tidak valid / expired. Export ulang dari browser yang sudah login YouTube.';
  else if(last.includes('format')||last.includes('images'))tip='\uD83C\uDFA5 Link bukan video biasa. Pastikan bukan Community Post, Shorts bergambar, atau Playlist.';
  else if(last.includes('429'))tip='\uD83D\uDEA6 Rate limit YouTube. Tunggu 5 menit lalu coba lagi.';
  else if(last.includes('pendek')||last.includes('short'))tip='\u23F1 Video terlalu pendek. Gunakan video minimal 42 detik.';
  else tip='Periksa log terminal di atas untuk detail error.';
  box.textContent='\uD83D\uDCA1 '+tip;box.classList.add('on');
}

function showDL(files){
  const list=document.getElementById('dll');
  files.forEach(f=>{
    const i=document.createElement('div');i.className='dli';
    i.innerHTML='<span class="dln">\uD83C\uDFA5 '+f+'</span><a class="dlb" href="/download/'+encodeURIComponent(f)+'" download>\u2B07 DOWNLOAD</a>';
    list.appendChild(i);
  });
  document.getElementById('dlw').classList.add('on');
}

function setChip(t,txt){
  const c=document.getElementById('chip');
  c.className='chip on '+(t==='r'?'cr':t==='o'?'co':'ce');
  document.getElementById('chipT').textContent=txt;
}
function setPb(s){const p=document.getElementById('pb');p.classList.remove('run');if(s==='r')p.classList.add('run');else if(s==='d')p.style.width='100%';}
function rst(){rstBtn();document.getElementById('chip').className='chip';}
function rstBtn(){const b=document.getElementById('startBtn');b.disabled=false;b.textContent='\u26A1 MULAI EKSEKUSI STABIL';}
ckSt();
</script>
</body>
</html>"""

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
    has_cookies  = os.path.exists(COOKIES_PATH)

    if has_cookies:
        log("\U0001f36a Cookies aktif")
    else:
        log("\u26a0\ufe0f  Tidak ada cookies")

    BASE_OPTS = {
        "quiet": True,
        "ffmpeg_location": CURRENT_DIR,
        "retries": 3,
        "merge_output_format": "mp4",
        "http_headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
    }
    if has_cookies:
        BASE_OPTS["cookiefile"] = COOKIES_PATH

    FORMAT_PRIORITY = [
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]",
        "bestvideo+bestaudio",
        "best[ext=mp4]",
        "best",
    ]

    for i, url in enumerate(links):
        temp_video  = os.path.join(OUTPUT_DIR, f"temp_{job_id}_{i}.mp4")
        output_name = os.path.join(OUTPUT_DIR, f"PODCAST_STABLE_{job_id}_{i+1}.mp4")
        try:
            # Step 1: cek info
            log(f"[{i+1}/{len(links)}] Memeriksa: {url}")
            with yt_dlp.YoutubeDL({**BASE_OPTS, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)

            duration = info.get("duration", 0)
            if not duration:
                raise ValueError("Bukan video YouTube biasa (duration=0). Pastikan link ke video, bukan Post/gambar.")
            if duration < 42:
                raise ValueError(f"Video terlalu pendek ({duration}s). Minimal 42 detik diperlukan.")
            log(f"[{i+1}] Video OK: \"{info.get('title','?')[:50]}\" ({duration}s)")

            # Step 2: download dengan format fallback
            log(f"[{i+1}] Downloading...")
            downloaded = False
            last_err = ""
            for fmt in FORMAT_PRIORITY:
                try:
                    opts = {**BASE_OPTS, "outtmpl": temp_video, "format": fmt}
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([url])
                    if os.path.exists(temp_video) and os.path.getsize(temp_video) > 10000:
                        log(f"[{i+1}] Download OK (fmt: {fmt[:45]})")
                        downloaded = True
                        break
                except Exception as e:
                    last_err = str(e)
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
            if not downloaded:
                raise RuntimeError(f"Semua format gagal. Error: {last_err}")

            # Step 3: re-framing (logic asli)
            clip = VideoFileClip(temp_video).subclipped(10, 40)
            w, h = clip.size
            target_w = h * (9 / 16)
            anchor_x_percent = find_main_face(clip)
            log(f"[{i+1}] Posisi wajah: {anchor_x_percent:.2%}")
            center_x = anchor_x_percent * w
            x1 = int(max(0, min(w - target_w, center_x - (target_w / 2))))
            x2 = int(x1 + target_w)
            log(f"[{i+1}] Crop 9:16...")
            final_clip = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)

            # Step 4: audio
            final_audio = clip.audio
            music_path = os.path.join(CURRENT_DIR, "bg_music.mp3")
            if os.path.exists(music_path):
                bg = AudioFileClip(music_path).with_duration(clip.duration)
                bg = afx.multiply_volume(bg, 0.1)
                final_audio = CompositeAudioClip([clip.audio, bg])

            # Step 5: render
            log(f"[{i+1}] Rendering...")
            final_clip.with_audio(final_audio).write_videofile(
                output_name, codec="libx264", audio_codec="aac", fps=24, threads=4, logger=None
            )
            clip.close()
            if os.path.exists(temp_video):
                os.remove(temp_video)
            log(f"[{i+1}] \u2705 Selesai: {os.path.basename(output_name)}")
            output_files.append(os.path.basename(output_name))

        except Exception as e:
            log(f"[{i+1}] \u274c Gagal: {str(e)}")
            if os.path.exists(temp_video):
                try: os.remove(temp_video)
                except: pass

    jobs[job_id]["files"]  = output_files
    jobs[job_id]["status"] = "done"
    log("\U0001f389 Semua proses selesai!")


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/cookie-status")
def cookie_status():
    exists = os.path.exists(COOKIES_PATH)
    size = ""
    if exists:
        b = os.path.getsize(COOKIES_PATH)
        size = f"{b//1024} KB" if b > 1024 else f"{b} B"
    return jsonify({"exists": exists, "size": size})

@app.route("/upload-cookies", methods=["POST"])
def upload_cookies():
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "Tidak ada file"}), 400
    content = f.read().decode("utf-8", errors="ignore")
    if len(content) < 50:
        return jsonify({"ok": False, "error": "File kosong"}), 400
    with open(COOKIES_PATH, "w", encoding="utf-8") as fp:
        fp.write(content)
    return jsonify({"ok": True})

@app.route("/delete-cookies", methods=["POST"])
def delete_cookies():
    if os.path.exists(COOKIES_PATH):
        os.remove(COOKIES_PATH)
    return jsonify({"ok": True})

@app.route("/start", methods=["POST"])
def start():
    data  = request.json or {}
    links = [l.strip() for l in data.get("links","").split("\n") if l.strip() and "http" in l]
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
        return jsonify({"error": "Tidak ditemukan"}), 404
    return jsonify(job)

@app.route("/download/<filename>")
def download(filename):
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(path):
        return "File tidak ditemukan", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("\n\U0001f3ac Viral Studio Pro running on http://localhost:7575\n")
    app.run(host="0.0.0.0", port=7575, debug=False)
