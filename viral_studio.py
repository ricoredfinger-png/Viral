#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viral Studio Pro - Web Edition
Jalankan : python viral_studio.py
Buka     : http://localhost:7575
"""

import os
import uuid
import logging
import threading

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, Response

# ── Konfigurasi ───────────────────────────────────────────────────────────────
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

# ── HTML ──────────────────────────────────────────────────────────────────────
# PENTING: semua emoji ditulis langsung sebagai karakter UTF-8.
# Jangan gunakan \uD83X (surrogate) atau \u1FXXX (5-digit) di dalam string Python
# karena akan menyebabkan UnicodeEncodeError saat encode ke UTF-8.
HTML = (
    '<!DOCTYPE html>\n'
    '<html lang="id">\n'
    '<head>\n'
    '<meta charset="UTF-8"/>\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1.0"/>\n'
    '<title>Viral Studio Pro</title>\n'
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono'
    '&family=Oswald:wght@400;700&display=swap" rel="stylesheet">\n'
    '<style>\n'
    '*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n'
    ':root{'
        '--cyan:#00F2EA;--pink:#FF0050;--bg:#080c12;--surf:#0e1520;'
        '--brd:#1a2535;--txt:#c8d8e8;--dim:#4a6280;--warn:#ffb020;--ok:#00e676'
    '}\n'
    'body{background:var(--bg);color:var(--txt);'
        'font-family:\'Share Tech Mono\',monospace;min-height:100vh;overflow-x:hidden}\n'
    'body::before{content:\'\';position:fixed;inset:0;'
        'background-image:linear-gradient(rgba(0,242,234,.03) 1px,transparent 1px),'
        'linear-gradient(90deg,rgba(0,242,234,.03) 1px,transparent 1px);'
        'background-size:40px 40px;pointer-events:none;z-index:0}\n'
    '.wrap{position:relative;z-index:1;max-width:820px;margin:0 auto;padding:36px 18px 80px}\n'
    'header{text-align:center;margin-bottom:36px}\n'
    '.badge{display:inline-block;font-size:10px;letter-spacing:3px;color:var(--cyan);'
        'border:1px solid var(--cyan);padding:3px 12px;margin-bottom:14px;animation:pb 2.4s infinite}\n'
    '@keyframes pb{0%,100%{box-shadow:0 0 0 0 rgba(0,242,234,.35)}'
        '60%{box-shadow:0 0 0 7px rgba(0,242,234,0)}}\n'
    'h1{font-family:\'Oswald\',sans-serif;font-size:clamp(26px,6vw,54px);'
        'font-weight:700;letter-spacing:2px;line-height:1.1;color:#fff}\n'
    'h1 em{color:var(--pink);font-style:normal}\n'
    '.sub{margin-top:9px;font-size:11px;color:var(--dim);letter-spacing:1px}\n'
    '.card{background:var(--surf);border:1px solid var(--brd);border-radius:4px;'
        'padding:20px;margin-bottom:16px;position:relative;overflow:hidden}\n'
    '.card::before{content:\'\';position:absolute;top:0;left:0;right:0;height:2px;'
        'background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.55}\n'
    '.card.cw::before{background:linear-gradient(90deg,transparent,var(--warn),transparent)}\n'
    '.lbl{font-size:10px;letter-spacing:2px;color:var(--cyan);margin-bottom:11px;'
        'display:flex;align-items:center;gap:8px}\n'
    '.lbl.w{color:var(--warn)}\n'
    '.lbl::after{content:\'\';flex:1;height:1px;background:var(--brd)}\n'
    '.ck-b{display:flex;align-items:center;gap:10px;padding:9px 14px;border-radius:2px;'
        'font-size:12px;margin-bottom:12px;border:1px solid}\n'
    '.ck-b.ok{background:rgba(0,230,118,.07);border-color:var(--ok);color:var(--ok)}\n'
    '.ck-b.no{background:rgba(255,176,32,.07);border-color:var(--warn);color:var(--warn)}\n'
    '.ci{font-size:17px;flex-shrink:0}\n'
    '.guide{display:none;margin:10px 0 4px}\n'
    '.guide ol{counter-reset:s;list-style:none;padding:0}\n'
    '.guide li{counter-increment:s;display:flex;gap:10px;padding:7px 0;'
        'font-size:12px;color:var(--dim);line-height:1.6;border-bottom:1px solid var(--brd)}\n'
    '.guide li:last-child{border-bottom:none}\n'
    '.guide li::before{content:counter(s);min-width:20px;height:20px;border-radius:50%;'
        'background:var(--warn);color:#000;display:flex;align-items:center;'
        'justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px}\n'
    '.guide a{color:var(--warn);text-decoration:none}\n'
    'code{background:#04070d;padding:2px 6px;border-radius:2px;color:var(--cyan);font-size:11px}\n'
    '#dz{border:2px dashed var(--brd);border-radius:3px;padding:18px;text-align:center;'
        'cursor:pointer;transition:border-color .2s,background .2s;font-size:13px;'
        'color:var(--dim);position:relative;margin-top:12px}\n'
    '#dz:hover,#dz.drag{border-color:var(--warn);background:rgba(255,176,32,.04);color:var(--warn)}\n'
    '#dz input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}\n'
    'textarea{width:100%;background:#060a10;border:1px solid var(--brd);border-radius:2px;'
        'color:var(--cyan);font-family:\'Share Tech Mono\',monospace;font-size:13px;'
        'padding:13px;resize:vertical;outline:none;transition:border-color .2s;line-height:1.7}\n'
    'textarea:focus{border-color:var(--cyan)}\n'
    'textarea::placeholder{color:var(--dim)}\n'
    '.hint{font-size:11px;color:var(--dim);margin-top:8px;line-height:1.7}\n'
    '.btn-go{display:block;width:100%;padding:17px;background:var(--pink);color:#fff;'
        'font-family:\'Oswald\',sans-serif;font-size:19px;font-weight:700;letter-spacing:3px;'
        'border:none;cursor:pointer;transition:background .2s,transform .1s;'
        'clip-path:polygon(12px 0%,100% 0%,calc(100% - 12px) 100%,0% 100%)}\n'
    '.btn-go:hover{background:#ff2266;transform:translateY(-1px)}\n'
    '.btn-go:disabled{background:#2a2a2a;cursor:not-allowed;transform:none;color:#555}\n'
    '.btn-s{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;'
        'background:transparent;border:1px solid var(--warn);color:var(--warn);'
        'font-family:\'Share Tech Mono\',monospace;font-size:11px;cursor:pointer;'
        'transition:background .2s,color .2s;margin-top:10px;letter-spacing:1px}\n'
    '.btn-s:hover{background:var(--warn);color:#000}\n'
    '.btn-s.red{border-color:var(--pink);color:var(--pink)}\n'
    '.btn-s.red:hover{background:var(--pink);color:#fff}\n'
    '.pw{height:4px;background:var(--brd);border-radius:2px;overflow:hidden;margin-top:18px}\n'
    '.pb{height:100%;width:0%;background:linear-gradient(90deg,var(--cyan),var(--pink));transition:width .4s}\n'
    '.pb.run{width:40%!important;animation:sw 1.2s infinite ease-in-out}\n'
    '@keyframes sw{0%{transform:translateX(-150%)}100%{transform:translateX(350%)}}\n'
    '.chip{display:none;align-items:center;gap:6px;font-size:11px;'
        'padding:4px 10px;margin-top:12px;border-radius:2px}\n'
    '.chip.on{display:inline-flex}\n'
    '.cr{background:rgba(0,242,234,.1);color:var(--cyan);border:1px solid var(--cyan)}\n'
    '.co{background:rgba(0,230,118,.1);color:var(--ok);border:1px solid var(--ok)}\n'
    '.ce{background:rgba(255,0,80,.1);color:var(--pink);border:1px solid var(--pink)}\n'
    '.dot{width:7px;height:7px;border-radius:50%;background:currentColor;animation:dp 1.2s infinite}\n'
    '@keyframes dp{0%,100%{opacity:1}50%{opacity:.2}}\n'
    '.term{background:#04070d;border:1px solid var(--brd);border-radius:2px;padding:14px;'
        'min-height:140px;max-height:250px;overflow-y:auto;font-size:12px;'
        'line-height:1.85;display:none;margin-top:18px}\n'
    '.term.on{display:block}\n'
    '.ln{color:var(--txt)}\n'
    '.lok{color:var(--ok)}\n'
    '.ler{color:var(--pink)}\n'
    '.lwn{color:var(--warn)}\n'
    '.cur{display:inline-block;width:8px;height:13px;background:var(--cyan);'
        'animation:bl 1s infinite;vertical-align:middle;margin-left:3px}\n'
    '@keyframes bl{0%,100%{opacity:1}50%{opacity:0}}\n'
    '.dlw{display:none;margin-top:6px}\n'
    '.dlw.on{display:block}\n'
    '.dli{display:flex;align-items:center;justify-content:space-between;'
        'padding:11px 15px;background:#04070d;border:1px solid var(--brd);'
        'border-radius:2px;margin-bottom:7px;animation:fi .35s ease;gap:10px}\n'
    '@keyframes fi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}\n'
    '.dln{font-size:12px;color:var(--txt);overflow:hidden;text-overflow:ellipsis;'
        'white-space:nowrap;flex:1;min-width:0}\n'
    '.dlb{display:inline-flex;align-items:center;gap:5px;padding:7px 15px;'
        'background:transparent;border:1px solid var(--cyan);color:var(--cyan);'
        'font-family:\'Share Tech Mono\',monospace;font-size:11px;letter-spacing:1px;'
        'text-decoration:none;white-space:nowrap;transition:background .2s,color .2s;flex-shrink:0}\n'
    '.dlb:hover{background:var(--cyan);color:#000}\n'
    '.ebox{background:#04070d;border:1px solid var(--pink);border-radius:2px;'
        'padding:12px 14px;font-size:11px;color:var(--pink);'
        'margin-top:12px;line-height:1.7;display:none}\n'
    '.ebox.on{display:block}\n'
    '</style>\n'
    '</head>\n'
    '<body>\n'
    '<div class="wrap">\n'
    '  <header>\n'
    '    <div class="badge">VIRAL STUDIO PRO &middot; PODCAST ENGINE</div>\n'
    '    <h1>&#x1F3AC; PODCAST<br><em>STABLE</em> ENGINE</h1>\n'
    '    <p class="sub">Auto face-lock &middot; Vertical 9:16 crop &middot; BG music blend</p>\n'
    '  </header>\n'
    '\n'
    '  <!-- COOKIES -->\n'
    '  <div class="card cw">\n'
    '    <div class="lbl w">&#x1F36A; YOUTUBE COOKIES</div>\n'
    '    <div id="ckB" class="ck-b no">'
        '<span class="ci">&#x26A0;&#xFE0F;</span>'
        '<span id="ckM">Cookies belum ada &mdash; YouTube akan blokir download dari VPS</span>'
    '</div>\n'
    '    <div class="guide" id="guide">\n'
    '      <ol>\n'
    '        <li>Install ekstensi '
            '<a href="https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc" target="_blank">'
            'Get cookies.txt LOCALLY</a> di Chrome/Firefox</li>\n'
    '        <li>Login ke <a href="https://youtube.com" target="_blank">youtube.com</a></li>\n'
    '        <li>Klik ikon ekstensi &rarr; <strong style="color:#fff">Export as cookies.txt</strong></li>\n'
    '        <li>Upload file <code>cookies.txt</code> ke kolom di bawah</li>\n'
    '      </ol>\n'
    '    </div>\n'
    '    <button class="btn-s" id="guideBtn" onclick="tg()">&#x1F4D6; CARA DAPAT COOKIES</button>\n'
    '    <div id="dz">\n'
    '      <input type="file" accept=".txt" onchange="upCk(this.files[0])">\n'
    '      &#x1F4C2; Drop <code>cookies.txt</code> di sini atau klik untuk pilih file\n'
    '    </div>\n'
    '    <button class="btn-s red" id="delBtn" style="display:none" onclick="delCk()">'
        '&#x1F5D1; HAPUS COOKIES'
    '</button>\n'
    '  </div>\n'
    '\n'
    '  <!-- LINKS -->\n'
    '  <div class="card">\n'
    '    <div class="lbl">INPUT LINKS</div>\n'
    '    <textarea id="links" rows="5" '
        'placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/xxxxx\n\n// Satu link per baris">'
    '</textarea>\n'
    '    <p class="hint">&#x26A0; Pastikan link ke <strong style="color:#fff">video biasa</strong> '
        '(bukan Community Post / gambar).<br>'
        '&#x2705; Contoh: <code>youtube.com/watch?v=...</code></p>\n'
    '  </div>\n'
    '\n'
    '  <button class="btn-go" id="startBtn" onclick="go()">&#x26A1; MULAI EKSEKUSI STABIL</button>\n'
    '  <div id="chip" class="chip"><span class="dot"></span><span id="chipT"></span></div>\n'
    '  <div class="pw"><div class="pb" id="pb"></div></div>\n'
    '  <div class="term" id="term"><span class="cur"></span></div>\n'
    '  <div class="ebox" id="ebox"></div>\n'
    '  <div class="dlw" id="dlw">\n'
    '    <div class="lbl" style="margin-top:8px">OUTPUT FILES</div>\n'
    '    <div id="dll"></div>\n'
    '  </div>\n'
    '</div>\n'
    '\n'
    '<script>\n'
    'var tmr=null,jid=null,nLog=0;\n'
    '\n'
    '/* -- Cookies -- */\n'
    'async function ckSt(){\n'
    '  var d=await(await fetch("/cookie-status")).json();\n'
    '  var b=document.getElementById("ckB");\n'
    '  var m=document.getElementById("ckM");\n'
    '  var db=document.getElementById("delBtn");\n'
    '  if(d.exists){\n'
    '    b.className="ck-b ok";\n'
    '    b.querySelector(".ci").textContent="OK";\n'
    '    m.textContent="Cookies aktif ("+d.size+") - siap download YouTube";\n'
    '    db.style.display="inline-flex";\n'
    '  } else {\n'
    '    b.className="ck-b no";\n'
    '    b.querySelector(".ci").textContent="(!)";\n'
    '    m.textContent="Cookies belum ada - YouTube akan blokir download dari VPS";\n'
    '    db.style.display="none";\n'
    '  }\n'
    '}\n'
    'async function upCk(f){\n'
    '  if(!f)return;\n'
    '  var t=await f.text();\n'
    '  if(t.length<50){alert("File tidak valid");return;}\n'
    '  var fd=new FormData();fd.append("file",f);\n'
    '  var d=await(await fetch("/upload-cookies",{method:"POST",body:fd})).json();\n'
    '  if(d.ok)ckSt();else alert("Upload gagal: "+d.error);\n'
    '}\n'
    'async function delCk(){\n'
    '  if(!confirm("Hapus cookies?"))return;\n'
    '  await fetch("/delete-cookies",{method:"POST"});\n'
    '  ckSt();\n'
    '}\n'
    'function tg(){\n'
    '  var g=document.getElementById("guide");\n'
    '  var b=document.getElementById("guideBtn");\n'
    '  var open=(g.style.display==="none"||g.style.display==="");\n'
    '  g.style.display=open?"block":"none";\n'
    '  b.textContent=open?"[ TUTUP PANDUAN ]":"[ CARA DAPAT COOKIES ]";\n'
    '}\n'
    'var dz=document.getElementById("dz");\n'
    'dz.addEventListener("dragover",function(e){e.preventDefault();dz.classList.add("drag")});\n'
    'dz.addEventListener("dragleave",function(){dz.classList.remove("drag")});\n'
    'dz.addEventListener("drop",function(e){\n'
    '  e.preventDefault();dz.classList.remove("drag");\n'
    '  var f=e.dataTransfer.files[0];if(f)upCk(f);\n'
    '});\n'
    '\n'
    '/* -- Job -- */\n'
    'async function go(){\n'
    '  var raw=document.getElementById("links").value.trim();\n'
    '  if(!raw)return alert("Masukkan link!");\n'
    '  var btn=document.getElementById("startBtn");\n'
    '  btn.disabled=true;btn.textContent="[MEMPROSES...]";\n'
    '  nLog=0;\n'
    '  var t=document.getElementById("term");\n'
    '  t.innerHTML=\'<span class="cur"></span>\';\n'
    '  t.classList.add("on");\n'
    '  document.getElementById("dlw").classList.remove("on");\n'
    '  document.getElementById("dll").innerHTML="";\n'
    '  document.getElementById("ebox").classList.remove("on");\n'
    '  setChip("r","Memproses...");\n'
    '  setPb("r");\n'
    '  try{\n'
    '    var r=await fetch("/start",{\n'
    '      method:"POST",\n'
    '      headers:{"Content-Type":"application/json"},\n'
    '      body:JSON.stringify({links:raw})\n'
    '    });\n'
    '    var d=await r.json();\n'
    '    if(d.error){alert(d.error);rst();return;}\n'
    '    jid=d.job_id;\n'
    '    if(tmr)clearInterval(tmr);\n'
    '    tmr=setInterval(poll,1800);\n'
    '  }catch(e){alert("Gagal: "+e.message);rst();}\n'
    '}\n'
    '\n'
    'async function poll(){\n'
    '  if(!jid)return;\n'
    '  try{\n'
    '    var job=await(await fetch("/status/"+jid)).json();\n'
    '    var t=document.getElementById("term");\n'
    '    var cur=t.querySelector(".cur");\n'
    '    var newLines=job.log.slice(nLog);\n'
    '    newLines.forEach(function(line){\n'
    '      var el=document.createElement("div");\n'
    '      var lo=line.toLowerCase();\n'
    '      var cls="ln";\n'
    '      if(lo.indexOf("selesai")>=0||lo.indexOf("[ok]")>=0||lo.indexOf("cookies aktif")>=0)cls="ln lok";\n'
    '      else if(lo.indexOf("gagal")>=0||lo.indexOf("error")>=0||lo.indexOf("[err]")>=0)cls="ln ler";\n'
    '      else if(lo.indexOf("warning")>=0||lo.indexOf("[warn]")>=0||lo.indexOf("tidak ada")>=0)cls="ln lwn";\n'
    '      el.className=cls;\n'
    '      el.textContent="> "+line;\n'
    '      t.insertBefore(el,cur);\n'
    '      t.scrollTop=t.scrollHeight;\n'
    '    });\n'
    '    nLog=job.log.length;\n'
    '    if(job.status==="done"){\n'
    '      clearInterval(tmr);tmr=null;\n'
    '      var hasF=job.files&&job.files.length>0;\n'
    '      var hasE=job.log.some(function(l){return l.toLowerCase().indexOf("gagal")>=0||l.toLowerCase().indexOf("error")>=0;});\n'
    '      if(hasF){setChip("o","[OK] Selesai! "+job.files.length+" file siap");}\n'
    '      else if(hasE){setChip("e","[GAGAL] Lihat log");showErr(job.log);}\n'
    '      else{setChip("o","Selesai");}\n'
    '      setPb("d");\n'
    '      if(hasF)showDL(job.files);\n'
    '      rstBtn();\n'
    '    }\n'
    '  }catch(e){console.error(e);}\n'
    '}\n'
    '\n'
    'function showErr(logs){\n'
    '  var box=document.getElementById("ebox");\n'
    '  var last="";\n'
    '  for(var i=logs.length-1;i>=0;i--){\n'
    '    if(logs[i].toLowerCase().indexOf("gagal")>=0||logs[i].toLowerCase().indexOf("error")>=0){\n'
    '      last=logs[i];break;\n'
    '    }\n'
    '  }\n'
    '  var tip="";\n'
    '  if(last.indexOf("Sign in")>=0||last.indexOf("bot")>=0)\n'
    '    tip="[KUNCI] Cookies tidak valid / expired. Export ulang dari browser yang sudah login YouTube.";\n'
    '  else if(last.indexOf("format")>=0||last.indexOf("images")>=0)\n'
    '    tip="[VIDEO] Link bukan video biasa. Pastikan bukan Community Post, Shorts bergambar, atau Playlist.";\n'
    '  else if(last.indexOf("429")>=0)\n'
    '    tip="[BATAS] Rate limit YouTube. Tunggu 5 menit lalu coba lagi.";\n'
    '  else if(last.indexOf("pendek")>=0)\n'
    '    tip="[WAKTU] Video terlalu pendek. Gunakan video minimal 42 detik.";\n'
    '  else\n'
    '    tip="Periksa log terminal di atas untuk detail error.";\n'
    '  box.textContent="[!] "+tip;\n'
    '  box.classList.add("on");\n'
    '}\n'
    '\n'
    'function showDL(files){\n'
    '  var list=document.getElementById("dll");\n'
    '  files.forEach(function(f){\n'
    '    var i=document.createElement("div");i.className="dli";\n'
    '    var a=document.createElement("a");a.className="dlb";\n'
    '    a.href="/download/"+encodeURIComponent(f);\n'
    '    a.download=f;a.textContent="DOWNLOAD";\n'
    '    var n=document.createElement("span");n.className="dln";n.textContent=f;\n'
    '    i.appendChild(n);i.appendChild(a);\n'
    '    list.appendChild(i);\n'
    '  });\n'
    '  document.getElementById("dlw").classList.add("on");\n'
    '}\n'
    '\n'
    'function setChip(t,txt){\n'
    '  var c=document.getElementById("chip");\n'
    '  c.className="chip on "+(t==="r"?"cr":t==="o"?"co":"ce");\n'
    '  document.getElementById("chipT").textContent=txt;\n'
    '}\n'
    'function setPb(s){\n'
    '  var p=document.getElementById("pb");\n'
    '  p.classList.remove("run");\n'
    '  if(s==="r")p.classList.add("run");\n'
    '  else if(s==="d")p.style.width="100%";\n'
    '}\n'
    'function rst(){rstBtn();document.getElementById("chip").className="chip";}\n'
    'function rstBtn(){\n'
    '  var b=document.getElementById("startBtn");\n'
    '  b.disabled=false;\n'
    '  b.textContent="[MULAI EKSEKUSI STABIL]";\n'
    '}\n'
    '\n'
    '// Inisialisasi\n'
    'document.getElementById("guide").style.display="none";\n'
    'ckSt();\n'
    '</script>\n'
    '</body>\n'
    '</html>\n'
)


# ── Core Engine (logic tidak diubah) ──────────────────────────────────────────

def find_main_face(clip):
    frames = [clip.get_frame(t) for t in np.linspace(0, min(5, clip.duration), 10)]
    x_positions = []
    for frame in frames:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        print("[%s] %s" % (job_id, msg))
        jobs[job_id]["log"].append(msg)

    jobs[job_id]["status"] = "running"
    output_files = []
    has_cookies  = os.path.exists(COOKIES_PATH)

    log("[OK] Cookies aktif" if has_cookies else "[WARN] Tidak ada cookies - mungkin diblokir YouTube")

    BASE_OPTS = {
        "quiet"              : True,
        "no_warnings"        : True,
        "ffmpeg_location"    : CURRENT_DIR,
        "retries"            : 3,
        "merge_output_format": "mp4",
        "http_headers"       : {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        },
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
        temp_video  = os.path.join(OUTPUT_DIR, "temp_%s_%d.mp4"  % (job_id, i))
        output_name = os.path.join(OUTPUT_DIR, "PODCAST_STABLE_%s_%d.mp4" % (job_id, i + 1))
        try:
            # Step 1: cek info video
            log("[%d/%d] Memeriksa: %s" % (i + 1, len(links), url))
            with yt_dlp.YoutubeDL(dict(BASE_OPTS, skip_download=True)) as ydl:
                info = ydl.extract_info(url, download=False)

            duration = info.get("duration", 0)
            if not duration:
                raise ValueError("Bukan video YouTube biasa (duration=0). "
                                 "Pastikan link ke VIDEO, bukan Post/gambar.")
            if duration < 42:
                raise ValueError("Video terlalu pendek (%ds). Minimal 42 detik." % duration)

            title = info.get("title", "?")[:50]
            log("[%d] Video OK: \"%s\" (%ds)" % (i + 1, title, duration))

            # Step 2: download dengan format fallback
            log("[%d] Downloading..." % (i + 1))
            downloaded = False
            last_err   = ""
            for fmt in FORMAT_PRIORITY:
                try:
                    opts = dict(BASE_OPTS, outtmpl=temp_video, format=fmt)
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([url])
                    if os.path.exists(temp_video) and os.path.getsize(temp_video) > 10000:
                        log("[%d] Download OK" % (i + 1))
                        downloaded = True
                        break
                except Exception as fe:
                    last_err = str(fe)
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                    err_low = last_err.lower()
                    if "requested format" in err_low or "only images" in err_low:
                        continue   # coba format berikutnya
                    raise          # error lain langsung raise

            if not downloaded:
                raise RuntimeError("Semua format gagal. Error: " + last_err)

            # Step 3: re-framing (logic asli tidak diubah)
            clip     = VideoFileClip(temp_video).subclipped(10, 40)
            w, h     = clip.size
            target_w = h * (9.0 / 16.0)

            anchor_x_percent = find_main_face(clip)
            log("[%d] Posisi wajah: %.1f%%" % (i + 1, anchor_x_percent * 100))
            center_x = anchor_x_percent * w

            x1 = int(max(0, min(w - target_w, center_x - (target_w / 2))))
            x2 = int(x1 + target_w)

            log("[%d] Crop 9:16..." % (i + 1))
            final_clip = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)

            # Step 4: audio
            final_audio = clip.audio
            music_path  = os.path.join(CURRENT_DIR, "bg_music.mp3")
            if os.path.exists(music_path):
                bg          = AudioFileClip(music_path).with_duration(clip.duration)
                bg          = afx.multiply_volume(bg, 0.1)
                final_audio = CompositeAudioClip([clip.audio, bg])

            # Step 5: render
            log("[%d] Rendering..." % (i + 1))
            final_clip.with_audio(final_audio).write_videofile(
                output_name,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                threads=4,
                logger=None,
            )

            clip.close()
            if os.path.exists(temp_video):
                os.remove(temp_video)

            fname = os.path.basename(output_name)
            log("[%d] Selesai: %s" % (i + 1, fname))
            output_files.append(fname)

        except Exception as e:
            log("[%d] [ERR] Gagal: %s" % (i + 1, str(e)))
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except Exception:
                    pass

    jobs[job_id]["files"]  = output_files
    jobs[job_id]["status"] = "done"
    log("Semua proses selesai! (%d berhasil)" % len(output_files))


# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Encode manual ke bytes UTF-8 untuk menghindari UnicodeEncodeError
    return Response(HTML.encode("utf-8"), mimetype="text/html; charset=utf-8")


@app.route("/cookie-status")
def cookie_status():
    exists = os.path.exists(COOKIES_PATH)
    size   = ""
    if exists:
        b    = os.path.getsize(COOKIES_PATH)
        size = ("%d KB" % (b // 1024)) if b > 1024 else ("%d B" % b)
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
    print("[cookies] Tersimpan: %s (%d chars)" % (COOKIES_PATH, len(content)))
    return jsonify({"ok": True})


@app.route("/delete-cookies", methods=["POST"])
def delete_cookies():
    if os.path.exists(COOKIES_PATH):
        os.remove(COOKIES_PATH)
    return jsonify({"ok": True})


@app.route("/start", methods=["POST"])
def start():
    data  = request.json or {}
    links = [l.strip() for l in data.get("links", "").split("\n")
             if l.strip() and "http" in l]
    if not links:
        return jsonify({"error": "Tidak ada link valid"}), 400
    job_id       = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "log": [], "files": []}
    threading.Thread(
        target=process_links, args=(job_id, links), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Tidak ditemukan"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename):
    safe = os.path.basename(filename)          # cegah path traversal
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(path):
        return "File tidak ditemukan", 404
    return send_file(path, as_attachment=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nViral Studio Pro - http://0.0.0.0:7575\n")
    app.run(host="0.0.0.0", port=7575, debug=False)
