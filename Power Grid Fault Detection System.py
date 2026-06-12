import subprocess, sys, os

def install_deps():
    pkgs = ["opencv-python","numpy","matplotlib","scipy","ultralytics","Pillow","tqdm"]
    for pkg in pkgs:
        try: __import__(pkg.replace("-","_").split("==")[0])
        except ImportError:
            print(f"[INSTALL] {pkg}...")
            subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

install_deps()

import cv2, numpy as np, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore")

C_BG      = (8,   12,  20)
C_ACCENT  = (0,  190, 255)   # cyan
C_GREEN   = (0,  220, 100)
C_ORANGE  = (255, 155,   0)
C_RED     = (255,  40,  40)
C_YELLOW  = (255, 220,   0)
C_WHITE   = (235, 242, 255)
C_GREY    = (90,  110, 135)
C_MAGENTA = (200,  40, 200)

COMP_COLORS = {
    "tower_body":       (0,   200, 255),   # cyan  (ref: tower_body box)
    "crossarm":         (0,   200, 255),   # cyan  (ref: crossarm)
    "crossarm_turret":  (0,   200, 255),   # cyan  (ref: crossarm_turrets)
    "insulator_string": (255, 255, 255),   # white (ref: insulator strings)
    "lattice_steel":    (0,   200, 255),   # cyan  (ref: lattice_steel_framework)
    "wire":             (180, 255,  80),   # lime green
    "generic":          (150, 150, 180),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def txt(img, text, x, y, scale=0.55, color=C_WHITE, thick=1, font=FONT):
    cv2.putText(img, text, (x,y), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), font, scale, color,   thick,   cv2.LINE_AA)


def draw_ref_box(canvas, x1, y1, x2, y2, label, conf, color):
    """Draws exactly like the reference: thin colored rect + text label above."""
    cv2.rectangle(canvas, (x1,y1),(x2,y2), color, 1, cv2.LINE_AA)

    lbl_str = f"{label.replace('_',' ')}  {conf:.2f}"
    scale   = 0.52
    thick   = 1
    (tw,th),_ = cv2.getTextSize(lbl_str, FONT, scale, thick)
    ly = max(y1-4, th+2)
    txt(canvas, lbl_str, x1, ly, scale, color, thick, FONT)


def draw_wires(canvas, wires):
    for x1,y1,x2,y2 in wires:
        cv2.line(canvas, (x1,y1),(x2,y2), COMP_COLORS["wire"], 1, cv2.LINE_AA)


def draw_top_hud(canvas, health, risk, counts, frame_idx, fps, status):
    h, w = canvas.shape[:2]
    bar_h = 48
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0,0),(w,bar_h), (8,12,20), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.line(canvas, (0,bar_h),(w,bar_h), C_ACCENT, 1)

    txt(canvas, "POWERLINE INTELLIGENCE  v2.0", 10, 30, 0.65, C_ACCENT, 2, FONT_BOLD)

    bw = 200; bx = w//2 - bw//2; by = 14; bh = 16
    cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh),(30,30,40),-1)
    fill = int(bw*health/100)
    hcol = C_GREEN if health>=70 else (C_ORANGE if health>=45 else C_RED)
    if fill>0: cv2.rectangle(canvas,(bx,by),(bx+fill,by+bh),hcol,-1)
    cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh),C_GREY,1)
    ht_str = f"HEALTH  {health:.0f}%"
    (tw,_),_ = cv2.getTextSize(ht_str,FONT_BOLD,0.50,1)
    txt(canvas, ht_str, w//2-tw//2, by+bh+14, 0.50, hcol, 1, FONT_BOLD)

    tc  = int(frame_idx/fps) if fps>0 else 0
    ts  = f"{tc//60:02d}:{tc%60:02d}"
    stat_col = C_GREEN if status=="NOMINAL" else (C_ORANGE if status=="WARNING" else C_RED)
    txt(canvas, status, w-230, 30, 0.65, stat_col, 2, FONT_BOLD)
    txt(canvas, ts,     w-100, 30, 0.65, C_WHITE,  2, FONT_BOLD)


def draw_bottom_strip(canvas, counts, risk):
    h, w = canvas.shape[:2]
    strip_h = 36
    overlay = canvas.copy()
    cv2.rectangle(overlay,(0,h-strip_h),(w,h),(8,12,20),-1)
    cv2.addWeighted(overlay,0.75,canvas,0.25,0,canvas)
    cv2.line(canvas,(0,h-strip_h),(w,h-strip_h),C_ACCENT,1)

    items = [
        (f"TOWER: {counts.get('tower_body',0)}",         C_ACCENT),
        (f"CROSSARM: {counts.get('crossarm',0)+counts.get('crossarm_turret',0)}", C_YELLOW),
        (f"INSULATOR: {counts.get('insulator_string',0)}",C_WHITE),
        (f"WIRES: {counts.get('wire',0)}",                COMP_COLORS['wire']),
        (f"RISK: {risk:.0f}%",                            C_GREEN if risk<35 else (C_ORANGE if risk<65 else C_RED)),
        ("dev: tubakhxn",                                  C_GREY),
    ]
    x = 12
    for label, col in items:
        txt(canvas, label, x, h-10, 0.52, col, 1, FONT_BOLD)
        (tw,_),_ = cv2.getTextSize(label, FONT_BOLD, 0.52, 1)
        x += tw + 36
        if x < w-200:
            cv2.line(canvas,(x-18,h-strip_h+8),(x-18,h-8),C_GREY,1)


def draw_risk_border(canvas, risk, frame_idx):
    if risk > 60 and frame_idx % 24 < 12:
        h,w = canvas.shape[:2]
        for t,a in [(6,40),(4,80),(2,160)]:
            ov = canvas.copy()
            cv2.rectangle(ov,(0,0),(w,h),C_RED,t)
            cv2.addWeighted(ov,a/255,canvas,1-a/255,0,canvas)
        txt(canvas,"⚠  CRITICAL RISK DETECTED  ⚠",
            w//2-200, 80, 0.80, C_RED, 2, FONT_BOLD)


class PowerlineDetector:
    def __init__(self, model):
        self.model = model

    def classify_box(self, x1, y1, x2, y2, fh, fw):
        w=x2-x1; h=y2-y1; aspect=w/(h+1e-3); area=w*h
        rel_y=(y1+y2)/2/fh
        if aspect<0.4 and h>fh*0.25:                         return "tower_body"
        if aspect>2.5 and rel_y<0.65:                        return "crossarm"
        if 1.4<aspect<=2.5 and rel_y<0.60:                   return "crossarm_turret"
        if 0.3<aspect<1.8 and area<(fw*fh*0.018):            return "insulator_string"
        if area>(fw*fh*0.08):                                 return "lattice_steel"
        return "generic"

    def detect_wires(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray,(3,3),0)
        edges = cv2.Canny(blur,25,80)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,
                                minLineLength=70,maxLineGap=40)
        segs=[]
        if lines is not None:
            for ln in lines:
                x1,y1,x2,y2=ln[0]
                if x2==x1: continue
                if abs((y2-y1)/(x2-x1+1e-3))<0.20:
                    segs.append((x1,y1,x2,y2))
        return segs

    def run(self, frame):
        fh,fw = frame.shape[:2]
        results = self.model(frame,verbose=False,conf=0.22)[0]
        comps=[]
        if results.boxes is not None:
            for box in results.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                conf=float(box.conf[0])
                label=self.classify_box(x1,y1,x2,y2,fh,fw)
                comps.append({"box":(x1,y1,x2,y2),"conf":conf,"label":label})
        wires=self.detect_wires(frame)
        return comps,wires


class CompTracker:
    def __init__(self):
        self.tracks={}; self.next_id=0
        self.lost=defaultdict(int); self.max_lost=22

    def update(self, comps):
        new={}; used=set()
        for comp in comps:
            x1,y1,x2,y2=comp["box"]
            cx=(x1+x2)/2; cy=(y1+y2)/2
            best_id=None; best_d=90
            for tid,info in self.tracks.items():
                if tid in used: continue
                d=np.hypot(cx-info["cx"],cy-info["cy"])
                if d<best_d: best_d=d; best_id=tid
            if best_id is None:
                best_id=self.next_id; self.next_id+=1
            new[best_id]={**comp,"cx":cx,"cy":cy}
            used.add(best_id)
        for tid in list(self.tracks):
            if tid not in used:
                self.lost[tid]+=1
                if self.lost[tid]<=self.max_lost: new[tid]=self.tracks[tid]
            else: self.lost[tid]=0
        self.tracks=new; return new


def compute_health(counts, wires):
    base=100
    if counts.get("insulator_string",0)<2: base-=20
    if counts.get("crossarm",0)+counts.get("crossarm_turret",0)<1: base-=15
    if wires<2: base-=10
    return float(np.clip(base,0,100))


def make_writer(path,fps,W,H):
    for fc in ["avc1","H264","h264","mp4v"]:
        w=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*fc),fps,(W,H))
        if w.isOpened(): print(f"[CODEC] {fc}"); return w
        w.release()
    raise RuntimeError("No codec")


# ── MAIN ───────────────────────────────────────────────────────
def process(video_path):
    print("╔══════════════════════════════════════════════╗")
    print("║   POWERLINE INTELLIGENCE SYSTEM  v2.0       ║")
    print("║   dev: tubakhxn                             ║")
    print("╚══════════════════════════════════════════════╝")

    model    = YOLO("yolov8n.pt")
    detector = PowerlineDetector(model)
    tracker  = CompTracker()
    print("[YOLO] loaded ✓")

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] {video_path}"); return
    fps  =cap.get(cv2.CAP_PROP_FPS) or 30
    W    =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames")

    writer = make_writer("powerline_output.mp4",fps,W,H)
    frame_idx=0
    hist_health=[]; hist_counts=[]
    total_counts_all=defaultdict(int)

    with tqdm(total=total,unit="fr",ncols=80,colour="cyan") as pbar:
        while True:
            ret,frame=cap.read()
            if not ret: break
            canvas=frame.copy()

            comps,wires=detector.run(frame)
            tracks=tracker.update(comps)

            frame_counts=defaultdict(int)
            for tid,info in tracks.items():
                frame_counts[info["label"]]+=1
            frame_counts["wire"]=len(wires)

            for k,v in frame_counts.items():
                total_counts_all[k]=max(total_counts_all[k],v)

            health=compute_health(dict(frame_counts),len(wires))
            risk  =max(0,100-health)
            status="NOMINAL" if health>=70 else ("WARNING" if health>=45 else "CRITICAL")
            hist_health.append(health); hist_counts.append(len(tracks))

            # ── draw wires first (background layer) ────────────
            draw_wires(canvas, wires)

            # ── draw component boxes (ref style) ───────────────
            for tid, info in tracks.items():
                x1,y1,x2,y2=info["box"]
                col=COMP_COLORS.get(info["label"],COMP_COLORS["generic"])
                draw_ref_box(canvas,x1,y1,x2,y2,info["label"],info["conf"],col)

            # ── HUD overlays ───────────────────────────────────
            draw_top_hud(canvas,health,risk,dict(frame_counts),frame_idx,fps,status)
            draw_bottom_strip(canvas,dict(frame_counts),risk)
            draw_risk_border(canvas,risk,frame_idx)

            writer.write(canvas); frame_idx+=1; pbar.update(1)

    cap.release(); writer.release()
    print("[DONE] powerline_output.mp4 ✓")

    # dashboard PNG
    fig,axes=plt.subplots(2,2,figsize=(14,8),facecolor="#080c14")
    for ax in axes.flat:
        ax.set_facecolor("#0e1420")
        for sp in ax.spines.values(): sp.set_color("#1a2840")
    fig.suptitle("POWERLINE INTELLIGENCE SYSTEM — INSPECTION REPORT  |  dev: tubakhxn",
                 color="#00beff",fontsize=14,fontweight="bold",y=0.98)
    fx=np.arange(len(hist_health))
    axes[0,0].fill_between(fx,hist_health,alpha=0.35,color="#00dc64")
    axes[0,0].plot(fx,hist_health,color="#00dc64",lw=1.5)
    axes[0,0].axhline(70,color="#ffa000",ls="--",lw=1,alpha=0.6,label="Warning")
    axes[0,0].axhline(45,color="#ff2828",ls="--",lw=1,alpha=0.6,label="Critical")
    axes[0,0].set_title("Infrastructure Health Score",color="#00beff",fontsize=11)
    axes[0,0].set_ylabel("Health %",color="#6a8aaa"); axes[0,0].tick_params(colors="#6a8aaa")
    axes[0,0].set_ylim(0,105); axes[0,0].legend(fontsize=8,facecolor="#0e1420",labelcolor="#aabbcc")
    axes[0,1].fill_between(fx,hist_counts,alpha=0.35,color="#00beff")
    axes[0,1].plot(fx,hist_counts,color="#00beff",lw=1.5)
    axes[0,1].set_title("Components Tracked",color="#00beff",fontsize=11)
    axes[0,1].set_ylabel("Count",color="#6a8aaa"); axes[0,1].tick_params(colors="#6a8aaa")
    comp_names=list(total_counts_all.keys())
    comp_vals=[total_counts_all[k] for k in comp_names]
    bcols=[tuple(c/255 for c in COMP_COLORS.get(k,COMP_COLORS["generic"])) for k in comp_names]
    axes[1,0].bar(range(len(comp_names)),comp_vals,color=bcols,edgecolor="#08121e")
    axes[1,0].set_xticks(range(len(comp_names)))
    axes[1,0].set_xticklabels([n.replace("_"," ") for n in comp_names],
                               rotation=30,ha="right",fontsize=8,color="#aabbcc")
    axes[1,0].set_title("Component Count (Peak)",color="#00beff",fontsize=11)
    axes[1,0].tick_params(colors="#6a8aaa")
    avg_h=np.mean(hist_health) if hist_health else 0
    min_h=np.min(hist_health)  if hist_health else 0
    axes[1,1].axis("off")
    tbl=axes[1,1].table(cellText=[
        ["Frames Processed",str(frame_idx)],
        ["Avg Health Score",f"{avg_h:.1f}%"],
        ["Min Health Score",f"{min_h:.1f}%"],
        ["Peak Components",str(max(hist_counts) if hist_counts else 0)],
        ["Final Status","NOMINAL" if avg_h>=70 else "WARNING"],
    ],colLabels=["Metric","Value"],loc="center",cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor("#141e30" if r%2==0 else "#1a2840")
        cell.set_text_props(color="#e0eeff"); cell.set_edgecolor("#2a3a5a")
    axes[1,1].set_title("Inspection Summary",color="#00beff",fontsize=11)
    plt.tight_layout()
    plt.savefig("powerline_dashboard.png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print("[DONE] powerline_dashboard.png ✓")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python powerline_intelligence_system.py video.mp4"); sys.exit(1)
    process(sys.argv[1])

