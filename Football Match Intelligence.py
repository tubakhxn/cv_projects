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

C_TEAM_A        = (220, 220, 220)   # silver-white ellipse
C_TEAM_A_FILL   = (255, 255, 255)   # white body tint
C_TEAM_A_LABEL  = (255, 255, 255)
C_TEAM_A_BG     = (50,  50,  70)

C_TEAM_B        = (50,  130, 255)   # vivid blue ellipse
C_TEAM_B_FILL   = (30,  90,  220)   # royal blue tint
C_TEAM_B_LABEL  = (140, 195, 255)
C_TEAM_B_BG     = (10,  25,  70)

C_BALL   = (0,   220, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)

PIXEL_TO_METER = 0.065
LABEL_FONT     = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE    = 0.72
LABEL_THICK    = 2


class TeamClassifier:
    def __init__(self):
        self.calibrated   = False
        self.feature_buf  = []   # (white_score, blue_score) per detection
        self.id_team      = {}   # tid → "team_a" | "team_b"
        self.center_a     = None   # (white_c, blue_c) for team_a
        self.center_b     = None

    def _features(self, frame, x1, y1, x2, y2):
        ty = int(y1 + (y2-y1)*0.15);  by = int(y1 + (y2-y1)*0.55)
        tx = int(x1 + (x2-x1)*0.20);  bx = int(x1 + (x2-x1)*0.80)
        if by <= ty or bx <= tx: return None
        crop = frame[ty:by, tx:bx]
        if crop.size < 100: return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H   = hsv[:,:,0].astype(np.float32)
        S   = hsv[:,:,1].astype(np.float32)
        V   = hsv[:,:,2].astype(np.float32)

        white_score = float(np.mean((S < 80) & (V > 70)))
        blue_score  = float(np.mean((H > 90) & (H < 140) & (S > 80)))

        return white_score, blue_score
    def _calibrate(self):
        data = np.array(self.feature_buf, dtype=np.float32)  
        if len(data) < 20: return

        c_a = np.array([data[:,0].max(), data[:,1].min()], dtype=np.float32)
        c_b = np.array([data[:,0].min(), data[:,1].max()], dtype=np.float32)

        for _ in range(20):  
            da   = np.linalg.norm(data - c_a, axis=1)
            db   = np.linalg.norm(data - c_b, axis=1)
            mask = da < db
            if mask.sum() > 0:  c_a = data[mask].mean(axis=0)
            if (~mask).sum()>0: c_b = data[~mask].mean(axis=0)

        if c_a[0] >= c_b[0]:
            self.center_a, self.center_b = c_a, c_b
        else:
            self.center_a, self.center_b = c_b, c_a

        self.calibrated = True
        print(f"[TEAM] Calibrated  white_center={self.center_a}  blue_center={self.center_b}")

    def assign(self, frame, x1, y1, x2, y2):
        feat = self._features(frame, x1, y1, x2, y2)
        if feat is None: return "unknown"
        ws, bs = feat

        if not self.calibrated:
            self.feature_buf.append(feat)
            if len(self.feature_buf) >= 40:
                self._calibrate()
            return "team_a" if ws > bs else "team_b"

        f   = np.array([ws, bs])
        d_a = np.linalg.norm(f - self.center_a)
        d_b = np.linalg.norm(f - self.center_b)
        return "team_a" if d_a <= d_b else "team_b"

    def reassign_all(self, frame, tracks):
        """Call once after calibration to fix early mis-assignments."""
        for tid, info in tracks.items():
            x1,y1,x2,y2 = [int(v) for v in info["box"]]
            feat = self._features(frame, x1, y1, x2, y2)
            if feat is None: continue
            f   = np.array(feat)
            d_a = np.linalg.norm(f - self.center_a)
            d_b = np.linalg.norm(f - self.center_b)
            self.id_team[tid] = "team_a" if d_a <= d_b else "team_b"


class PlayerTracker:
    def __init__(self, fps=25):
        self.tracks  = {}; self.next_id = 0
        self.lost    = defaultdict(int); self.max_lost = 25
        self.fps     = fps
        self.teams   = {}
        self.dists   = defaultdict(float)
        self.speeds  = defaultdict(float)
        self._just_calibrated = False  

    def update(self, dets, frame, tc):
        new  = {}; used = set()
        just_cal = (not self._just_calibrated) and tc.calibrated
        if just_cal: self._just_calibrated = True

        for x1,y1,x2,y2,conf,cls in dets:
            cx=(x1+x2)/2; cy=(y1+y2)/2
            best_id=None; best_d=100
            for tid,info in self.tracks.items():
                if tid in used: continue
                d=np.hypot(cx-info["cx"],cy-info["cy"])
                if d<best_d: best_d=d; best_id=tid

            if best_id is None:
                best_id=self.next_id; self.next_id+=1
                self.teams[best_id]=tc.assign(frame,x1,y1,x2,y2)
            else:
                if tc.calibrated:
                    self.teams[best_id]=tc.assign(frame,x1,y1,x2,y2)

            if best_id in self.tracks:
                dpx=np.hypot(cx-self.tracks[best_id]["cx"],cy-self.tracks[best_id]["cy"])
                self.dists[best_id]+=dpx*PIXEL_TO_METER
                self.speeds[best_id]=dpx*PIXEL_TO_METER*self.fps*3.6

            new[best_id]={"box":(x1,y1,x2,y2),"cx":cx,"cy":cy,"conf":conf}
            used.add(best_id)

        for tid in list(self.tracks):
            if tid not in used:
                self.lost[tid]+=1
                if self.lost[tid]<=self.max_lost: new[tid]=self.tracks[tid]
            else: self.lost[tid]=0

        self.tracks=new
        return new


class Possession:
    def __init__(self):
        self.total=defaultdict(int)

    def update(self, tracks, pt, ball):
        if ball is None: return
        bx,by=ball; best_d=120; best_t="none"
        for tid,info in tracks.items():
            d=np.hypot(info["cx"]-bx,info["cy"]-by)
            if d<best_d: best_d=d; best_t=pt.teams.get(tid,"none")
        if best_t!="none": self.total[best_t]+=1

    def pct(self):
        tot=sum(self.total.values())
        if tot==0: return 50.0,50.0
        return (round(self.total.get("team_a",0)/tot*100,1),
                round(self.total.get("team_b",0)/tot*100,1))


def txt(img, text, x, y, scale=0.72, color=C_WHITE, thick=2,
        font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img,text,(x,y),font,scale,(0,0,0),thick+3,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),font,scale,color,thick,cv2.LINE_AA)


def centered_label_block(img, lines, colors, cx, y_start,
                          scale=LABEL_SCALE, thick=LABEL_THICK,
                          bg_color=(20,20,20), alpha=0.75):
    font    = LABEL_FONT
    pad_x,pad_y,gap = 14, 7, 5
    sizes   = [cv2.getTextSize(l,font,scale,thick)[0] for l in lines]
    max_w   = max(s[0] for s in sizes)
    total_h = sum(s[1] for s in sizes) + gap*(len(lines)-1)
    h_img,w_img = img.shape[:2]

    bx1 = max(0,  cx - max_w//2 - pad_x)
    by1 = max(0,  y_start - pad_y)
    bx2 = min(w_img-1, cx + max_w//2 + pad_x)
    by2 = min(h_img-1, y_start + total_h + pad_y)

    overlay = img.copy()
    cv2.rectangle(overlay,(bx1,by1),(bx2,by2),bg_color,-1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    y_cur = y_start
    for line, color, (tw, th) in zip(lines, colors, sizes):
        lx = cx - tw//2
        txt(img, line, lx, y_cur+th, scale, color, thick, font)
        y_cur += th + gap


def draw_player_overlay(canvas, x1, y1, x2, y2, team, alpha=0.22):
    h_img,w_img = canvas.shape[:2]
    ix1 = max(0, x1+4);  iy1 = max(0, y1+4)
    ix2 = min(w_img-1, x2-4);  iy2 = min(h_img-1, int(y1+(y2-y1)*0.82))
    if ix2<=ix1 or iy2<=iy1: return
    fill = C_TEAM_A_FILL if team=="team_a" else C_TEAM_B_FILL
    overlay = canvas.copy()
    cv2.rectangle(overlay,(ix1,iy1),(ix2,iy2),fill,-1)
    cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)


def draw_cam_hud(frame, dx, dy, cum_x, cum_y):
    font=cv2.FONT_HERSHEY_DUPLEX; scale=0.80; thick=2
    b1 = f"Camera Movement:     X={dx:+.2f}  |  Y={dy:+.2f}"
    b2 = f"Camera Displacement: X={cum_x:+.2f}  |  Y={cum_y:+.2f}"
    (w1,h1),_ = cv2.getTextSize(b1,font,scale,thick)
    (w2,h2),_ = cv2.getTextSize(b2,font,scale,thick)
    p=12
    cv2.rectangle(frame,(0,0),(w1+p*2,h1+p*2),(110,0,150),-1)
    cv2.putText(frame,b1,(p,h1+p),font,scale,C_WHITE,thick,cv2.LINE_AA)
    y2=h1+p*2+4
    cv2.rectangle(frame,(0,y2),(w2+p*2,y2+h2+p*2),(170,15,15),-1)
    cv2.putText(frame,b2,(p,y2+h2+p),font,scale,C_WHITE,thick,cv2.LINE_AA)


def draw_minimap(canvas, tracks, pt, W, H, mw=220, mh=140):
    mx=W-mw-14; my=H-mh-42
    ov=canvas.copy()
    cv2.rectangle(ov,(mx,my),(mx+mw,my+mh),(8,22,8),-1)
    cv2.rectangle(ov,(mx,my),(mx+mw,my+mh),(0,90,0),2)
    cv2.line(ov,(mx+mw//2,my),(mx+mw//2,my+mh),(0,65,0),1)
    cv2.ellipse(ov,(mx+mw//2,my+mh//2),(22,15),0,0,360,(0,65,0),1)
    cv2.addWeighted(ov,0.82,canvas,0.18,0,canvas)
    for tid,info in tracks.items():
        px=int(mx+(info["cx"]/W)*mw); py=int(my+(info["cy"]/H)*mh)
        px=np.clip(px,mx+2,mx+mw-2); py=np.clip(py,my+2,my+mh-2)
        team=pt.teams.get(tid,"unknown")
        col=(C_TEAM_A if team=="team_a" else C_TEAM_B if team=="team_b" else (120,120,120))
        cv2.circle(canvas,(px,py),5,col,-1,cv2.LINE_AA)
        cv2.circle(canvas,(px,py),5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(canvas,"MINIMAP",(mx+4,my-5),cv2.FONT_HERSHEY_DUPLEX,0.42,(100,190,100),1,cv2.LINE_AA)


def draw_bottom_hud(canvas, pa, pb, avg_s, act, n_players, W, H):
    bar_h=54; y0=H-bar_h-22
    ov=canvas.copy()
    cv2.rectangle(ov,(0,y0),(W,y0+bar_h),(8,12,20),-1)
    cv2.addWeighted(ov,0.80,canvas,0.20,0,canvas)
    cv2.line(canvas,(0,y0),(W,y0),(50,50,80),1)
    font=cv2.FONT_HERSHEY_DUPLEX; scale=0.72; thick=2
    # possession bar
    bw=300; bx=W//2-bw//2; by=y0+8; bh2=16
    af=int(bw*pa/100)
    cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh2),(40,40,40),-1)
    if af>0: cv2.rectangle(canvas,(bx,by),(bx+af,by+bh2),C_TEAM_A,-1)
    if af<bw:cv2.rectangle(canvas,(bx+af,by),(bx+bw,by+bh2),C_TEAM_B,-1)
    cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh2),(80,80,100),1)
    pt_=f"A {pa}%  |  POSSESSION  |  {pb}% B"
    (pw,ph),_=cv2.getTextSize(pt_,font,0.58,1)
    txt(canvas,pt_,W//2-pw//2,by+bh2+20,0.58,C_WHITE,1,font)
    stat1=f"AVG SPD  {avg_s:.1f} km/h"
    (sw,_),_=cv2.getTextSize(stat1,font,scale,thick)
    txt(canvas,stat1,bx-sw-50,y0+bar_h//2+10,scale,(0,220,130),thick,font)
    txt(canvas,f"PLAYERS {n_players}   ACT {act:.0f}",bx+bw+30,y0+bar_h//2+10,scale,(0,200,255),thick,font)


def save_heatmap(pos_a, pos_b, shape, path):
    H,W=shape[:2]
    fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor="#080c12")
    for ax,pos,title,cmap in zip(axes,[pos_a,pos_b],
        ["TEAM A (WHITE JERSEY) — HEATMAP","TEAM B (BLUE JERSEY) — HEATMAP"],
        ["binary","Blues"]):
        ax.set_facecolor("#0a1020")
        hm=np.zeros((H//4,W//4),dtype=np.float32)
        for px,py in pos:
            xi,yi=int(px/4),int(py/4)
            if 0<=xi<hm.shape[1] and 0<=yi<hm.shape[0]: hm[yi,xi]+=1
        hm=gaussian_filter(hm,sigma=8)
        if hm.max()>0: hm/=hm.max()
        ax.imshow(hm,cmap=cmap,interpolation="bilinear",aspect="auto",origin="upper")
        ax.set_title(title,color="#00c8ff",fontsize=13,fontweight="bold",pad=10)
        ax.axis("off")
    fig.suptitle("FOOTBALL MATCH ANALYTICS — HEATMAPS  |  dev: tubakhxn",
                 color="#00c8ff",fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"[DONE] {path} ✓")


def save_dashboard(hist_spd, hist_act, poss, pt, frame_idx, path):
    fig,axes=plt.subplots(2,2,figsize=(14,8),facecolor="#080c12")
    for ax in axes.flat:
        ax.set_facecolor("#0a1220")
        for sp in ax.spines.values(): sp.set_color("#1a2840")
    fig.suptitle("FOOTBALL MATCH ANALYTICS — REPORT  |  dev: tubakhxn",
                 color="#00c8ff",fontsize=14,fontweight="bold",y=0.98)
    fx=np.arange(len(hist_spd))
    axes[0,0].fill_between(fx,hist_spd,alpha=0.35,color="#00dc64")
    axes[0,0].plot(fx,hist_spd,color="#00dc64",lw=1.5)
    axes[0,0].set_title("Avg Player Speed (km/h)",color="#00c8ff",fontsize=11)
    axes[0,0].set_ylabel("km/h",color="#6a8aaa"); axes[0,0].tick_params(colors="#6a8aaa")
    axes[0,1].fill_between(fx,hist_act,alpha=0.35,color="#ffa000")
    axes[0,1].plot(fx,hist_act,color="#ffa000",lw=1.5)
    axes[0,1].set_title("Match Activity Score",color="#00c8ff",fontsize=11)
    axes[0,1].set_ylabel("Score",color="#6a8aaa"); axes[0,1].tick_params(colors="#6a8aaa")
    pa,pb=poss.pct()
    axes[1,0].pie([pa,pb],
        labels=[f"Team A (White)\n{pa}%",f"Team B (Blue)\n{pb}%"],
        colors=[(0.86,0.86,0.86),(0.20,0.51,1.0)],
        textprops={"color":"#e0eeff","fontsize":11},
        wedgeprops={"edgecolor":"#080c12","linewidth":2},startangle=90)
    axes[1,0].set_title("Possession %",color="#00c8ff",fontsize=11)
    axes[1,1].axis("off")
    peak_s=max(hist_spd) if hist_spd else 0
    avg_s=np.mean(hist_spd) if hist_spd else 0
    td=sum(pt.dists.values())
    tbl=axes[1,1].table(cellText=[
        ["Frames Processed",str(frame_idx)],
        ["Avg Speed",f"{avg_s:.1f} km/h"],
        ["Peak Speed",f"{peak_s:.1f} km/h"],
        ["Team A (White) Poss.",f"{pa}%"],
        ["Team B (Blue) Poss.",f"{pb}%"],
        ["Total Dist (est.)",f"{td:.0f} m"],
    ],colLabels=["Metric","Value"],loc="center",cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor("#101a2c" if r%2==0 else "#1a2840")
        cell.set_text_props(color="#deeeff"); cell.set_edgecolor("#2a3a5a")
    axes[1,1].set_title("Match Summary",color="#00c8ff",fontsize=11)
    plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"[DONE] {path} ✓")


def make_writer(path, fps, W, H):
    for fc in ["avc1","H264","h264","mp4v"]:
        w=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*fc),fps,(W,H))
        if w.isOpened(): print(f"[CODEC] {fc}"); return w
        w.release()
    raise RuntimeError("No codec found")


def process(video_path):
    print("╔══════════════════════════════════════════════╗")
    print("║   FOOTBALL MATCH ANALYTICS  v3.1            ║")
    print("║   dev: tubakhxn                             ║")
    print("╚══════════════════════════════════════════════╝")

    model = YOLO("yolov8n.pt")
    print("[YOLO] loaded ✓")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] Cannot open {video_path}"); return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames")

    writer  = make_writer("football_output.mp4", fps, W, H)
    tc      = TeamClassifier()
    pt      = PlayerTracker(fps=fps)
    poss    = Possession()
    pos_a=[]; pos_b=[]
    hist_spd=[]; hist_act=[]
    prev_gray=None; cum_x=0.0; cum_y=0.0
    speed_smooth = defaultdict(lambda: deque(maxlen=8))
    frame_idx=0
    was_calibrated = False

    print("[PROC] Processing...")
    with tqdm(total=total,unit="fr",ncols=80,colour="cyan") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            canvas = frame.copy()

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dx=dy=0.0
            if prev_gray is not None:
                pts = cv2.goodFeaturesToTrack(prev_gray,150,0.01,10)
                if pts is not None:
                    npts,st,_=cv2.calcOpticalFlowPyrLK(prev_gray,curr_gray,pts,None)
                    gn=npts[st==1]; go=pts[st==1]
                    if len(gn):
                        fl=gn-go
                        dx=float(np.median(fl[:,0])); dy=float(np.median(fl[:,1]))
            cum_x+=dx; cum_y+=dy
            prev_gray=curr_gray

            results = model(frame,verbose=False,conf=0.38)[0]
            dets=[]; ball_pos=None
            if results.boxes is not None:
                for box in results.boxes:
                    cls=int(box.cls[0]); name=model.names[cls]
                    x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                    conf=float(box.conf[0])
                    if name=="person":
                        dets.append((x1,y1,x2,y2,conf,"player"))
                    elif name=="sports ball":
                        ball_pos=((x1+x2)//2,(y1+y2)//2)

            tracks = pt.update(dets, frame, tc)
            poss.update(tracks, pt, ball_pos)

            if tc.calibrated and not was_calibrated:
                was_calibrated = True
                print("[TEAM] Re-classifying all tracks post-calibration...")
                for tid in tracks:
                    x1,y1,x2,y2=[int(v) for v in tracks[tid]["box"]]
                    pt.teams[tid] = tc.assign(frame,x1,y1,x2,y2)

            if ball_pos:
                bx,by=ball_pos
                for r,a in [(22,35),(15,70),(9,160),(5,255)]:
                    ov2=canvas.copy()
                    cv2.circle(ov2,(bx,by),r,C_BALL,-1,cv2.LINE_AA)
                    cv2.addWeighted(ov2,a/255.0,canvas,1-a/255.0,0,canvas)
                cv2.circle(canvas,(bx,by),9,C_WHITE,2,cv2.LINE_AA)

            for tid, info in tracks.items():
                x1,y1,x2,y2=[int(v) for v in info["box"]]
                team = pt.teams.get(tid,"unknown")
                spd_raw = pt.speeds.get(tid,0.0)
                speed_smooth[tid].append(spd_raw)
                spd  = float(np.mean(speed_smooth[tid]))
                dist = pt.dists.get(tid,0.0)
                cx_  = (x1+x2)//2;  cy_ = (y1+y2)//2
                rx   = max((x2-x1)//2+14, 24)
                ry   = max(int(rx*0.33), 9)

                if   team=="team_a": ecol,lcol,bgcol = C_TEAM_A, C_TEAM_A_LABEL, C_TEAM_A_BG
                elif team=="team_b": ecol,lcol,bgcol = C_TEAM_B, C_TEAM_B_LABEL, C_TEAM_B_BG
                else:                ecol,lcol,bgcol = (120,120,120), C_WHITE, (30,30,30)

                draw_player_overlay(canvas, x1, y1, x2, y2, team, alpha=0.28)

                glow = tuple(int(c*0.30) for c in ecol)
                cv2.ellipse(canvas,(cx_,y2),(rx+5,ry+4),0,0,360,glow,4,cv2.LINE_AA)
                cv2.ellipse(canvas,(cx_,y2),(rx,ry),0,0,360,ecol,2,cv2.LINE_AA)

                centered_label_block(
                    canvas,
                    lines   = [f"{spd:.1f} km/h", f"{dist:.1f} m"],
                    colors  = [lcol, tuple(int(c*0.72) for c in lcol)],
                    cx      = cx_,
                    y_start = y2 + ry + 8,
                    scale   = LABEL_SCALE,
                    thick   = LABEL_THICK,
                    bg_color= bgcol,
                    alpha   = 0.74
                )

            draw_cam_hud(canvas, dx, dy, cum_x, cum_y)
            draw_minimap(canvas, tracks, pt, W, H)

            pa,pb = poss.pct()
            spds  = [pt.speeds[t] for t in tracks]
            avg_s = float(np.mean(spds)) if spds else 0.0
            act   = min(len(tracks)*avg_s/10,100) if avg_s>0 else 0.0
            hist_spd.append(avg_s); hist_act.append(act)

            draw_bottom_hud(canvas, pa, pb, avg_s, act, len(tracks), W, H)

            cv2.rectangle(canvas,(0,H-22),(W,H),(0,0,0),-1)
            cv2.putText(canvas,"FOOTBALL MATCH ANALYTICS  v3.1  |  dev: tubakhxn",
                        (8,H-6),cv2.FONT_HERSHEY_DUPLEX,0.48,(150,150,150),1,cv2.LINE_AA)

            writer.write(canvas)

            for tid,info in tracks.items():
                rx_=(info["box"][0]+info["box"][2])//2
                ry_=(info["box"][1]+info["box"][3])//2
                t=pt.teams.get(tid,"")
                if t=="team_a": pos_a.append((rx_,ry_))
                elif t=="team_b": pos_b.append((rx_,ry_))

            frame_idx+=1; pbar.update(1)

    cap.release(); writer.release()
    print("[DONE] football_output.mp4 ✓")
    save_heatmap(pos_a, pos_b, (H,W), "football_heatmap.png")
    save_dashboard(hist_spd, hist_act, poss, pt, frame_idx, "football_dashboard.png")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python football_match_analytics.py video.mp4"); sys.exit(1)
    process(sys.argv[1])