import subprocess, sys, os

def _install():
    for pkg in ["opencv-python","numpy","ultralytics","tqdm"]:
        try:
            __import__(pkg.replace("-","_"))
        except ImportError:
            print(f"[INSTALL] {pkg} …")
            subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])
_install()

import cv2, numpy as np, warnings
from collections import defaultdict, deque
from ultralytics import YOLO
from tqdm import tqdm
warnings.filterwarnings("ignore")

C_BG       = ( 10,  12,  18)
C_CYAN     = (  0, 230, 255)
C_GREEN    = (  0, 255, 120)
C_YELLOW   = (255, 220,   0)
C_ORANGE   = (255, 150,   0)
C_RED      = (255,  40,  40)
C_MAGENTA  = (255,   0, 210)
C_WHITE    = (240, 245, 255)
C_GREY     = ( 90, 110, 130)
C_DARK     = (  8,  12,  20)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_B     = cv2.FONT_HERSHEY_DUPLEX

TRACK_CLS  = {
    0:"per", 1:"bic", 2:"car", 3:"moto",
    5:"bus", 7:"trk", 14:"bir", 15:"cat",
    16:"dog", 24:"bkpk", 26:"hbag", 28:"suit",
    39:"btl", 56:"chr", 57:"pot", 60:"tbl",
    67:"ph",  73:"lap", 77:"cellph",
}

def _txt(img, text, x, y, scale=0.48, color=C_WHITE, thick=1, font=FONT):
    cv2.putText(img,text,(x,y),font,scale,(0,0,0),thick+2,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),font,scale,color,   thick,  cv2.LINE_AA)

def _box_color(conf):
    if conf >= 0.75: return C_GREEN
    if conf >= 0.50: return C_YELLOW
    return C_ORANGE

def _corner_rect(img, x1,y1,x2,y2, color, thick=2, tlen=14):
    """Draw corner-tick rectangle (YOLOvX / Waymo style)."""
    cv2.rectangle(img,(x1,y1),(x2,y2),color,1,cv2.LINE_AA)
    for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*tlen,cy),color,thick,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+dy*tlen),color,thick,cv2.LINE_AA)

def _iou(b1,b2):
    ix1,iy1=max(b1[0],b2[0]),max(b1[1],b2[1])
    ix2,iy2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

class Tracker:
    def __init__(self, max_lost=20, hist_len=60):
        self.tracks   = {}          # tid -> deque of (cx,cy,frame)
        self.boxes    = {}          # tid -> (x1,y1,x2,y2)
        self.confs    = {}          # tid -> conf
        self.cls_ids  = {}          # tid -> cls int
        self.vels     = {}          # tid -> (vx,vy)
        self.lost     = defaultdict(int)
        self.next_id  = 0
        self.max_lost = max_lost
        self.hist_len = hist_len

    def update(self, dets, frame_idx):
        """
        dets: list of (x1,y1,x2,y2,conf,cls)
        Returns dict  tid -> (x1,y1,x2,y2,conf,cls)
        """
        # seed on first frame
        if not self.tracks:
            for det in dets:
                self._new(det, frame_idx)
            return dict(zip(self.tracks.keys(),
                            [self.boxes[t] + (self.confs[t],self.cls_ids[t])
                             for t in self.tracks]))

        matched = {}; used_det = set(); used_tid = set()

        for tid in list(self.tracks.keys()):
            vx,vy  = self.vels[tid]
            pb     = self.boxes[tid]
            pcx    = (pb[0]+pb[2])/2 + vx
            pcy    = (pb[1]+pb[3])/2 + vy
            pw,ph  = pb[2]-pb[0], pb[3]-pb[1]
            pred   = (pcx-pw/2, pcy-ph/2, pcx+pw/2, pcy+ph/2)

            best_iou = 0.20; best_di = -1
            for di,det in enumerate(dets):
                if di in used_det: continue
                if det[5] != self.cls_ids[tid]: continue
                if _iou(pred, det[:4]) > best_iou:
                    best_iou = _iou(pred, det[:4]); best_di = di

            if best_di >= 0:
                det = dets[best_di]
                self.boxes[tid]   = det[:4]
                self.confs[tid]   = det[4]
                cx = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
                prev = self.tracks[tid][-1]
                self.vels[tid]  = (cx-prev[0], cy-prev[1])
                self.tracks[tid].append((cx,cy,frame_idx))
                self.lost[tid]  = 0
                matched[tid]    = det[:4]+(det[4],det[5])
                used_det.add(best_di); used_tid.add(tid)

        for di,det in enumerate(dets):
            if di not in used_det:
                tid = self._new(det, frame_idx)
                matched[tid] = det[:4]+(det[4],det[5])

        for tid in list(self.tracks.keys()):
            if tid not in used_tid:
                self.lost[tid] += 1
                if self.lost[tid] > self.max_lost:
                    for d in (self.tracks,self.boxes,self.confs,
                              self.cls_ids,self.vels,self.lost):
                        d.pop(tid,None)
        return matched

    def _new(self, det, frame_idx):
        tid = self.next_id; self.next_id += 1
        cx  = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
        self.tracks[tid]  = deque([(cx,cy,frame_idx)], maxlen=self.hist_len)
        self.boxes[tid]   = det[:4]
        self.confs[tid]   = det[4]
        self.cls_ids[tid] = det[5]
        self.vels[tid]    = (0.0, 0.0)
        self.lost[tid]    = 0
        return tid


def draw_zoom_inset(canvas, frame, box, inset_frac=0.28, pad_ratio=1.6):
    H, W = canvas.shape[:2]
    x1,y1,x2,y2 = [int(v) for v in box]
    bw, bh = x2-x1, y2-y1
    pw = int(bw*pad_ratio); ph = int(bh*pad_ratio)
    rx1 = max(0, x1 - (pw-bw)//2); ry1 = max(0, y1 - (ph-bh)//2)
    rx2 = min(W, rx1+pw);           ry2 = min(H, ry1+ph)
    if rx2-rx1 < 4 or ry2-ry1 < 4: return

    crop   = frame[ry1:ry2, rx1:rx2]
    iw     = int(W * inset_frac)
    ih     = int(iw * crop.shape[0] / max(crop.shape[1],1))
    ih     = min(ih, int(H*0.32))

    resized = cv2.resize(crop,(iw,ih),interpolation=cv2.INTER_LINEAR)

    mx, my = 12, 54
    ox     = W - iw - mx; oy = my

    border_col = C_CYAN
    cv2.rectangle(canvas,(ox-3,oy-3),(ox+iw+3,oy+ih+3),border_col,2,cv2.LINE_AA)

    _corner_rect(canvas, ox-3,oy-3, ox+iw+3,oy+ih+3, C_CYAN, thick=2, tlen=10)

    canvas[oy:oy+ih, ox:ox+iw] = resized

    sl_ov = canvas[oy:oy+ih, ox:ox+iw].copy()
    for row in range(0, ih, 3):
        cv2.line(sl_ov,(0,row),(iw,row),(0,0,0),1)
    cv2.addWeighted(sl_ov,0.35,canvas[oy:oy+ih, ox:ox+iw],0.65,0,
                    canvas[oy:oy+ih, ox:ox+iw])

    cx_i = ox + iw//2; cy_i = oy + ih//2
    arm  = 14
    cv2.line(canvas,(cx_i-arm,cy_i),(cx_i+arm,cy_i),C_CYAN,1,cv2.LINE_AA)
    cv2.line(canvas,(cx_i,cy_i-arm),(cx_i,cy_i+arm),C_CYAN,1,cv2.LINE_AA)
    cv2.circle(canvas,(cx_i,cy_i),5,C_CYAN,1,cv2.LINE_AA)

    # label
    lbl = "ZOOM TARGET"
    _txt(canvas,lbl,ox,oy+ih+14,0.44,C_CYAN,1,FONT_B)

    cv2.rectangle(canvas,(rx1,ry1),(rx2,ry2),C_CYAN,1,cv2.LINE_AA)
    cv2.line(canvas,(rx2,ry1),(ox+iw+3,oy-3),C_CYAN,1,cv2.LINE_AA)


def draw_trail(canvas, hist):
    pts = list(hist)
    n   = len(pts)
    for i in range(1,n):
        p1 = (int(pts[i-1][0]),int(pts[i-1][1]))
        p2 = (int(pts[i][0]),  int(pts[i][1]))
        a  = int(50 + 200*(i/n))
        col= tuple(int(c*a/255) for c in C_MAGENTA)
        cv2.line(canvas,p1,p2,col,2,cv2.LINE_AA)
    if pts:
        cv2.circle(canvas,(int(pts[-1][0]),int(pts[-1][1])),5,C_MAGENTA,-1,cv2.LINE_AA)


def draw_hud(canvas, n_obj, locked_id, frame_idx, fps, total):
    H,W = canvas.shape[:2]
    # top bar
    ov = canvas.copy()
    cv2.rectangle(ov,(0,0),(W,46),(6,8,16),-1)
    cv2.addWeighted(ov,0.78,canvas,0.22,0,canvas)
    cv2.line(canvas,(0,46),(W,46),C_CYAN,1)

    _txt(canvas,"YOLOvX  AERIAL SURVEILLANCE",10,30,0.65,C_CYAN,2,FONT_B)

    tc = int(frame_idx/fps) if fps>0 else 0
    _txt(canvas,f"{tc//60:02d}:{tc%60:02d}",W-80,30,0.65,C_WHITE,2,FONT_B)

    status = f"OBJ:{n_obj}"
    if locked_id is not None:
        status += f"  TRACKING ID:{locked_id}"
        _txt(canvas,status,W//2-120,30,0.58,C_MAGENTA,1,FONT_B)
    else:
        status += "  CLICK TO LOCK TARGET"
        _txt(canvas,status,W//2-130,30,0.55,C_GREEN,1,FONT_B)

    # bottom bar
    ov2 = canvas.copy()
    cv2.rectangle(ov2,(0,H-32),(W,H),(6,8,16),-1)
    cv2.addWeighted(ov2,0.78,canvas,0.22,0,canvas)
    cv2.line(canvas,(0,H-32),(W,H-32),C_CYAN,1)

    items=[
        ("CLICK·ZOOM·TRACK",   C_CYAN),
        (f"OBJECTS: {n_obj}",  C_GREEN),
        ("Q=QUIT",             C_GREY),
        ("R=RESET",            C_GREY),
        ("SPACE=PAUSE",        C_GREY),
        ("dev: tubakhxn",      C_GREY),
    ]
    x=12
    for lbl,col in items:
        _txt(canvas,lbl,x,H-9,0.44,col,1,FONT_B)
        (tw,_),_=cv2.getTextSize(lbl,FONT_B,0.44,1)
        x+=tw+28
        if x<W-120:
            cv2.line(canvas,(x-14,H-30),(x-14,H-6),C_GREY,1)

    prog = int(W*frame_idx/max(total,1))
    cv2.rectangle(canvas,(0,H-3),(prog,H),C_CYAN,-1)


def draw_logo(canvas):
    H,W = canvas.shape[:2]
    for i,ch in enumerate("YOLOvX"):
        cols=[C_YELLOW,C_YELLOW,C_YELLOW,C_YELLOW,C_CYAN,C_MAGENTA]
        _txt(canvas,ch, W-130+i*20, H-42, 0.90, cols[i], 2, FONT_B)



def process(video_path, output_path="yolovx_output.mp4"):
    print("╔══════════════════════════════════════════╗")
    print("║  YOLOvX AERIAL SURVEILLANCE TRACKER     ║")
    print("║  dev: tubakhxn                          ║")
    print("╚══════════════════════════════════════════╝")

    model   = YOLO("yolov8n.pt"); print("[YOLO] loaded ✓")
    cap     = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] cannot open {video_path}"); return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}x{H} @ {fps:.1f}fps | {total} frames")

    writer  = None
    for fc in ["avc1","H264","h264","mp4v"]:
        w = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*fc),fps,(W,H))
        if w.isOpened(): writer=w; print(f"[CODEC] {fc}"); break
        w.release()
    if writer is None: raise RuntimeError("No working codec found")

    tracker    = Tracker()
    locked_id  = None      
    frame_idx  = 0
    paused     = False

    click_pt   = [None]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt[0] = (x, y)

    cv2.namedWindow("YOLOvX Aerial Tracker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLOvX Aerial Tracker", on_mouse)

    with tqdm(total=total,unit="fr",ncols=80,colour="green") as pbar:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: break

            canvas = frame.copy()

            results = model(frame,verbose=False,conf=0.28)[0]
            dets    = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    dets.append((x1,y1,x2,y2,float(box.conf[0]),cls))

            matched = tracker.update(dets, frame_idx)

            if click_pt[0] is not None:
                mx,my = click_pt[0]; click_pt[0]=None
                best_dist = 1e9; best_tid = None
                for tid,(x1,y1,x2,y2,cf,cl) in matched.items():
                    cx=(x1+x2)/2; cy=(y1+y2)/2
                    d = np.hypot(mx-cx,my-cy)
                    if d < best_dist and d < max(x2-x1,y2-y1)*1.5:
                        best_dist=d; best_tid=tid
                if best_tid is not None:
                    locked_id = best_tid
                    print(f"[LOCK] ID:{locked_id}")

            for tid,(x1,y1,x2,y2,cf,cl) in matched.items():
                cls_name = TRACK_CLS.get(cl, f"obj")
                is_locked = (tid == locked_id)
                col = C_MAGENTA if is_locked else _box_color(cf)

                _corner_rect(canvas,x1,y1,x2,y2,col,
                             thick=3 if is_locked else 2,
                             tlen=16 if is_locked else 10)

                lbl = f"#{tid} {cf:.2f} {cls_name}"
                (tw,th),_ = cv2.getTextSize(lbl,FONT,0.40,1)
                cv2.rectangle(canvas,(x1,y1-th-6),(x1+tw+4,y1),
                              (10,12,18),-1)
                _txt(canvas,lbl,x1+2,y1-3,0.40,col,1)

                vx,vy = tracker.vels.get(tid,(0,0))
                spd   = np.hypot(vx,vy)
                if spd > 0.5:
                    cx_=(x1+x2)//2; cy_=(y1+y2)//2
                    sc=min(spd*7,40)
                    ex=int(cx_+vx/spd*sc); ey=int(cy_+vy/spd*sc)
                    cv2.arrowedLine(canvas,(cx_,cy_),(ex,ey),
                                   C_YELLOW,1,tipLength=0.5,line_type=cv2.LINE_AA)

            if locked_id is not None and locked_id in tracker.tracks:
                draw_trail(canvas, tracker.tracks[locked_id])
                box = tracker.boxes.get(locked_id)
                if box:
                    draw_zoom_inset(canvas, frame, box)
                    cx_=(box[0]+box[2])//2; cy_=(box[1]+box[3])//2
                    r0 = max(box[2]-box[0],box[3]-box[1])//2+8
                    for ri in range(3):
                        a=int(200-ri*55)
                        cv2.circle(canvas,(cx_,cy_),r0+ri*9,
                                  tuple(int(c*a/255) for c in C_MAGENTA),
                                  1,cv2.LINE_AA)
            elif locked_id is not None:
                locked_id = None

            draw_hud(canvas, len(matched), locked_id, frame_idx, fps, total)
            draw_logo(canvas)

            writer.write(canvas)

            disp = cv2.resize(canvas, (min(W,1280), min(H,720)),
                              interpolation=cv2.INTER_LINEAR)
            cv2.imshow("YOLOvX Aerial Tracker", disp)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('r'): locked_id=None; print("[RESET] lock cleared")
            elif key == ord(' '): paused = not paused; print(f"[{'PAUSE' if paused else 'RESUME'}]")

            if not paused:
                frame_idx += 1; pbar.update(1)

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"[DONE] {output_path} ✓")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolovx_aerial_tracker.py  <video.mp4>  [output.mp4]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "yolovx_output.mp4"
    process(sys.argv[1], out)