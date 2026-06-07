import cv2
import numpy as np
import time
from collections import defaultdict, deque


class CCTVAnalytics:
    def __init__(self, frame_w=1280, frame_h=720):
        self.w = frame_w
        self.h = frame_h
        self.heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
        self.track_history = defaultdict(lambda: deque(maxlen=300))

        # Thresholds
        self.loiter_sec        = 8.0
        self.loiter_spread_px  = 100
        self.run_speed_px_s    = 90
        self.crowd_threshold   = 5

        self.suspicious_ids = {}          # tid -> reason string
        self.alert_log      = deque(maxlen=25)
        self.frame_count    = 0
        self.density_grid   = [[0]*4 for _ in range(4)]

    # ── Main update ────────────────────────────────────────────────────────
    def update(self, tracked_persons, frame_shape):
        now = time.time()
        self.frame_count += 1
        h, w = frame_shape[:2]
        current_ids = set()

        # Decay heatmap every 3rd frame
        if self.frame_count % 3 == 0:
            self.heatmap *= 0.997

        for p in tracked_persons:
            tid = p.get('track_id', -1)
            if tid < 0:
                continue
            x1, y1, x2, y2 = p['box']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_ids.add(tid)

            # Heatmap accumulation (scaled to heatmap resolution)
            hx = int(np.clip(cx * self.w / w, 0, self.w - 1))
            hy = int(np.clip(cy * self.h / h, 0, self.h - 1))
            cv2.circle(self.heatmap, (hx, hy), 28, 2.5, -1)
            np.clip(self.heatmap, 0, 255, out=self.heatmap)

            self.track_history[tid].append((cx, cy, now))

        # Remove tracks absent >5 s
        stale = [t for t, q in self.track_history.items()
                 if t not in current_ids and q and now - q[-1][2] > 5]
        for t in stale:
            del self.track_history[t]

        # ── Suspicious detection ───────────────────────────────────────────
        new_suspicious = {}
        for tid, hist in self.track_history.items():
            if tid not in current_ids or len(hist) < 5:
                continue
            hl = list(hist)

            # Loitering
            span = hl[-1][2] - hl[0][2]
            if span >= self.loiter_sec:
                xs = [x for x, y, t in hl]
                ys = [y for x, y, t in hl]
                spread = max(max(xs)-min(xs), max(ys)-min(ys))
                if spread < self.loiter_spread_px:
                    reason = f'LOITERING {span:.0f}s'
                    new_suspicious[tid] = reason
                    if self.suspicious_ids.get(tid, '').split()[0] != 'LOITERING':
                        self._log(f'⚠️ Person #{tid} loitering {span:.0f}s', 'loiter')

            # Running
            if len(hl) >= 4:
                rec = hl[-4:]
                dt  = rec[-1][2] - rec[0][2]
                if dt > 0.05:
                    dx    = rec[-1][0] - rec[0][0]
                    dy    = rec[-1][1] - rec[0][1]
                    speed = ((dx**2 + dy**2)**0.5) / dt
                    if speed > self.run_speed_px_s:
                        reason = f'RUNNING {speed:.0f}px/s'
                        if tid not in new_suspicious:
                            new_suspicious[tid] = reason
                        if self.suspicious_ids.get(tid, '').split()[0] != 'RUNNING':
                            self._log(f'🏃 Person #{tid} running fast', 'run')

        self.suspicious_ids = new_suspicious

        # ── Density grid (4×4) ─────────────────────────────────────────────
        grid = [[0]*4 for _ in range(4)]
        cw   = w / 4
        ch   = h / 4
        for p in tracked_persons:
            x1, y1, x2, y2 = p['box']
            col = min(int(((x1+x2)/2) / cw), 3)
            row = min(int(((y1+y2)/2) / ch), 3)
            grid[row][col] += 1

        for r in range(4):
            for c in range(4):
                if grid[r][c] >= self.crowd_threshold:
                    msg = f'👥 Crowd Zone({r},{c}): {grid[r][c]} persons'
                    last = [a['msg'] for a in list(self.alert_log)[-4:]]
                    if msg not in last:
                        self._log(msg, 'crowd')

        self.density_grid = grid

        return {
            'total':             len(tracked_persons),
            'suspicious_count':  len(new_suspicious),
            'suspicious':        {str(k): v for k, v in new_suspicious.items()},
            'density_grid':      grid,
            'alerts':            list(self.alert_log)[-12:],
        }

    def _log(self, msg, kind):
        self.alert_log.append({
            'time': time.strftime('%H:%M:%S'),
            'msg':  msg,
            'type': kind
        })

    # ── Heatmap overlay ────────────────────────────────────────────────────
    def get_heatmap_overlay(self, frame):
        h, w = frame.shape[:2]
        hm = cv2.resize(self.heatmap, (w, h))
        if hm.max() > 0:
            hm_u8 = (hm / hm.max() * 255).astype(np.uint8)
        else:
            hm_u8 = np.zeros((h, w), dtype=np.uint8)
        hm_u8 = cv2.GaussianBlur(hm_u8, (21, 21), 0)
        colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.5, colored, 0.5, 0)

    def reset_heatmap(self):
        self.heatmap[:] = 0
