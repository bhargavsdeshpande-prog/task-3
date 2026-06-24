import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)[:, :4]

        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0

    def update(self, bbox):
        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape(-1)

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    return w * h / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 1e-6)

class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            matched = False
            for t in self.trackers:
                if iou(det, t.predict()) > 0.3:
                    t.update(det)
                    updated_tracks.append((t.id, det))
                    matched = True
                    break
            if not matched:
                t = KalmanBoxTracker(det)
                self.trackers.append(t)
                updated_tracks.append((t.id, det))

        return updated_tracks