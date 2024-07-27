import cv2
import matplotlib

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

def draw_keypoints(outputs, image):
    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        if outputs[0]['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                            3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for ie, e in enumerate(edges):
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                pt1 = (int(keypoints[e[0], 0]), int(keypoints[e[0], 1]))
                pt2 = (int(keypoints[e[1], 0]), int(keypoints[e[1], 1]))
                cv2.line(image, pt1, pt2, tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return image