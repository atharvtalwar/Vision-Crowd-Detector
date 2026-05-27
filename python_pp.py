import cv2
import imutils
import numpy as np
import argparse

# ── Load YOLOv4-tiny ──────────────────────────────────────────────────────────
net = cv2.dnn.readNet(
    "models/yolov4-tiny.weights",
    "models/yolov4-tiny.cfg"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detect(frame, conf_threshold=0.3, nms_threshold=0.3):
    h, w = frame.shape[:2]

    # Run at larger resolution for better small-person detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > conf_threshold:
                cx, cy, bw, bh = (detection[:4] * np.array([w, h, w, h])).astype(int)
                x = cx - bw // 2
                y = cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    person_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = confidences[i]
            person_count += 1
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {person_count} ({conf:.0%})',
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, 'Status: Detecting', (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person_count}', (40, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('output', frame)
    return frame


def detectByCamera(output_path):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        video = cv2.VideoCapture(1)
    if not video.isOpened():
        print('Unable to open camera.')
        return

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = None
    print('Detecting people... Press Q to quit.')

    while True:
        check, frame = video.read()
        if not check or frame is None:
            break

        frame = imutils.resize(frame, width=640)
        frame = detect(frame)

        if output_path is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer is not None:
        writer.release()
    video.release()
    cv2.destroyAllWindows()


def detectByPathVideo(path, output_path):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Video Not Found. Please Enter a Valid Path.')
        return

    # Get original FPS to play back at correct speed
    fps = video.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = None
    print('Detecting people... Press Q to quit.')

    while video.isOpened():
        check, frame = video.read()
        if not check or frame is None:
            break

        frame = imutils.resize(frame, width=640)
        frame = detect(frame)

        if output_path is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, int(fps), (w, h))
            writer.write(frame)

        # Sync display to video FPS — was waitKey(1) which plays too fast
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
            break

    if writer is not None:
        writer.release()
    video.release()
    cv2.destroyAllWindows()


def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    if image is None:
        print('Image Not Found. Please Enter a Valid Path.')
        return

    image = imutils.resize(image, width=640)
    result = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def humanDetector(args):
    if args['camera']:
        print('[INFO] Opening Web Cam.')
        detectByCamera(args['output'])
    elif args['video'] is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(args['video'], args['output'])
    elif args['image'] is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(args['image'], args['output'])
    else:
        print('[INFO] No input provided. Use -v, -i, or -c.')


def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default=None, help="path to Video File")
    ap.add_argument("-i", "--image", default=None, help="path to Image File")
    ap.add_argument("-c", "--camera", action="store_true", help="Use the camera.")
    ap.add_argument("-o", "--output", type=str, help="path to optional output file")
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = argsParser()
    humanDetector(args)