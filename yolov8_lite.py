import os
import cv2
import argparse
from rknnlite.api import RKNNLite
from coco_utils import COCO_test_helper
import numpy as np
np.set_printoptions(threshold=np.inf)

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow",
           "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut",
           "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ",
           "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ",
           "hair drier", "toothbrush ")

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # # basic params
    parser.add_argument('--model_path', type=str, default='./yolov8n.rknn', help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3568', help='target RKNPU platform')
    parser.add_argument('--img_folder', type=str, default='./imgs', help='img folder for inference')
    parser.add_argument('--img_save', action='store_true', default=True, help='save the result')

    args = parser.parse_args()
    print(vars(args))
    # init model
    rknn_lite = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(args.model_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    if args.target == 'rk3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()

    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)
    # run test
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i + 1, img_list[i]), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0, 0, 0)
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        input_data = img
        print('--> Running model ')
        outputs = rknn_lite.inference(inputs=[input_data])

        ################################################################
        # Prepare output array
        output = np.array(outputs[0])
        output = output.transpose((0,2,1))
        rows = output.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = output[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    output[0][i][0] - (0.5 * output[0][i][2]),
                    output[0][i][1] - (0.5 * output[0][i][3]),
                    output[0][i][2],
                    output[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            box = co_helper.get_real_box(box)

            draw_bounding_box(
                img_src,
                class_ids[index],
                scores[index],
                round(box[0]),
                round(box[1]),
                round(box[2]+box[0]),
                round(box[3]+box[1]),
            )
        cv2.imwrite('./results/'+img_name,img_src)
