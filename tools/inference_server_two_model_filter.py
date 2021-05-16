'''
server中使用GPU做infer
面向工业瓷砖大赛demo
'''
from tools.inferServer_two_model import inferServer
import json
from mmdet.apis import init_detector, inference_detector, async_inference_detector
import cv2
import numpy as np
import time


# RET = {
#     "name": "226_46_t20201125133518273_CAM1.jpg",
#     "image_height": 6000,
#     "image_width": 8192,
#     "category": 4,
#     "bbox": [
#         1587,
#         4900,
#         1594,
#         4909
#     ],
#     "score": 0.130577
# }


class myserver(inferServer):
    def __init__(self, model1, model2):
        super().__init__(model1, model2)
        print("init_myserver")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model = model.to(device)
        self.model1 = model1
        self.model2 = model2

    def pre_process(self, request):
        print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        file_t = request.files['img_t']
        # print(file.filename)
        file_data = file.read()
        file_data_t = file_t.read()
        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        img_t = cv2.imdecode(np.frombuffer(file_data_t, np.uint8), cv2.IMREAD_COLOR)
        return [img, img_t, file.filename]

    # pridict default run as follow：
    def pridect(self, data):
        # ret = self.model(data)
        img, img_t, filename = data
        imgOut = None
        try:
            surf = cv2.xfeatures2d.SURF_create()
            kp, des = surf.detectAndCompute(img, None)
            kp_t, des_t = surf.detectAndCompute(img_t, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, des_t, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append(m)
            if len(good) > 4:
                ptsA = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                ptsB = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                ransacReprojThreshold = 4
                H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
                imgOut = cv2.warpPerspective(img_t, H, (img.shape[1], img.shape[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except:
            print("match error")
            pass

        # st = time.time()
        if imgOut is not None:
            result1 = inference_detector(self.model1, img)
            result2 = inference_detector(self.model2, imgOut)
            result = [result1, result2]
        else:
            result1 = inference_detector(self.model1, img)
            result = [result1, None]
        # print("infer time (second): ", time.time() - st)
        return [result, filename]

    def post_process_single(self, data):
        if isinstance(data, tuple):
            bbox_result, _ = data
        else:
            bbox_result = data
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # xmin, ymin, xmax, ymax, score
        return bboxes, labels

    def nms(self, dets, thresh=0.3):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def post_process(self, result):
        data, filename = result
        data1, data2 = data
        if data2 is not None:
            bboxes1, labels1 = self.post_process_single(data1)
            bboxes2, labels2 = self.post_process_single(data2)

            label2bboxs_raw = {}
            label2bboxs_temp = {}

            for box, label in zip(bboxes1, labels1):
                if label + 1 not in label2bboxs_raw:
                    label2bboxs_raw[label + 1] = []
                label2bboxs_raw[label + 1].append(box)
            for box, label in zip(bboxes2, labels2):
                if label + 1 not in label2bboxs_temp:
                    label2bboxs_temp[label + 1] = []
                label2bboxs_temp[label + 1].append(box)

            predict_result = []
            max_score = 0
            flag = False
            for kind in range(1, 8 + 1, 1):
                if kind not in label2bboxs_raw:
                    continue
                boxes_kind = np.array(label2bboxs_raw[kind])
                for bbox in boxes_kind:
                    flag_rm = 0
                    xmin, ymin, xmax, ymax, score = bbox
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    dict_instance = dict()
                    dict_instance['name'] = str(filename)
                    dict_instance['category'] = int(kind)
                    score = round(float(score), 6)
                    dict_instance["score"] = score
                    dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
                    if kind == 1 or kind == 6:
                        predict_result.append(dict_instance)
                        max_score = max(score, max_score)
                    else:
                        temp_boxes_kind = np.array(label2bboxs_temp[kind])
                        for temp_bbox in temp_boxes_kind:
                            txmin, tymin, txmax, tymax, tscore = temp_bbox
                            txmin, tymin, txmax, tymax = int(txmin), int(tymin), int(txmax), int(tymax)
                            tscore = round(float(tscore), 6)
                            if tscore > 0.3:
                                if abs(txmin - xmin) < 10 and abs(txmax - xmax) < 10 \
                                    and abs(tymin - ymin) < 10 and abs(tymax - ymax) < 10:
                                    flag_rm = 1
                    if flag_rm == 0:
                        predict_result.append(dict_instance)
                        max_score = max(score, max_score)
                    # if int(kind) == 1:
                    #     if score >= 0.1:
                    #         flag = True
                    # if int(kind) == 6:
                    #     if score >= 0.005:
                    #         flag = True
            if max_score < 0.3:
                predict_result = []
            return json.dumps(predict_result)
        else:
            if isinstance(data1, tuple):
                bbox_result, _ = data1
            else:
                bbox_result = data1
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            predict_result = []
            max_score = 0
            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax, score = bbox
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                dict_instance = dict()
                dict_instance['name'] = str(filename)
                dict_instance['category'] = int(label) + 1
                score = round(float(score), 6)
                dict_instance["score"] = score
                dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
                predict_result.append(dict_instance)
                max_score = max(score, max_score)
                # if int(label) == 0:
                #     if score >= 0.1:
                #         flag = True
                # if int(label) == 5:
                #     if score >= 0.005:
                #         flag = True
            if max_score < 0.3:
                predict_result = []
            return json.dumps(predict_result)


# 赛题数据同学的output2result函数供大家参考，infer_score_thre为自定义阈值
# def output2result(result, name, infer_score_thre):
#     image_name = name
#     predict_rslt = []
#     for i, res_perclass in enumerate(result):
#         class_id = i + 1
#         for per_class_results in res_perclass:
#             xmin, ymin, xmax, ymax, score = per_class_results
#             if score < infer_score_thre:
#                 continue
#             xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
#             dict_instance = dict()
#             dict_instance['name'] = image_name
#             dict_instance['category'] = class_id
#             dict_instance["score"] = round(float(score), 6)
#             dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
#             predict_rslt.append(dict_instance)
#
#     return predict_rslt


if __name__ == '__main__':
    t = time.time()
    # config_file1 = 'configs/experiment/round2/un_re2101.py'
    # checkpoint_file1 = 'work_dirs/un2101.pth'

    # config_file = 'configs/experiment/round2/un_re250.py'
    # checkpoint_file = 'work_dirs/un250.pth'

    config_file1 = 'configs/experiment/round2/cas_dcn_r50.py'
    checkpoint_file1 = 'work_dirs/cas_dcn_r50.pth'

    config_file2 = 'configs/experiment/round2/cas_dcn_r50_single.py'
    checkpoint_file2 = 'work_dirs/cas_dcn_r50.pth'
    # config_file2 = 'configs/experiment/round2/cas_dcn_r101.py'
    # checkpoint_file2 = 'work_dirs/cas_dcn_r101.pth'
    # build the model from a config file and a checkpoint file
    mymodel1 = init_detector(config_file1, checkpoint_file1, device='cuda:0')
    mymodel2 = init_detector(config_file2, checkpoint_file2, device='cuda:0')
    myserver = myserver(mymodel1, mymodel2)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=True)  # myserver.run("127.0.0.1", 1234)
    print("all infer time: ", time.time() - t)
