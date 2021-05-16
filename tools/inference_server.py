'''
server中使用GPU做infer
面向工业瓷砖大赛demo
'''
from tools.inferServer import inferServer
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
    def __init__(self, model):
        super().__init__(model)
        print("init_myserver")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model = model.to(device)
        self.model = model

    def pre_process(self, request):
        # print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        # file_t = request.files['img_t']
        # print(file.filename)
        file_data = file.read()
        # file_data_t = file_t.read()
        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        # img_t = cv2.imdecode(np.frombuffer(file_data_t, np.uint8), cv2.IMREAD_COLOR)
        return [img, file.filename]

    # pridict default run as follow：
    def pridect(self, data):
        # ret = self.model(data)
        img, filename = data
        st = time.time()
        result = inference_detector(self.model, img)
        # print("infer time (second): ", time.time() - st)
        return [result, filename]

    def post_process(self, result):
        data, filename = result
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
        predict_result = []
        max_score = 0
        flag = False
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
            if int(label) == 0:
                if score >= 0.1:
                    flag = True
            if int(label) == 5:
                if score >= 0.005:
                    flag = True
        if max_score < 0.3 and not flag:
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
    # t = time.time()
    # config_file = 'configs/experiment/round2/un_re2101.py'
    # checkpoint_file = 'work_dirs/un2101.pth'
    # checkpoint_file = 'work_dirs/un_re2101/epoch_36.pth'

    # config_file = 'configs/experiment/round2/un_re250.py'
    # checkpoint_file = 'work_dirs/un250.pth'
    # checkpoint_file = 'work_dirs/un_re250/epoch_36.pth'

    # config_file = 'configs/experiment/round2/cas_dcn_r50.py'
    # checkpoint_file = 'work_dirs/cas_dcn_r50.pth'

    # config_file = 'configs/experiment/round2/cas_res2net101.py'
    # checkpoint_file = 'work_dirs/cas_res2net101.pth'

    # config_file = 'configs/experiment/round2/cas_s50.py'
    # checkpoint_file = 'work_dirs/cas_s50.pth'

    # config_file = 'configs/experiment/round2/cas_dcn_r101.py'
    # checkpoint_file = 'work_dirs/cas_dcn_r101.pth'

    # build the model from a config file and a checkpoint file
    mymodel = init_detector(config_file, checkpoint_file, device='cuda:0')
    myserver = myserver(mymodel)
    # run your server, defult ip=localhost port=8080 debuge=false
    # myserver.run(debuge=True)  # myserver.run("127.0.0.1", 1234)
    myserver.run(debuge=False)  # myserver.run("127.0.0.1", 1234)
    # print("inference total time: ", time.time()-t)