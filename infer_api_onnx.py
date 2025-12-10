import os
import cv2
import numpy as np

import onnxruntime as ort

import torch
import torchvision

from loguru import logger
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


CLASSES = ('pine_wilt', )


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


def decode_outputs(outputs, dtype=torch.float32):
    grids = []
    strides = []
    _hw = [(128, 128), (64, 64), (32, 32)]
    _strides = (8, 16, 32)
    for (hsize, wsize), stride in zip(_hw, _strides):
        yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs = torch.cat([
        (outputs[..., 0:2] + grids) * strides,
        torch.exp(outputs[..., 2:4]) * strides,
        outputs[..., 4:]
    ], dim=-1)
    return outputs


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


class InferAPI:

    def __init__(self, onnx_path, img_size=(1024, 1024), conf_th=0.01):
        self.model = self._load_model(onnx_path)
        self.img_size = img_size
        self.conf_th = conf_th

    def _load_model(self, onnx_path):
        opts = ort.SessionOptions()
        model = ort.InferenceSession(onnx_path,
                                     sess_options=opts,
                                     # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                                     providers=[("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
                                     )
        return model

    def preprocess(self, img_cv2):
        logger.debug(f'org_shape: {img_cv2.shape}')
        img_data = cv2.resize(img_cv2, self.img_size).astype(np.float32)
        img_t = np.transpose(img_data, axes=(2, 0, 1))[None, ...] / 255.0
        logger.debug(f'img_t: {img_t.shape}')
        return img_t

    def predict(self, img_cv2):
        org_h, org_w = img_cv2.shape[0:2]
        img_t = self.preprocess(img_cv2)
        scale_w = org_w / img_t.shape[3]
        scale_h = org_h / img_t.shape[2]
        logger.debug(f'scale_w: {scale_w}')
        logger.debug(f'scale_h: {scale_h}')

        # inference
        output = self.model.run(['output'], {'input': img_t})[0]
        output = torch.from_numpy(output)
        logger.debug(f"model_output: {output.shape}")
        # output = decode_outputs(output)
        predictions = postprocess(output,
                                  num_classes=len(CLASSES),
                                  conf_thre=self.conf_th,
                                  nms_thre=0.3,
                                  class_agnostic=False)
        # logger.debug(f'prediction: {predictions[0].shape}')

        # parse prediction for batch
        pred_labels = []
        for i, p in enumerate(predictions):
            if p is not None:
                p = p.cpu().detach().numpy()
                logger.debug(f'[PRED] img_{i}, {p.shape}')  # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                # logger.debug(p)
                p[:, 0:4:2] *= scale_w
                p[:, 1:4:2] *= scale_h
                # logger.debug(p)
                p[:, 0:4] = xyxy2cxcywh(p[:, 0:4])
            else:
                # logger.debug(f'[PRED] img_{i}, []')
                p = np.empty(shape=(0, 7), dtype=np.float32)
            logger.debug(f'[PRED] img_{i}, {p}')
            # pred_bboxes = np.concatenate([p[:, 0:4],
            #                               np.ones((p.shape[0], 1), dtype=np.float32) * 1.5702,
            #                               p[:, 4:5],
            #                               ], axis=1)
            pred_labels.append(p)  # [pred_bboxes] means cls_id=0 of this img
        print('\n')

        # visual
        visual_dir = 'results'
        os.makedirs(visual_dir, exist_ok=True)
        if visual_dir is not None:
            # img_id = img_ids[i]
            pred = pred_labels[i]
            gt = np.empty(shape=(0, 5), dtype=np.float32) #gt_labels[i]
            '''
            pred: [N, 7], (cx, cy, w, h, conf, cls_score, cls_id)
            gt: [N, 5), (cx, cy, w, h, score)
            '''
        return pred_labels

    def predict_for_labelme(self, img_cv2):
        org_h, org_w = img_cv2.shape[0:2]
        img_t = self.preprocess(img_cv2)
        scale_w = org_w / img_t.shape[3]
        scale_h = org_h / img_t.shape[2]
        logger.debug(f'scale_w: {scale_w}')
        logger.debug(f'scale_h: {scale_h}')

        # inference
        output = self.model.run(['output'], {'input': img_t})[0]
        output = torch.from_numpy(output)
        logger.debug(f"model_output: {output.shape}")
        # output = decode_outputs(output)
        predictions = postprocess(output, num_classes=len(CLASSES), conf_thre=0.01, nms_thre=0.4, class_agnostic=False)
        # logger.debug(f'prediction: {predictions[0].shape}')

        # parse prediction for batch
        pred_labels = []
        for i, p in enumerate(predictions):
            if p is not None:
                p = p.cpu().detach().numpy()
                logger.debug(f'[PRED] img_{i}, {p.shape}')  # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                # logger.debug(p)
                p[:, 0:4:2] *= scale_w
                p[:, 1:4:2] *= scale_h
                # logger.debug(p)
                # p[:, 0:4] = xyxy2cxcywh(p[:, 0:4])
            else:
                # logger.debug(f'[PRED] img_{i}, []')
                p = np.empty(shape=(0, 7), dtype=np.float32)
            logger.debug(f'[PRED] img_{i}, {p}')
            # pred_bboxes = np.concatenate([p[:, 0:4],
            #                               np.ones((p.shape[0], 1), dtype=np.float32) * 1.5702,
            #                               p[:, 4:5],
            #                               ], axis=1)
            pred_labels.append(p)  # [pred_bboxes] means cls_id=0 of this img
        return pred_labels


def eval():

    onnx_path = './weights/pine_wilt_gsd10_s1024_exp4m10_20251210.onnx'
    infer = InferAPI(onnx_path=onnx_path, img_size=(1024, 1024), conf_th=0.05)

    # load image and do inference
    img_path = './cM3_10_18_png.rf.50a5a5311f1dc295a84d270276ccf841.jpg'

    org_img = cv2.imread(img_path)
    infer.predict(org_img)



if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    eval()







