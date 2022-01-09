
import time
import cv2
import mtcnn.caffe_pb2 as pb
import numpy as np
from multiprocessing import Process,Manager



class Module:
    # def __init__(self):
    #     pass

    def __call__(self, *arg):
        return self.forward(*arg)

    def get_modules(self):
        ms = []
        all_attr = self.__dict__
        for attr in all_attr:
            attr = all_attr[attr]
            if isinstance(attr, Module):
                ms.append(attr)
            elif "__iter__" in dir(attr):
                for i in attr:
                    if isinstance(i, Module):
                        ms.append(i)
        return ms

    def get_params(self):
        modules = self.get_modules()
        params = []
        all_att = self.__dict__
        for att in all_att:
            att = all_att[att]
            if isinstance(att, Parameter):
                params.append(att)
        for m in modules:
            params.extend(m.get_params())
        return params


class Linear(Module):
    def __init__(self, input_num, output_num):
        self.params = Parameter(np.random.normal(0, 1 / np.sqrt(input_num), size=(input_num, output_num)))
        # self.weight = Parameter(np.random.normal(0, 1 ,size=(input_num,output_num)))
        self.bias = Parameter(np.zeros((1, output_num)))

    def forward(self, x):  # 子类和父类同名函数: 重写, 不会再去管父类的, 如果想调用, 只能通过 super.forward()
        self.x = x
        return x @ self.params.value + self.bias.value

    def backward(self, G):
        self.params.grad += self.x.T @ G
        self.bias.grad += np.sum(G, axis=0)
        back_G = G @ self.params.value.T

        return back_G


class Parameter:
    def __init__(self, param):
        self.value = param
        self.grad = np.zeros(param.shape)

    def zero_grad(self):
        self.grad[...] = 0


class SigmoidLayer(Module):
    def sigmoid(self, x):  # sigmoid 函数
        xtemp = x.copy()
        epx = 0.0001
        p = xtemp < 0
        p1 = xtemp >= 0
        xtemp[p] = np.exp(xtemp[p]) / (np.exp(xtemp[p]) + 1)
        xtemp[p1] = 1 / (1 + np.exp(-xtemp[p1]))
        return np.clip(xtemp, a_min=epx, a_max=1 - epx)

    def forward(self, x):
        self.x = x
        return self.sigmoid(x)

    def backward(self, G):
        return G * (self.sigmoid(self.x) * (1 - self.sigmoid(self.x)))


class ReLU(Module):
    def __init__(self, inplace=True):
        # super().__init__("ReLU")
        self.inplace = inplace

    def forward(self, x):
        self.x_negative = x < 0
        if not self.inplace:
            x = x.copy()

        x[self.x_negative] = 0
        return x

    def backward(self, G):
        if not self.inplace:
            G = G.copy()

        G[self.x_negative] = 0
        return G


class SoftmaxCrossEntropy(Module):

    @classmethod
    def softmax(cls, x):
        # max_x = `np.max(x)
        # tempx = np.exp(x - max_x)
        # tempsum = np.sum(tempx, axis=1, keepdims=True)
        # result = tempx / tempsum
        #
        # return result
        expx = np.exp(x)
        sumx = np.sum(expx, axis=1, keepdims=True)
        return expx / sumx

    def forward(self, x, label_one_hot):
        self.label_one_hot = label_one_hot
        self.pro = self.softmax(x)
        self.batch_size = self.pro.shape[0]
        loss = -np.sum(label_one_hot * np.log(self.pro)) / self.batch_size
        return loss

    def backward(self):
        return (self.pro - self.label_one_hot) / self.batch_size

    def get_predict(self):
        self.predict = np.argmax(self.pro, axis=1).reshape(-1, 1)
        return self.predict


class DropOut(Module):
    def __init__(self, keep_live=0.7):
        self.keep_live = keep_live

    def forward(self, x):
        temp_mask = np.random.uniform(0, 1, x.shape)
        self.mask = temp_mask < 1 - self.keep_live
        x[self.mask] = 0

        return x / (self.keep_live + 0.0001)

    def backward(self, G):
        G[self.mask] = 0
        return G / (1 + self.keep_live)


class Flatten(Module):
    def forward(self, x):
        # self.x = x.copy()
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, G):
        return G.reshape(self.shape)


class Sequencial(Module):
    def __init__(self, *layers):
        self.items = list(layers)

    def get_modules(self):
        return self.items

    def forward(self, x):
        for layer in self.items:
            x = layer(x)
        return x

    def backward(self, G):
        for layer in self.items[::-1]:
            G = layer.backward(G)
        return G


class PReLU(Module):
    def __init__(self, size):
        # self.params = Parameter(np.array(mtcnn_param_dict.blobs[0].data).reshape(-1, 1))
        self.params = Parameter(np.random.normal(0, 1, size=(size, 1)))

    def forward(self, x):  #

        for i in range(x.shape[1]):
            x[:, i][x[:, i] < 0] *= self.params.value[i]

        return x

    def backward(self):
        pass


class Conv2d(Module):

    def __init__(self, outc, inc, kernel_size):
        self.params = Parameter(
            np.random.normal(0, 1 / np.sqrt(kernel_size ** 2 * inc), size=(outc, inc, kernel_size, kernel_size)))
        self.bias = Parameter(np.zeros((outc, 1)))
        self.stride = 1
        self.kernel_size = kernel_size
    def forward(self, x):
        self.x = x.copy()
        result = self.convolution(x)

        return result


    def convolution2(self, itensor):
        # itensor = self.pad(itensor)
        n, ic, ih, iw = itensor.shape  # 图像像素的维度  , n : 几张图片
        self.itensor_shape = itensor.shape
        ks, kn, kh, kw = self.params.value.shape  # 卷积核的大小   ks : 几组核, kn 通道数
        s = kn * kh * kw  # 填充数据的行大小
        w = (ih - kh + 1) * (iw - kw + 1)  # 填充数据的列大小
        self.column = np.zeros((s, w * n))  # 先创建0元素
        self.kcol = self.params.value.reshape(ks, s)
        out_image_nums = n * ks
        c = 0
        for p in range(n):
            for h in range(0, ih - kh + 1, self.stride):
                for l in range(0, iw - kw + 1, self.stride):
                    temp_item = itensor[p, :, h:h + kh, l:l + kw].reshape(-1, 1)
                    self.column[:, None, c] = temp_item
                    c += 1

        output = (self.kcol @ self.column + self.bias.value)
        self.shape = output.shape
        output = output.reshape(out_image_nums, ih - kh + 1, iw - kw + 1)

        return_result = [[] for x in range(n)]  # 对图片做调整, 按照同一张图片和不同的通道组合
        for i in range(out_image_nums):
            return_result[i % n].append(output[i])
        return np.array(return_result)

    def convolution(self, itensor):
        img_num , img_c , img_h, img_w = itensor.shape

        col = self.params.value.reshape(self.params.value.shape[0], -1)

        column = np.zeros((img_num,col.shape[1],(img_h-self.kernel_size+1) * (img_w-self.kernel_size +1)))

        c = 0

        for h in range(img_h-self.kernel_size+1):
            for w in range(img_w-self.kernel_size +1):
                column[:,:,c,None] = itensor[:,:,h:h+self.kernel_size,w:w+self.kernel_size].reshape(img_num,-1,1)
                c += 1
        result = col @ column + self.bias.value
        self.shape = result.shape
        result = result.reshape(img_num,-1,img_h - self.kernel_size + 1 , img_w - self.kernel_size + 1)

        return result


    def backward(self, G):

        G = G.reshape(self.shape)
        temp_grad = G @ self.column.T
        self.param.grad = temp_grad.reshape(self.param.value.shape)
        temp_back_G = self.kcol.T @ G
        back_G = np.zeros(self.itensor_shape)
        n, ic, ih, iw = self.itensor_shape
        ks, kn, kh, kw = self.param.value.shape
        c = 0
        for p in range(n):
            for h in range(0, ih - kh + 1, self.stride):
                for l in range(0, iw - kw + 1, self.stride):
                    item = temp_back_G[:, c].reshape(kn, kh, kw)
                    back_G[p, :, h:h + kh, l:l + kw] += item
                    c += 1
        return back_G


    def backward2(self, G):

        result_G = np.zeros((G.shape[0] * G.shape[1], G.shape[2], G.shape[3]))
        temp_G = G.reshape(result_G.shape)
        sign = 0
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                result_G[i + j * G.shape[0], :, :] = temp_G[sign, :, :]
                sign += 1
        G = result_G.reshape(G.shape)
        G = G.reshape(self.shape)
        temp_grad = G @ self.column.T
        self.param.grad = temp_grad.reshape(self.param.value.shape)
        temp_back_G = self.kcol.T @ G
        back_G = np.zeros(self.itensor_shape)
        n, ic, ih, iw = self.itensor_shape
        ks, kn, kh, kw = self.param.value.shape
        c = 0
        for p in range(n):
            for h in range(0, ih - kh + 1, self.stride):
                for l in range(0, iw - kw + 1, self.stride):
                    item = temp_back_G[:, c].reshape(kn, kh, kw)
                    back_G[p, :, h:h + kh, l:l + kw] += item
                    c += 1
        return back_G


class Pooling(Module):

    def __init__(self, kernel_size=2, stride=2):

        self.kernel_size = kernel_size
        self.stride = stride


    def forward(self,x):
        result_w = int(np.ceil((x.shape[3] - self.kernel_size)/self.stride + 1))
        result_h = int(np.ceil((x.shape[2] - self.kernel_size)/self.stride + 1))

        result = np.zeros((x.shape[0],x.shape[1],result_h,result_w))

        for h in range(result_h):
            t_h = h*self.stride
            for w in range(result_w):
                t_w = w*self.stride
                max_num = np.max(x[:,:,t_h:t_h+self.kernel_size,
                                 t_w:t_w+self.kernel_size],axis=(2,3))
                result[:,:,h,w] = max_num
        return result


class Pnet(Module):
    def __init__(self, model_path):
        # pent_param = load_mtcnn_param(model_path)
        self.backbone = Sequencial(
            Conv2d(10, 3, 3),
            PReLU(10),
            Pooling(),
            Conv2d(16, 10, 3),
            PReLU(16),
            Conv2d(32, 16, 3),
            PReLU(32)
        )
        self.head_confidence = Conv2d(2, 32, 1)
        self.head_bbox = Conv2d(4, 32, 1)

        model_name = ["conv1", "PReLU1", "pool1", "conv2", "PReLU2", "conv3", "PReLU3", "conv4-1", "conv4-2"]
        model_objs = self.backbone.items + [self.head_confidence, self.head_bbox]

        load_model_param(model_path, model_name, model_objs)

    def forward(self, x):  # 这里只是做推理, 还没有训练

        x = self.backbone(x)

        result1 = self.head_confidence(x)
        result2 = self.head_bbox(x)

        result1 = SoftmaxCrossEntropy.softmax(result1)

        return result1, result2


class Rnet(Module):
    def __init__(self, model_path):
        self.crop_size = 24
        self.backbone = Sequencial(
            Conv2d(28, 3, 3), PReLU(28), Pooling(3, 2),
            Conv2d(48, 28, 3), PReLU(48), Pooling(3, 2),
            Conv2d(64, 48, 2), PReLU(64),
            Flatten(), Linear(576, 128), PReLU(128)
        )
        self.head_confidence = Linear(128, 2)
        self.head_bbox = Linear(128, 4)

        model_name = ["conv1", "prelu1", "pool1", "conv2", "prelu2", "pool2", "conv3",
                      "prelu3", "flatten", "conv4", "prelu4", "conv5-1", "conv5-2"]  # 组合名称列表
        model_objs = self.backbone.items + [self.head_confidence, self.head_bbox]  # 组合每层的对象列表

        load_model_param(model_path, model_name, model_objs)  # 加载参数, 而不是放在每一个类中加载
        pass

    def forward(self, x):
        x = self.backbone(x)
        return SoftmaxCrossEntropy.softmax(self.head_confidence(x)), self.head_bbox(x)


class Onet(Module):
    def __init__(self, model_path):
        self.crop_size = 48
        self.backbone = Sequencial(
            Conv2d(32,3,3),PReLU(32),Pooling(3,2),
            Conv2d(64,32,3),PReLU(64),Pooling(3,2),
            Conv2d(64,64,3),PReLU(64),Pooling(2,2),
            Conv2d(128,64,2),PReLU(128),
            Flatten(),
            Linear(1152,256), PReLU(256)
        )
        self.head_confidence = Linear(256,2)
        self.head_bbox = Linear(256,4)
        self.head_landmark = Linear(256,10)

        model_name = ["conv1","prelu1","pool1","conv2","prelu2","pool2","conv3","prelu3",
                      "pool3","conv4","prelu4","flatten","conv5","prelu5","conv6-1","conv6-2","conv6-3"]
        model_objs = self.backbone.items + [self.head_confidence , self.head_bbox,self.head_landmark]

        load_model_param(model_path,model_name,model_objs)

    def forward(self,x):
        x = self.backbone(x)
        return SoftmaxCrossEntropy.softmax(self.head_confidence(x)), self.head_bbox(x), self.head_landmark(x)


def load_model_param(model_path, model_name_list, model_objs_list):
    net = pb.NetParameter()  # 读取模型文件, 使用字典存放, 可以使用名称对其索引
    model_param_dict = {}
    with open(model_path, "rb") as f:
        net.ParseFromString(f.read())
    for i in net.layer:
        model_param_dict[i.name] = i

    for name, objs in zip(model_name_list, model_objs_list):
        if name not in model_param_dict:
            continue
        param = model_param_dict[name]
        if isinstance(objs, Conv2d):
            # 这里不加[:] 或者  不写 dtype=np.float都是可以的`
            objs.params.value[:] = np.array(param.blobs[0].data, dtype=np.float32).reshape(objs.params.value.shape)
            objs.bias.value[:] = np.array(param.blobs[1].data, dtype=np.float32).reshape(objs.bias.value.shape)
        elif isinstance(objs, PReLU):
            objs.params.value[:] = np.array(param.blobs[0].data, dtype=np.float32).reshape(objs.params.value.shape)
        elif isinstance(objs, Linear):
            # 注意linear层参数加载
            # 注意linear层参数加载需要内部转职
            objs.params.value[:] = np.array(param.blobs[0].data, dtype=np.float32).reshape(objs.params.value.shape[1],
                                                                                           objs.params.value.shape[0]).T
            objs.bias.value[:] = np.array(param.blobs[1].data, dtype=np.float32).reshape(objs.bias.value.shape)


class BBox:
    def __init__(self, point1x, point1y, point2x, point2y, score):
        # 左上角坐标和右下角坐标
        self.point1x = point1x
        self.point1y = point1y
        self.point2x = point2x
        self.point2y = point2y

        # 长和宽计算
        self.width = self.point2x - self.point1x
        self.height = self.point2y - self.point1y

        # 面积计算
        self.area = self.width * self.height

        self.score = score
        self.center = (self.point2x + self.point1x) * 0.5, (self.point2y + self.point1y) * 0.5
    # # 中心计算
    # def center(self):
    #     return (self.point2x + self.point1x) / 2, (self.point2y + self.point1y) / 2

    def Iou(self, other_bbox):
        point1x = max(self.point1x, other_bbox.point1x)
        point1y = max(self.point1y, other_bbox.point1y)
        point2x = min(self.point2x, other_bbox.point2x)
        point2y = min(self.point2y, other_bbox.point2y)

        if point2x - point1x < 0 or point2y - point1y < 0:  # 这里一定要注意
            return 0

        and_area = np.abs((point2x - point1x) * (point2y - point1y))

        or_area = self.area + other_bbox.area - and_area

        return and_area / (or_area + 0.001)

    def locations(self):
        return self.point1x, self.point1y, self.point2x, self.point2y

    def __repr__(self):
        return f"{self.point1x:.2f},{self.point1y:.2f},{self.point2x:.2f},{self.point2y:.2f},{self.score:.2f}"

    def iou(self, others):

        box =  np.array(self.locations())
        boxes = np.array([(other.locations()) for other in others])

        # 交集面积
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        cross_areas = w * h

        # 计算框面积
        box_area = (box[2] - box[1]) * (box[3] - box[1])

        boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 计算iou
        iou = cross_areas / (box_area + boxes_areas - cross_areas)

        return iou

    def dif(self,other):
        return abs(self.point1x - other.point1x) +abs(self.point1y -other.point1y) + abs(self.point2x - other.point2x)+ abs(self.point2y - other.point2y)


def nms(bbox_objs, nms_T):
    bbox_list = sorted(bbox_objs, key=lambda x: x.score, reverse=True)
    bbox_list_state = np.ones(len(bbox_list))
    bbox_list_new = []
    for index_i, box_i in enumerate(bbox_list):
        if bbox_list_state[index_i] == 0:
            continue
        bbox_list_new.append(box_i)
        # for index_j, box_j in enumerate(bbox_list[index_i+1:]):
        #     if box_i.Iou(box_j) > nms_T:
        #         bbox_list_state[index_j] = 0
        for index_j, box_j in enumerate(bbox_list):
            if bbox_list_state[index_j] == 0 or index_j == index_i:
                continue
            if box_i.Iou(box_j) > nms_T:
                bbox_list_state[index_j] = 0
    return bbox_list_new


class MTCNN(Module):

    def __init__(self):
        self.pnet = Pnet("mtcnn\det1.caffemodel")
        self.rnet = Rnet("mtcnn\det2.caffemodel")
        self.onet = Onet("mtcnn\det3.caffemodel")

    # 图像金字塔
    def pyrdown(self, img_data, min_scale=0.1, max_scale=0.9, factor=0.709):
        now_scale = max_scale
        img_pyrs = []

        while now_scale > min_scale and img_data.shape[1] * now_scale >= 12 \
                and img_data.shape[0] * now_scale >= 12:
            img_pyrs.append([cv2.resize(img_data, None, fx=now_scale, fy=now_scale), now_scale])
            now_scale *= factor


        # print("金字塔已存入")
        return img_pyrs

    def proposal(self,img_pyrs,  conf_T=0.6, nms_T=0.5):

            bbox_list = []
            # print("取出金字塔, 开始pnet")
            # img_pyrs = qu_1.get()
            for img, scale in img_pyrs:
                # 如果放在 pydown 做的话, 会很麻烦
                stride = 2
                cellsize = 12
                img = img.transpose(2, 1, 0)[None]  # transpose和加维度都在这里进行
                conf, reg = self.pnet(img)

                y, x = np.where(conf[0, 1] > conf_T)  # the positive output feature map

                # restore the coordinates to the original image to get the location of the bbox detected by PNet
                for oy, ox in zip(y, x):
                    score = conf[0, 1, oy, ox]  # save the score

                    bx = (oy * stride - 1) / scale
                    by = (ox * stride - 1) / scale
                    br = (oy * stride + cellsize) / scale
                    bb = (ox * stride + cellsize) / scale

                    regx = reg[0, 0, oy, ox]
                    regy = reg[0, 1, oy, ox]
                    regr = reg[0, 2, oy, ox]
                    regb = reg[0, 3, oy, ox]

                    bw = br - bx + 1
                    bh = bb - by + 1

                    bx = bx + regx * bw
                    by = by + regy * bh
                    br = br + regr * bw
                    bb = bb + regb * bh
                    bbox_list.append(BBox(bx, by, br, bb, score))

                # new_bbox_list = nms(bbox_list, nms_T, )
            new_bbox_list = nms(bbox_list, nms_T)
            # print("pnet结果存入")
            # qu_2.put(new_bbox_list)
            return new_bbox_list

    def refine_rnet(self, objs,conf_T=0.7, nms_T=0.4):

            # print("pnet结果取出, rnet开始")
            # objs = qu_2.get()
            if len(objs) == 0:
                return []
            batch_crop_image = []

            for obj in objs:
                x, y, r, b = obj.locations()
                maxl = max(obj.width, obj.height)
                cx, cy = obj.center
                x = cx - maxl * 0.5
                y = cy - maxl * 0.5
                r = cx + maxl * 0.5
                b = cy + maxl * 0.5
                crop_resized = np.zeros((24,24, 3), dtype=np.float32)
                self.crop_resize_to_affine(self.input_img, (x, y, r, b), crop_resized)
                batch_crop_image.append(crop_resized.transpose(2, 1, 0)[None])

            batch_image = np.vstack(batch_crop_image)

            predict = self.rnet(batch_image)
            conf_all, regx_all = predict[:2]


            keep_objs = []
            for batch_index, obj in enumerate(objs):
                conf, reg = conf_all[batch_index, 1], regx_all[batch_index]
                if conf > conf_T:
                    regx, regy, regr, regb = reg
                    maxl = max(obj.width, obj.height)
                    cx, cy = obj.center
                    x = cx - 0.5*maxl + regx * maxl # 这里老师的代码进行了乘法的结合律, 极其  难以理解
                    y = cy - 0.5*maxl + regy * maxl # 拆开括号就好理解了
                    r = cx + 0.5*maxl + regr * maxl
                    b = cy + 0.5*maxl + regb * maxl
                    new_objs = BBox(x,y,r,b,conf)


                    keep_objs.append(new_objs)
            result = nms(keep_objs,nms_T)
            # print("rnet结果存入")
            # qu_3.put(result)
            return result

    def refine_onet(self, objs,conf_T=0.7, nms_T=0.2):

        # while True:
        #     print("rnet结果取出,onet开始")
        #     objs = qu_3.get()
            if len(objs) == 0:
                return []
            batch_crop_image = []

            for obj in objs:

                x, y, r, b = obj.locations()
                maxl = max(obj.width, obj.height)
                cx, cy = obj.center
                x = cx - maxl * 0.5
                y = cy - maxl * 0.5
                r = cx + maxl * 0.5
                b = cy + maxl * 0.5
                crop_resized = np.zeros((48,48, 3), dtype=np.float32)
                self.crop_resize_to_affine(self.input_img, (x, y, r, b), crop_resized)
                batch_crop_image.append(crop_resized.transpose(2, 1, 0)[None])

            batch_image = np.vstack(batch_crop_image)

            predict = self.onet(batch_image)
            conf_all, regx_all = predict[:2]


            landmarks = predict[2]


            keep_objs = []
            for batch_index, obj in enumerate(objs):
                conf, reg = conf_all[batch_index, 1], regx_all[batch_index]
                if conf > conf_T:
                    regx, regy, regr, regb = reg
                    maxl = max(obj.width, obj.height)
                    cx, cy = obj.center
                    x = cx - 0.5*maxl + regx * maxl # 这里老师的代码进行了乘法的结合律, 极其  难以理解
                    y = cy - 0.5*maxl + regy * maxl # 拆开括号就好理解了
                    r = cx + 0.5*maxl + regr * maxl
                    b = cy + 0.5*maxl + regb * maxl
                    new_objs = BBox(x,y,r,b,conf)


                    keys = landmarks[batch_index]
                    new_objs.landmarks = []
                    for x,y in zip(keys[:5],keys[5:]):
                        x = cx - 0.5 * maxl + x*maxl
                        y = cy - 0.5 * maxl + y*maxl
                        new_objs.landmarks.append((x,y))
                    keep_objs.append(new_objs)
            result = nms(keep_objs,nms_T)
            # print("onet结果存入")
            # qu_4.put(result)
            return result

    def crop_resize_to_affine(self, src, roi, dst):
        dh, dw = dst.shape[:2]
        sh, sw = src.shape[:2]
        x, y, r, b = roi
        rcx, rcy = (x + r) / 2, (y + b) / 2
        rh, rw = b - y + 1, r - x + 1
        sx = dw / rw
        sy = dh / rh

        M = np.array([
            [sx, 0, -sx * rcx + dw * 0.5],
            [0, sy, -sy * rcy + dh * 0.5]
        ])
        cv2.warpAffine(src, M, dsize=(dw, dh), dst=dst)

    def my_crop_size(self, data, point1x, point1y, point2x, point2y, size=24):

        result = data[point1y:point2y, point1x:point2x,  :]
        result = cv2.resize(result, (size,size))

        return result.transpose(2, 1, 0)

    def detect(self, image, min_scale=0.08, max_scale=0.1  ):
        self.input_img = image[..., ::-1]
        self.input_img = ((self.input_img - 127.5) / 128.0).astype(np.float32)
        self.img_pyrs = self.pyrdown(self.input_img, min_scale, max_scale)
        self.objs = self.proposal(self.img_pyrs)
        self.robjs = self.refine_rnet(self.objs)
        result = self.refine_onet(self.robjs)
        return result

def detect_process_func(frame_q,objs_q,mtcnn_,min_scale,max_scale): # 检测进程
    time_list = []  # 测试时间
    while True:
        frame = frame_q.get() # 获取画面
        s_time = time.time()
        objs = mtcnn_.detect(frame,min_scale=min_scale,max_scale=max_scale) # 检测
        time_list.append(time.time() - s_time)
        objs_q.put_nowait(objs)  # 存入结果
        if len(time_list) % 50 == 0:
            print(f"平均延迟: {np.mean(time_list):.3f}s")
            time_list.clear()

if __name__ == '__main__':

    old_objs = 0        # 保存上一次检测结果
    cap = cv2.VideoCapture(0) # 启动摄像头
    mtcnn = MTCNN()  # mtcnn实例
    manager = Manager() # 进程管理


    frame_max = 2 # 该值越小, 实时性越好, 但准确率降低, 该值必须大于等于2 ,建议取值 2-5
    fil = 1  # 该值越大, 检测窗口越稳定,但是检测延迟越高, 建议取值: 0~5
    max_scale = 0.1  # 最大金字塔放缩比例    (建议0.1~0.2 , 仅限于自拍场景, 其他场景自行测试)
    min_scale = 0.08    # 最小金字塔放缩比例 (建议 0.08 ~ 0.1, min_scale 于 max_scale 越接近实时性越好)



    frame_q = manager.Queue(frame_max)  # 画面队列
    objs_q = manager.Queue() # 检测结果队列


    detect_pro = Process(target=detect_process_func,args=(frame_q,objs_q,mtcnn,min_scale,max_scale))  # 创建进程
    detect_pro.start() # 启动进程

    while True:
        ret, frame = cap.read()

        if frame_q.full():  # 存入画面
            frame_q.get_nowait()
        frame_q.put_nowait(frame)

        if objs_q.empty() == False:
            objs = objs_q.get_nowait()  # 获取结果
        else:
            objs = old_objs  # 如果结果队列为空, 则使用上次的结果

        if objs:

            if old_objs and  len(objs) == len(old_objs):   # 解决检测窗口抖动问题 , 如果预测的和上一次结果不大, 就不改变检测结果
                for index in range(len(objs)):
                    for (x, y) ,(x2,y2) in zip(objs[index].landmarks,old_objs[index].landmarks):
                        if (abs(x-x2)<fil) and (abs(y-y2)<fil):
                            objs = old_objs

            for obj in objs:   # 画出最终结果

                bx, by, br, bb = np.round(obj.locations()).astype(np.int32)
                cv2.rectangle(frame, (bx, by), (br, bb), (0, 255, 0), 2)

                for x, y in obj.landmarks:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1, 16)

        old_objs = objs  # 保留当前框
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


