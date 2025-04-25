from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    # model = YOLO("yolov9e-seg.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8s.pt")  # 加载预训练模型（推荐用于训练）
    model = YOLO('ultralytics/cfg/models/v8/yolov8-seg.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重

    # Use the model
    results = model.train(data="lunwen.yaml", epochs=300, batch=16, workers=8, close_mosaic=50, name='Mobilefour')  # 训练模型
    #results = model.train(model=,#传入的model.yaml文件或者model.pt文件，用于构建网络和初始化，不同点在于只传入yaml文件的话参数会随机初始化
                          #data=,#训练数据集的配置yaml文件
                          #epochs=,#训练轮次，默认100
                          #patience=,#早停训练观察的轮次，默认50，如果50轮没有精度提升，模型会直接停止训练
                          #batch=,#训练批次，默认16
                          #imgsz=,#训练图片大小，默认640
                          #save=,#保存训练过程和训练权重，默认开启
                          #save_period=,#训练过程中每x个轮次保存一次训练模型，默认-1（不开启）
                          #cache=,#是否采用ram进行数据载入，设置True会加快训练速度，但是这个参数非常吃内存，一般服务器才会设置
                          #device=,#要运行的设备，即cuda device =0或Device =0,1,2,3或device = cpu
                          #workers=,#载入数据的线程数。windows一般为4，服务器可以大点，windows上这个参数可能会导致线程报错，发现有关线程报错，可以尝试减少这个参数，这个参数默认为8，大部分都是需要减少的
                          #project=,#项目文件夹的名，默认为runs
                          #name=,#用于保存训练文件夹名，默认exp，依次累加
                          #exist_ok=,#是否覆盖现有保存文件夹，默认Flase
                          #pretrained=,#是否加载预训练权重，默认Flase
                          #optimizer=,#优化器选择，默认SGD，可选[SGD、Adam、AdamW、RMSProP]
                          #verbose=,#是否打印详细输出
                          #seed=,#随机种子，用于复现模型，默认0
                          #deteministic=,#设置为True，保证实验的可复现性
                          #single_cls=,#将多类数据训练为单类，把所有数据当作单类训练，默认Flase
                          #image_weights=,#使用加权图像选择进行训练，默认Flase
                          #rect=,#使用矩形训练，和矩形推理同理，默认False

                         #cos_lr=,#使用余弦学习率调度，默认Flase
                          #close_mosaic=,#最后x个轮次禁用马赛克增强，默认10
                          #resume=,#断点训练，默认Flase
                          #lr0=,#初始化学习率，默认0.01
                          #lrf=,#最终学习率，默认0.01
                          #label_smoothing=,#标签平滑参数，默认0.0
                          #dropout=,#使用dropout正则化(仅对训练进行分类)，默认0.0
                          #)  # 常用参数
    # results = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
#task="seg",