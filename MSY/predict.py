from ultralytics import YOLO

model = YOLO(r'runs/segment/Mobilefour2/weights/best.pt')
preds = model.predict(source=r'datasets/looklook', save_txt=True, save=True, show_boxes=False, name='forpaper')
