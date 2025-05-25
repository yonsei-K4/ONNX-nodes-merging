import onnxruntime as ort
import numpy as np
import cv2
import glob

# 1. ONNX 모델 로드
session = ort.InferenceSession("/models/vgg16-7_dynamic.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [None, 3, 224, 224]
print("Input name:", input_name)

# 2. 이미지 전처리 함수
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = img[:, :, ::-1]                 # BGR to RGB
    img = img.transpose(2, 0, 1)          # HWC to CHW
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std
    return img

# 3. 여러 이미지 로드 (예: jpg 파일들)
image_paths = glob.glob("images/*.jpg")  # 디렉터리 안의 이미지들
batch = np.stack([preprocess(p) for p in image_paths], axis=0).astype(np.float32)  # shape: [N, 3, 224, 224]
print("Batch shape:", batch.shape)

# 4. 추론
outputs = session.run(None, {input_name: batch})
output = outputs[0]  # shape: [N, 1000]

# 5. 결과 해석
top1_classes = np.argmax(output, axis=1)
for path, cls in zip(image_paths, top1_classes):
    print(f"{path} -> predicted class: {cls}")
