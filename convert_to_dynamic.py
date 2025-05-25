import onnx

model = onnx.load("/models/vgg16-7.onnx")

# 입력 텐서의 shape 수정: [1, 3, 224, 224] -> [None, 3, 224, 224]
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

onnx.save(model, "/models/vgg16-7_dynamic.onnx")