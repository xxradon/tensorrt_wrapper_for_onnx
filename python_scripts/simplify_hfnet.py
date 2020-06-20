import onnx
import onnxmltools
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('hfnet_github_global.onnx')

# convert model
model2 = onnx.shape_inference.infer_shapes(model)
onnxmltools.utils.save_model(model2, 'hfnet_github_inference.onnx')
# for i in model2.graph.value_info:
#     print(i) # Empty shape!
# model_simp, check = simplify(model)
# onnxmltools.utils.save_model(model_simp, 'hfnet_github_simplify.onnx')
# assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object