import onnx
import onnxruntime

ort_session = onnxruntime.InferenceSession('models/weights/Retinaface_m25_dynamic_batch.onnx')

def to_numpy(tensor):
    return tensor.deteach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(torch_out,))


model = onnx.load('/home/ngocnkd/project/Pytorch_Retinaface/models/weights/Retinaface_m25_dynamic_batch.onnx')
onnx.checker.check_model(model)