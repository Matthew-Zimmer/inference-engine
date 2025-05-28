import onnx_graphsurgeon as gs
import onnx, sys

output_name = sys.argv[1]

# Load graph
onnx_model = onnx.load("onnx/model.onnx")
graph = gs.import_onnx(onnx_model)
graph.outputs = [out for out in graph.outputs if out.name != output_name]
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "onnx/simplified_model.onnx")
