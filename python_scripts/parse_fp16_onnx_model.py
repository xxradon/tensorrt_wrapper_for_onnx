import onnx
import json
import os
import numpy as np

node_func = {}
onnx_data_type = {}
onnx_data_type["DEFAULT"] = 0
onnx_data_type["FLOAT"] = 1
onnx_data_type["UINT8"] = 2
onnx_data_type["INT8"] = 3
onnx_data_type["UINT16"] = 4
onnx_data_type["INT16"] = 5
onnx_data_type["INT64"] = 6
onnx_data_type["INT64"] = 7
onnx_data_type["STRING"] = 8
onnx_data_type["BOOL"] = 9
onnx_data_type["FLOAT16"] = 10
onnx_data_type["DOUBLE"] = 11
onnx_data_type["UINT32"] = 12
onnx_data_type["UINT64"] = 13
onnx_data_type["COMPLEX64"] = 14
onnx_data_type["COMPLEX128"] = 15
onnx_data_type["BFLOAT16"] = 16

def get_node_attribute(node_attr, attributes):
    for i in range(len(node_attr)):
        if node_attr[i].type == 2:
            attributes[node_attr[i].name] = [node_attr[i].i]
        elif node_attr[i].type == 7:
            attributes[node_attr[i].name] = list(node_attr[i].ints)
        else:
            pass

# conv op 
def conv_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    # for i in range(len(node_attr)):
    #     attributes[node_attr[i].name] = list(node_attr[i].ints)
    get_node_attribute(node_attr, attributes)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    node["weights_offset"] = weights_info[inputs[1]]["offset"]
    node["weights_count"] = weights_info[inputs[1]]["count"]
    node["weights_data_type"] = weights_info[inputs[1]]["data_type"]
    node["bias_offset"]    = weights_info[inputs[2]]["offset"]
    node["bias_count"] = weights_info[inputs[2]]["count"]
    node["bias_data_type"] = weights_info[inputs[2]]["data_type"]
    return node
node_func["Conv"] = conv_node_func


#clip op
def clip_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    for i in range(len(inputs)):
        if i == 0:
            continue
        ele = weights_info[inputs[i]]
        if ele["data_type"] == onnx_data_type["FLOAT16"]:
            fp32_np_data = np.frombuffer(ele["raw_data"], dtype=np.float16).astype(np.float32)
            ele["raw_data"] = fp32_np_data.tobytes()
            ele["data_type"] = onnx_data_type["FLOAT"]
            ele["count"] = 2 * ele["count"]
            # print("run here")
    return node
node_func["Clip"] = clip_node_func

#Add op
def add_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node    
node_func["Add"] = add_node_func

#Sub op
def sub_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node    
node_func["Sub"] = sub_node_func

#Mul op
def mul_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node    
node_func["Mul"] = mul_node_func

#Div op
def div_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node
node_func["Div"] = div_node_func

#Sqrt op
def sqrt_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node    
node_func["Sqrt"] = sqrt_node_func

#Reciprocal op
def reciprocal_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node    
node_func["Reciprocal"] = reciprocal_node_func


#Reshape op
def reshape_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["shape_data_type"] = weights_info[inputs[1]]["data_type"]
    return node    
node_func["Reshape"] = reshape_node_func

#Softmax op
def softmax_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    # for i in range(len(node_attr)):
    #     attributes[node_attr[i].name] = list(node_attr[i].ints)
    get_node_attribute(node_attr, attributes)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node
node_func["Softmax"] = softmax_node_func

#ReduceSum op
def reducesum_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    # for i in range(len(node_attr)):
    #     attributes[node_attr[i].name] = list(node_attr[i].ints)
    get_node_attribute(node_attr, attributes)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node
node_func["ReduceSum"] = reducesum_node_func

#Transpose op
def transpose_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    for i in range(len(node_attr)):
        attributes[node_attr[i].name] = list(node_attr[i].ints)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node
node_func["Transpose"] = transpose_node_func

#Max op
def max_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    for i in range(len(node_attr)):
        attributes[node_attr[i].name] = list(node_attr[i].ints)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    for i in inputs:
        if i in weights_info.keys():
            node["data_type"] = weights_info[i]["data_type"]
            node["offset"] = weights_info[i]["offset"]
            node["count"]  = weights_info[i]["count"]
    return node
node_func["Max"] = max_node_func

#MaxPool op
def maxpool_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    for i in range(len(node_attr)):
        attributes[node_attr[i].name] = list(node_attr[i].ints)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node
node_func["MaxPool"] = maxpool_node_func

#Cast op
def cast_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    get_node_attribute(node_attr, attributes)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node
node_func["Cast"] = cast_node_func

#Abs op
def abs_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node
node_func["Abs"] = abs_node_func

#Pad op
def pad_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["padding_data_type"] = weights_info[inputs[1]]["data_type"]
    node["padding_offset"] = weights_info[inputs[1]]["offset"]
    node["padding_count"] = weights_info[inputs[1]]["count"]
    return node
node_func["Pad"] = pad_node_func

#Greater op
def greater_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    for i in range(len(node_attr)):
        attributes[node_attr[i].name] = list(node_attr[i].ints)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    for i in inputs:
        if i in weights_info.keys():
            node["data_type"] = weights_info[i]["data_type"]
            node["offset"] = weights_info[i]["offset"]
            node["count"]  = weights_info[i]["count"]
    return node
node_func["Greater"] = greater_node_func

#Equal op
def equal_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node
node_func["Equal"] = equal_node_func

#NonZero op
def nonzero_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node
node_func["NonZero"] = nonzero_node_func

#Slice op
def slice_node_func(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["starts_data_type"] = weights_info[inputs[1]]["data_type"]
    node["starts_offset"] = weights_info[inputs[1]]["offset"]
    node["starts_count"] = weights_info[inputs[1]]["offset"]
    node["ends_data_type"] = weights_info[inputs[2]]["data_type"]
    node["ends_offset"] = weights_info[inputs[2]]["offset"]
    node["ends_count"] = weights_info[inputs[2]]["offset"]
    node["axes_data_type"] = weights_info[inputs[3]]["data_type"]
    node["axes_offset"] = weights_info[inputs[3]]["offset"]
    node["axes_count"] = weights_info[inputs[3]]["offset"]
    return node
node_func["Slice"] = slice_node_func

def get_node_info(node_type, node_info, weights_info):
    if node_type in node_func.keys():
        return node_func[node_type](node_info, weights_info)
    else:
        print("not support {} op for now!!!!".format(node_type))
        return ""

def parse_onnx_graph(graph, weights_info):
    node  = graph.node
    node_info_map = {}
    net_json_graph = {}
    for i in range(len(node)):
        node_info = get_node_info(node[i].op_type, node[i], weights_info)
        if node_info != "":
            node_info_map[node[i].name] = node_info
    net_json_graph["nodes_info"] = node_info_map
    # net_json_graph["weights_info"] = weights_info
    return net_json_graph

def save_simplify_graph(simply_graph, name):
    json_str = json.dumps(simply_graph)
    with open(name + '.json', 'w') as f:
        f.write(json_str)

def get_graph_weights(graph):
    weights = graph.initializer
    output_tensor = list(graph.output)
    output_tensor = [ x.name for x in output_tensor ]
    input_tensor  = list(graph.input)
    fp16_flag = 0
    offset = 0
    weights_info = {}
    for i in range(len(input_tensor)):
        temp = {}
        temp["count"] = 0
        temp["offset"] = -1
        temp["data_type"] = input_tensor[0].type.tensor_type.elem_type
        dims = input_tensor[i].type.tensor_type.shape.dim
        input_shape = [dims[i].dim_value for i in range(len(dims))]
        temp["tensor_shape"] = input_shape
        weights_info[input_tensor[i].name] = temp
        if temp["data_type"] == onnx_data_type["FLOAT16"]:
            fp16_flag = 1

    for ele in weights:
        temp = {}
        temp["count"] = len(ele.raw_data)
        temp["offset"] = offset
        temp["data_type"] = ele.data_type
        shape = list(ele.dims)
        if shape == []:
            shape = [1]
        temp["tensor_shape"] = shape
        temp["raw_data"] = ele.raw_data
        offset = offset + len(ele.raw_data)
        weights_info[ele.name] = temp

    weights_info["net_output"] = output_tensor
    return weights_info, fp16_flag


def update_weights_offset_and_save(weights_info, file_name):
    offset = 0
    with open(file_name + "_weights.bin", "wb") as f:
        for ele in weights_info:
            if "offset" in weights_info[ele] and weights_info[ele]["offset"] != -1:
                weights_info[ele]["offset"] = offset
                offset += weights_info[ele]["count"]
                f.write(weights_info[ele]["raw_data"])
                del weights_info[ele]["raw_data"]

def topo_lazy_search(visited_tensors, nodes_info, topo_node_order):
    for ele in nodes_info.keys():
        flag = True
        node_info = nodes_info[ele]
        for input_tensor in node_info["inputs"]:
            if input_tensor not in visited_tensors:
                flag = False
        if flag == True and (ele not in topo_node_order):
            topo_node_order.append(ele)
            visited_tensors.extend(node_info["outputs"])



def generate_topo_order(nodes_info, weights_info):
    topo_node_order = []
    input_tensors = []
    visited_tensors = []
    for tensor_name in weights_info.keys():
        if isinstance(weights_info[tensor_name],dict):
            if weights_info[tensor_name]["offset"] == -1:
                input_tensors.append(tensor_name)
            visited_tensors.append(tensor_name)
    
    while len(topo_node_order) < len(nodes_info):
        topo_lazy_search(visited_tensors, nodes_info, topo_node_order)
    return topo_node_order
    
def find_depend_node(graph, out_tensor):
    for ele in graph.keys():
        if ele == "weights_info":
            continue
        node_info = graph[ele]
        if out_tensor in node_info["outputs"]:
            return ele
    return None

def generate_depend_nodes(topo_order, simply_graph):
    topo_order.reverse()
    depend_nodes = {}
    for elem in topo_order:
        input_tensors = simply_graph[elem]["inputs"]
        node = []
        for index in range(len(input_tensors)):
            temp_node = find_depend_node(simply_graph, input_tensors[index])
            if (temp_node not in node) and (temp_node != None):
                node.append(temp_node)
        
        depend_nodes[elem] = node

    topo_order.reverse()
    return depend_nodes



if __name__ == "__main__":
    #1 load onnx model
    onnx_model = onnx.load("hfnet_github_desc_fp16.onnx")
    #2 check onnx model
    # onnx.checker.check_model(onnx_model)
    #3 parse onnx graph
    graph = onnx_model.graph
    weights_info, fp16_flag = get_graph_weights(graph)
    simply_graph = parse_onnx_graph(graph, weights_info)
    #4 update weights info offset and save to file
    update_weights_offset_and_save(weights_info, "hfnet_github")
    #5 generate topology order from simply graph
    topo_order = generate_topo_order(simply_graph["nodes_info"], weights_info)
    #6 generate depend nodes from topo order
    # depend_nodes = generate_depend_nodes(topo_order, simply_graph["nodes_info"])

    
    #7 save weights and simplify graph to files used for tensorrt
    simply_graph["topo_order"] = topo_order
    simply_graph["weights_info"] = weights_info
    simply_graph["fp16_flag"] = fp16_flag
    save_simplify_graph(simply_graph, "hfnet_github_graph")

    print("convert success!!!")