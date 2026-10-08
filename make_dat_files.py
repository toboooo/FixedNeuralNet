import os
from array import array
import torch

for file in os.listdir("pieces"):
	if not file.endswith(".pth"):
		continue
	piece = torch.load(os.path.join("pieces", file),
		map_location=torch.device("cpu"))
	
	weights = piece["weight"].flatten()
	weights = [value.item() for value in weights]
	weights_size = [s for s in piece["weight"].size()]
	bias = [value.item() for value in piece["bias"]]
	bias_size = [s for s in piece["bias"].size()]
	print(file, weights_size, bias_size)

	weight_file_name = file.replace(".pth", "_w.dat")
	weights_size_array = array('i', weights_size)
	weight_array = array('d', weights)
	weight_file = open(weight_file_name, "wb")
	weights_size_array.tofile(weight_file)
	weight_array.tofile(weight_file)
	weight_file.close()
	
	bias_file_name = file.replace(".pth", "_b.dat")
	bias_size_array = array('i', bias_size)
	bias_array = array('d', bias)
	bias_file = open(bias_file_name, "wb")
	bias_size_array.tofile(bias_file)
	bias_array.tofile(bias_file)
	bias_file.close()
