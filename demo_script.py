
"""Script for running inference on NPU (tflite models)"""

import time
import numpy as np
import tflite_runtime.interpreter as tflite_rt
import pickle
import os

# @function for single sample based inference
# interpreter -> laoded model as interpreter
# sample -> input sample from test dataset
# prdctd_label -> output, shape is as per model
# inference_time_ms -> single sample inference time
def run_inference_using_interpreter(interpreter, sample):
	interpreter.allocate_tensors()
	interpreter.set_tensor(input_details[0]['index'], sample)
	start_time = time.time()
	interpreter.invoke()
	stop_time = time.time()
	model_out = interpreter.get_tensor(output_details[0]['index'])
	prdctd_label=np.array(model_out)
	inference_time_ms = (stop_time - start_time) * 1000
	return prdctd_label, inference_time_ms
	
########### Main program #############
tflite_model = "mnist_bilinear_demo_ptq.tflite"
images_labels = "mnist_samples_labels"	

print('\t\t\t\t\t---------------------------------------------- Loading demo dataset ---------------------------------------')
with open(images_labels + '.pkl', 'rb') as file: 
	samples = pickle.load(file)
	labels = pickle.load(file)
samples = np.array(samples)
labels = np.array(labels)

print('\t\t\t\t\t---------------------------------------------- Loading model with XNNPack for CPU ---------------------------------------')	
interpreter_cpu_xnn = tflite_rt.Interpreter(tflite_model)
interpreter_cpu_xnn.allocate_tensors()
input_details = interpreter_cpu_xnn.get_input_details()
output_details = interpreter_cpu_xnn.get_output_details()
print("\t\t\t\t\tmodel name = ", tflite_model, "input dtype = ",input_details[0]['dtype'], "output dtype = ",output_details[0]['dtype'])
print('\t\t\t\t\t---------------------------------------------- Starting CPU/XNNPack inference loop ---------------------------------------')
for loop_var in range(len(samples)):
	sample_for_inf = samples[loop_var].reshape(1,28,28,1)
	pred, time_ms = run_inference_using_interpreter(interpreter_cpu_xnn,sample_for_inf)
	pred= np.array(pred)
	print("predictions ->", pred, "\n\t\ttrue_label ->", labels[loop_var], "predicted_label ->", pred.argmax(), "inf. time ->", time_ms)

print('\t\t\t\t\t---------------------------------------------- Loading model with VX delegate for NPU ---------------------------------------')	
ext_delegate =  [ tflite_rt.load_delegate('/usr/lib/libvx_delegate.so') ]
interpreter_npu_vx = tflite_rt.Interpreter(tflite_model, experimental_delegates = ext_delegate)
interpreter_npu_vx.allocate_tensors()
input_details = interpreter_npu_vx.get_input_details()
output_details = interpreter_npu_vx.get_output_details()
print("\t\t\t\t\tmodel name = ", tflite_model, "input dtype = ",input_details[0]['dtype'], "output dtype = ",output_details[0]['dtype'])
print('\t\t\t\t\t---------------------------------------------- Starting NPU/VX inference loop ---------------------------------------')
for loop_var in range(len(samples)):
	sample_for_inf = samples[loop_var].reshape(1,28,28,1)
	pred, time_ms = run_inference_using_interpreter(interpreter_npu_vx,sample_for_inf)
	pred= np.array(pred)
	print("predictions ->", pred, "\n\t\ttrue_label ->", labels[loop_var], "predicted_label ->", pred.argmax(), "inf. time ->", time_ms)

