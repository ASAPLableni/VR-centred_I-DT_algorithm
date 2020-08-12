###########################################
###########################################
### I-DT algorithm in VR-centred system ###
###########################################
###########################################

import numpy as np
import pandas as pd
import sys

### Definition of the normalized scalar product in three dimensions.
def scalar_product(x1, x2, y1, y2, z1, z2):

	num = x1*x2+y1*y2+z1*z2
	den1 = np.sqrt(x1**2+y1**2+z1**2)
	den2 = np.sqrt(x2**2+y2**2+z2**2)
    
	return np.abs(num)/(den1*den2)

### Definition of the progressbar to check that the algorithm is working.
def my_progressbar_show(j, count, prefix="", size=80, file=sys.stdout):
	x = int(size*j/count)
	file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
	file.flush()

def compute_disp_angle(zip_obj):

	mean_vertex_x, mean_vertex_y, mean_vertex_z, et_x_col, et_y_col, et_z_col, th = zip_obj
    
	mean_vertex_x = mean_vertex_x[0] ; mean_vertex_y = mean_vertex_y[0] 
	mean_vertex_z = mean_vertex_z[0]
	th = th[0]
    
	mssg = "not_found"
	result_list = list()
    
	all_x = np.array(et_x_col)-mean_vertex_x
	all_y = np.array(et_y_col)-mean_vertex_y
	all_z = np.array(et_z_col)-mean_vertex_z
    
	iteration = np.arange(1, all_x.shape[0], 1) 
	iteration = iteration[::-1]
	for i in iteration:
		dist_x_i = all_x[i]
		dist_y_i = all_y[i]
		dist_z_i = all_z[i]
        
		diag = list(map(lambda j:scalar_product(dist_x_i, all_x[j], dist_y_i, all_y[j], dist_z_i, all_z[j]), 
                        np.arange(i)))
                     
		result_list += list( np.arccos(np.abs(diag))*(180/np.pi) )
        
		if th != False and th < np.nanmax(result_list):
			mssg = "found"
			break
            
		return result_list, mssg


### This function is the implementation of the I-DT algorithm in VR-centred system. 
### The used thresholds are 0.25s for the time window and 1 degree for the dispersion threshold.
### The input of the function time selects the column with the time variable.
### The inputs et_(x,y,z) are coordinates of the gaze in x,y,z.
### The inputs head_pos_(x,y,x) are the coordinates for the head position in x,y,z.

def IDT_VR(data, 
	   time_th = 0.25, disp_th=1, 
	   time = "time", 
	   et_x="et_x", et_y="et_y", et_z="et_z", 
	   head_pos_x="head_pos_x", head_pos_y="head_pos_y", head_pos_z="head_pos_z",
	   frec_th=30, debug=False):
    
	data = data.reset_index(drop=True)
	data["class_disp"] = ["?"]*data.shape[0]
	initial_idx = data.index.values[0]
	final_idx = data.index.values[-1]
    
	while True:
		try:
			data.iloc[initial_idx]
		except:
			break
            
		init_time = data[time].iloc[initial_idx]
		fin_time = time_th+init_time
		end_idx = np.argsort( np.abs(data[time].values-fin_time) )[0]
            
		if end_idx == initial_idx and end_idx != final_idx:
			end_idx += 1
        
		j = end_idx
		if debug:
			my_progressbar_show(j-1, final_idx)
        
		sub_data = data.iloc[initial_idx:(end_idx+1)]
        
		frec_list = list()
		for i in range(1, sub_data.shape[0]):
			t_int = sub_data[time].iloc[i]-sub_data[time].iloc[i-1]
			frec_list.append(1/t_int)
        
		if np.sum(np.array(frec_list) < frec_th) == 0:
            
			head_mean_pos_x = np.nanmean(sub_data[head_pos_x])
			head_mean_pos_y = np.nanmean(sub_data[head_pos_y])
			head_mean_pos_z = np.nanmean(sub_data[head_pos_z])

			output = list(map(compute_disp_angle, 
                                      zip([[head_mean_pos_z]*sub_data.shape[0]],
                                          [[head_mean_pos_x]*sub_data.shape[0]], 
                                          [[head_mean_pos_y]*sub_data.shape[0]],
                                          [sub_data[et_z].values], 
                                          [sub_data[et_y].values], 
                                          [sub_data[et_x].values], 
                                          [[disp_th]*sub_data.shape[0]])) )

			list_thetas, mssg = output[0]

			if len(list_thetas) != 0 and mssg != "found":
				while True:
					if final_idx != end_idx:
						end_idx += 1
						j = end_idx

					if debug:    
						my_progressbar_show(j-1, final_idx)

					sub_data = data.iloc[initial_idx:(end_idx+1)]
					head_mean_pos_x = np.nanmean(sub_data[head_pos_x])
					head_mean_pos_y = np.nanmean(sub_data[head_pos_y])
					head_mean_pos_z = np.nanmean(sub_data[head_pos_z])
                    
					frec_list = list()
					frec_list = sub_data[time].iloc[1:]-sub_data[time].iloc[:-1]
					frec_list = frec_list.values
					frec_list = 1/frec_list
                    
					if np.sum(frec_list < frec_th) == 0:
						output = list(map(compute_disp_angle, 
                                          zip([[head_mean_pos_z]*sub_data.shape[0]],
                                              [[head_mean_pos_x]*sub_data.shape[0]], 
                                              [[head_mean_pos_y]*sub_data.shape[0]],
                                              [sub_data[et_z].values], 
                                              [sub_data[et_y].values], 
                                              [sub_data[et_x].values], 
                                              [[disp_th]*sub_data.shape[0]])) )

						list_thetas, mssg = output[0]
						if mssg == "found" or j >= final_idx:  

							data["class_disp"].iloc[initial_idx:end_idx] = 0
							data["class_disp"].iloc[end_idx] = 1

							initial_idx = end_idx+1

							break
					else:
						data["class_disp"].iloc[initial_idx:end_idx] = 0
						data["class_disp"].iloc[end_idx] = 1

						initial_idx = end_idx+1

						break
			else:
				data["class_disp"].iloc[initial_idx] = 1
				initial_idx += 1
		else:
			data["class_disp"].iloc[initial_idx] = 1
			initial_idx += 1
    
	return data
