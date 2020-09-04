# VR-centred-I-DT-algorithm

In *I-DT_alg_VR_centred.py* can be found a Python code that defines the I-DT algorithm for a VR centred system. The input of the algorithm is a Dataframe with seven columns: One is the time; three are the position of the gaze; three are the position of the subject's head. The algorithm returns the same Dataframe with an additional binary column named "class_disp". If it is a zero the frame belongs to a fixation; if it is a one the frame belongs to a saccade. 

This algorithm is part of a publication.

https://www.mdpi.com/1424-8220/20/17/4956

If you find this code useful for your research please cite:

Llanes-Jurado, J.; Marín-Morales, J.; Guixeres, J.; Alcañiz, M. Development and Calibration of an Eye-Tracking Fixation Identification Algorithm for Immersive Virtual Reality. Sensors 2020, 20, 4956. 
