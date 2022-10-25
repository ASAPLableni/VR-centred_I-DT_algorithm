# ###########################################
# ###########################################
# ### I-DT algorithm in VR-centred system ###
# ###########################################
# ###########################################

import numpy as np
import numba as nb
import sys


class IDTVR:
    def __init__(self,
                 time_th=0.25, disp_th=1, freq_th=30, numba_allow=True):
        self.time_th = time_th
        self.disp_th = disp_th
        self.numba_allow = numba_allow
        self.freq_th = freq_th

        self.class_disp = None

    # ### This function is the implementation of the I-DT algorithm in VR-centred system.
    # ### The used thresholds are 0.25s for the time window and 1 degree for the dispersion threshold.
    # ### The input of the function time selects the column with the time variable.
    # ### The inputs et_(x,y,z) are coordinates of the gaze in x,y,z.
    # ### The inputs head_pos_(x,y,x) are the coordinates for the head position in x,y,z.

    def fit_compute(self,
                    data,
                    time="time",
                    et_x="et_x", et_y="et_y", et_z="et_z",
                    head_pos_x="head_pose_x", head_pos_y="head_pose_y", head_pos_z="head_pose_z",
                    debug=False):
        data = data.reset_index(drop=True)
        data["class_disp"] = ["?"] * data.shape[0]
        initial_idx = data.index.values[0]
        final_idx = data.index.values[-1]

        while True:
            try:
                data.iloc[initial_idx]
            except:
                break

            init_time = data[time].iloc[initial_idx]
            fin_time = self.time_th + init_time
            end_idx = np.argsort(np.abs(data[time].values - fin_time))[0]

            if end_idx == initial_idx and end_idx != final_idx:
                end_idx += 1

            j = end_idx
            if debug:
                IDTVR.my_progressbar_show(j - 1, final_idx)

            sub_data = data.iloc[initial_idx:(end_idx + 1)]

            freq_list = [1 / (sub_data[time].iloc[i] - sub_data[time].iloc[i - 1]) for i in range(1, sub_data.shape[0])]

            if np.sum(np.array(freq_list) < self.freq_th) == 0:

                head_mean_pos_x = np.nanmean(sub_data[head_pos_x])
                head_mean_pos_y = np.nanmean(sub_data[head_pos_y])
                head_mean_pos_z = np.nanmean(sub_data[head_pos_z])

                output = list(map(self.compute_disp_angle,
                                  zip([[head_mean_pos_z] * sub_data.shape[0]],
                                      [[head_mean_pos_x] * sub_data.shape[0]],
                                      [[head_mean_pos_y] * sub_data.shape[0]],
                                      [sub_data[et_z].values],
                                      [sub_data[et_y].values],
                                      [sub_data[et_x].values])))

                list_thetas, msg = output[0]

                if len(list_thetas) != 0 and msg != "found":
                    while True:
                        if final_idx != end_idx:
                            end_idx += 1
                            j = end_idx

                        if debug:
                            IDTVR.my_progressbar_show(j - 1, final_idx)

                        sub_data = data.iloc[initial_idx:(end_idx + 1)]

                        head_mean_pos_x = np.nanmean(sub_data[head_pos_x])
                        head_mean_pos_y = np.nanmean(sub_data[head_pos_y])
                        head_mean_pos_z = np.nanmean(sub_data[head_pos_z])

                        t_diff_array = sub_data[time].iloc[1:].values - sub_data[time].iloc[:-1].values
                        freq_list = 1 / t_diff_array

                        if np.sum(freq_list < self.freq_th) == 0:
                            output = list(map(self.compute_disp_angle,
                                              zip([[head_mean_pos_z] * sub_data.shape[0]],
                                                  [[head_mean_pos_x] * sub_data.shape[0]],
                                                  [[head_mean_pos_y] * sub_data.shape[0]],
                                                  [sub_data[et_z].values],
                                                  [sub_data[et_y].values],
                                                  [sub_data[et_x].values])))

                            list_thetas, msg = output[0]
                            if msg == "found" or j >= final_idx:
                                data["class_disp"].iloc[initial_idx:end_idx] = 0
                                data["class_disp"].iloc[end_idx] = 1

                                initial_idx = end_idx + 1

                                break
                        else:
                            data["class_disp"].iloc[initial_idx:end_idx] = 0
                            data["class_disp"].iloc[end_idx] = 1

                            initial_idx = end_idx + 1

                            break
                else:
                    data["class_disp"].iloc[initial_idx] = 1
                    initial_idx += 1
            else:
                data["class_disp"].iloc[initial_idx] = 1
                initial_idx += 1

        self.class_disp = data["class_disp"]

        return data

    @staticmethod
    @nb.jit(nopython=True)
    def get_result_numba(all_x, all_y, all_z, disp_th):
        label_break = 0.0
        result_list = []
        for i in range(all_x.shape[0] - 1, 1 - 1, -1):
            for j in range(0, i):
                num = all_x[i] * all_x[j] + all_y[i] * all_y[j] + all_z[i] * all_z[j]
                den1 = np.sqrt(all_x[i] ** 2 + all_y[i] ** 2 + all_z[i] ** 2)
                den2 = np.sqrt(all_x[j] ** 2 + all_y[j] ** 2 + all_z[j] ** 2)

                result = np.abs(num) / (den1 * den2)
                result_degrees = np.arccos(result) * (180 / np.pi)
                result_list.append(result_degrees)

            if disp_th < np.nanmax(result_list):
                label_break = 1
                break

        msg = "found" if label_break > 0 else "not_found"

        return result_list, msg

    def get_result_normal(self, all_x, all_y, all_z):
        msg = "not_found"

        iteration = np.arange(1, all_x.shape[0], 1)[::-1]
        result_list = []
        for i in iteration:
            dist_x_i, dist_y_i, dist_z_i = all_x[i], all_y[i], all_z[i]

            diagonal = list(map(
                lambda j: IDTVR.scalar_product(dist_x_i, all_x[j], dist_y_i, all_y[j], dist_z_i, all_z[j]),
                np.arange(i)
            ))

            result_list += list(np.arccos(np.abs(diagonal)) * (180 / np.pi))

            if self.disp_th < np.nanmax(result_list):
                msg = "found"
                break

        return result_list, msg

    def compute_disp_angle(self, zip_obj):
        mean_vertex_x, mean_vertex_y, mean_vertex_z, et_x_col, et_y_col, et_z_col = zip_obj

        mean_vertex_x = mean_vertex_x[0]
        mean_vertex_y = mean_vertex_y[0]
        mean_vertex_z = mean_vertex_z[0]

        all_x = np.array(et_x_col) - mean_vertex_x
        all_y = np.array(et_y_col) - mean_vertex_y
        all_z = np.array(et_z_col) - mean_vertex_z

        if self.numba_allow:
            result_list, msg = self.get_result_numba(all_x, all_y, all_z, disp_th=self.disp_th)
        else:
            result_list, msg = self.get_result_normal(all_x, all_y, all_z)

        return result_list, msg

    # ### Definition of the normalized scalar product in three dimensions.
    @staticmethod
    def scalar_product(x1, x2, y1, y2, z1, z2):
        num = x1 * x2 + y1 * y2 + z1 * z2
        den1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
        den2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

        return np.abs(num) / (den1 * den2)

    # ### Definition of the progressbar to check that the algorithm is working.
    @staticmethod
    def my_progressbar_show(j, count, prefix="", size=80, file=sys.stdout):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()
