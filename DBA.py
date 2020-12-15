'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import reduce


__author__ ="Francois Petitjean"

def performDBA(series, n_iterations=10):
    n_series = len(series)
    max_length = reduce(max, map(len, series))

    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)

    medoid_ind = approximate_medoid_index(series,cost_mat,delta_mat)
    center = series[medoid_ind]
    # x = np.arange(1, 97)
    # color_arr = cm.rainbow(np.linspace(0, 1, n_iterations))
    # colors = iter(color_arr)

    # plot_or_not = False
    # if (plot_or_not):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(121)
    #     ax2 = fig.add_subplot(122)
    #     ax1.set_title('center Plot')
    #     ax2.set_title('variance Plot')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')

    for i in range(0,n_iterations):
        # cur_color = next(colors)
        center, dtw_horizontal_var, dtw_vertical_var, normal_vertical_var, a, b, c, d = DBA_update(center, series, cost_mat, path_mat, delta_mat)
        # if (plot_or_not):
            # ax1.plot(x, center, color=cur_color)
            # ax2.plot(x, dtw_horizontal_var, color=cur_color)
            # ax3.plot(x, dtw_vertical_var, colors=cur_color)
            # ax4.plot(x, normal_vertical_var, colors=cur_color)
    return center, dtw_horizontal_var, dtw_vertical_var, normal_vertical_var, a, b, c, d

def approximate_medoid_index(series,cost_mat,delta_mat):
    if len(series)<=50:
        indices = range(0,len(series))
    else:
        indices = np.random.choice(range(0,len(series)),50,replace=False)

    medoid_ind = -1
    best_ss = 1e20
    for index_candidate in indices:
        candidate = series[index_candidate]
        ss = sum_of_squares(candidate,series,cost_mat,delta_mat)
        if(medoid_ind==-1 or ss<best_ss):
            best_ss = ss
            medoid_ind = index_candidate
    return medoid_ind

def sum_of_squares(s,series,cost_mat,delta_mat):
    return sum(map(lambda t:squared_DTW(s,t,cost_mat,delta_mat),series))

def DTW(s,t,cost_mat,delta_mat):
    return np.sqrt(squared_DTW(s,t,cost_mat,delta_mat))

def squared_DTW(s,t,cost_mat,delta_mat):
    s_len = len(s)
    t_len = len(t)
    length = len(s)
    fill_delta_mat_dtw(s, t, delta_mat)
    cost_mat[0, 0] = delta_mat[0, 0]
    for i in range(1, s_len):
        cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]

    for j in range(1, t_len):
        cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]

    for i in range(1, s_len):
        for j in range(1, t_len):
            diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
            if(diag <=left):
                if(diag<=top):
                    res = diag
                else:
                    res = top
            else:
                if(left<=top):
                    res = left
                else:
                    res = top
            cost_mat[i, j] = res+delta_mat[i, j]
    return cost_mat[s_len-1,t_len-1]

def fill_delta_mat_dtw(center, s, delta_mat):
    slim = delta_mat[:len(center),:len(s)]
    np.subtract.outer(center, s,out=slim)
    np.square(slim, out=slim)

def DBA_update(center, series, cost_mat, path_mat, delta_mat):
    options_argmin = [(-1, -1), (0, -1), (-1, 0)]
    center_length = len(center)
    adjusted_series_mat = np.zeros((len(series), center_length))
    adjusted_series_weight_mat = np.zeros((len(series), center_length), dtype=int)            #be careful when defining the 2-dimensional matrix
    series_mapping_mat = np.zeros((len(series), center_length), dtype=int)
    current_series_idx = 0
    for s in series:
        s_len = len(s)
        fill_delta_mat_dtw(center, s, delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        path_mat[0, 0] = -1

        for i in range(1, center_length):
            cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
            path_mat[i, 0] = 2

        for j in range(1, s_len):
            cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
            path_mat[0, j] = 1

        for i in range(1, center_length):
            for j in range(1, s_len):
                diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
                if(diag <=left):
                    if(diag<=top):
                        res = diag
                        path_mat[i,j] = 0
                    else:
                        res = top
                        path_mat[i,j] = 2
                else:
                    if(left<=top):
                        res = left
                        path_mat[i,j] = 1
                    else:
                        res = top
                        path_mat[i,j] = 2

                cost_mat[i, j] = res+delta_mat[i, j]

        i = center_length-1
        j = s_len-1

        while(path_mat[i, j] != -1):
            if (adjusted_series_weight_mat[current_series_idx, i] == 0):
                adjusted_series_mat[current_series_idx, i] = s[j]
                series_mapping_mat[current_series_idx, i] = j
            adjusted_series_weight_mat[current_series_idx, i] += 1
            move = options_argmin[path_mat[i, j]]
            i += move[0]
            j += move[1]
        assert(i == 0 and j == 0)
        if (adjusted_series_weight_mat[current_series_idx, i] == 0):
            adjusted_series_mat[current_series_idx, i] = s[j]
            series_mapping_mat[current_series_idx, i] = j
        adjusted_series_weight_mat[current_series_idx, i] += 1
        current_series_idx += 1

    # adjust_series_mat          映射的value
    # adjusted_series_weight_mat 往前映射的距离
    # series_mapping_mat         当前映射的index

    updated_weight = np.sum(adjusted_series_weight_mat, 0)
    updated_center = np.divide(np.sum(adjusted_series_mat * adjusted_series_weight_mat, 0), updated_weight)
    # dtw_vertical_variance = np.sqrt(np.divide(np.sum(np.power(adjusted_series_mat - updated_center, 2) * adjusted_series_weight_mat, 0), updated_weight))
    dtw_vertical_variance = calculateVerticalVariance(series_mapping_mat, adjusted_series_weight_mat, updated_weight, updated_center, series)
    dtw_horizontal_variance = calculateHorizontalVariance(adjusted_series_weight_mat, series_mapping_mat, updated_center, updated_weight)
    normal_vertical_variance = np.sqrt(np.divide(np.sum(np.power(series - updated_center, 2), 0), len(series)))
    dtw_special_vertical_variance = calculateVerticalVariance(series_mapping_mat, adjusted_series_weight_mat, updated_weight, updated_center, series, True)
    return updated_center, dtw_horizontal_variance, dtw_vertical_variance, normal_vertical_variance, adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat, dtw_special_vertical_variance

def calculateVerticalVariance(series_mapping_mat, adjusted_series_weight_mat, updated_weight, updated_center, series, isSpecial=False):
    res = np.zeros(series_mapping_mat.shape)
    for i in np.arange(0, res.shape[0]):
        for j in np.arange(0, res.shape[1]):
            normalVariance = calSingleVerticalVariance(i, j, series_mapping_mat, adjusted_series_weight_mat, updated_center, series)
            if (isSpecial):
                res[i, j] = np.divide(normalVariance, adjusted_series_weight_mat[i][j])
            else:
                res[i, j] = normalVariance
    if (isSpecial):
        return np.sqrt(np.divide(np.sum(res, 0), res.shape[0]))
    else:
        return np.sqrt(np.divide(np.sum(res, 0), updated_weight))

def calculateHorizontalVariance(adjusted_series_weight_mat, series_mapping_mat, updated_center, updated_weight):
    delta_mat = series_mapping_mat - np.arange(0, series_mapping_mat.shape[1])
    addup_variance = np.frompyfunc(calSingleHorizontalVariance, 2, 1)
    # delta_square_mat = addup_variance(delta_mat, adjusted_series_weight_mat)
    delta_square_mat = getHorizontalDeltaMat(delta_mat, adjusted_series_weight_mat)
    deviation = np.divide(np.sum(delta_square_mat, 0), updated_weight)
    return np.sqrt(deviation)


def getHorizontalDeltaMat(delta_mat, adjusted_series_weight_mat):
    res = np.zeros(delta_mat.shape)
    for i in np.arange(0, res.shape[0]):
        for j in np.arange(0, res.shape[1]):
            res[i, j] = calSingleHorizontalVariance(delta_mat[i, j], adjusted_series_weight_mat[i, j])
    return res


def calSingleVerticalVariance(series_index, index_of_center, series_mapping_mat, adjusted_series_weight_mat, updated_center, series):
    begin, end = getStartAndEndMapping(series_mapping_mat, adjusted_series_weight_mat, series_index, index_of_center)
    differnce = series[series_index][begin: end] - updated_center[index_of_center]
    res = np.sum(np.power(differnce, 2))
    return res

def getStartAndEndMapping(series_mapping_mat, adjusted_series_weight_mat, series_index, index_of_center) :
    begin_mapping_j = series_mapping_mat[series_index, index_of_center] + 1
    end_mapping_j = begin_mapping_j - adjusted_series_weight_mat[series_index, index_of_center]
    return end_mapping_j, begin_mapping_j

def calSingleHorizontalVariance(j_start_delta, count):
    return np.sum(np.power(np.ones(count, dtype=int) * j_start_delta - np.arange(0, count), 2))


def main():
    #generating synthetic data
    n_series = 20
    length = 200

    series = list()
    padding_length=30
    indices = range(0, length-padding_length)
    main_profile_gen = np.array([np.sin(2*np.pi*j/len(indices)) for j in indices])
    randomizer = lambda j:np.random.normal(j,0.02)
    randomizer_fun = np.vectorize(randomizer)
    for i in range(0,n_series):
        n_pad_left = np.random.randint(0,padding_length)
        #adding zero at the start or at the end to shif the profile
        series_i = np.pad(main_profile_gen,(n_pad_left,padding_length-n_pad_left),mode='constant',constant_values=0)
        #chop some of the end to prove it can work with multiple lengths
        l = np.random.randint(length-20,length+1)
        series_i = series_i[:l]
        #randomize a bit
        series_i = randomizer_fun(series_i)

        series.append(series_i)
    series = np.array(series)

    #plotting the synthetic data
    for s in series:
        plt.plot(range(0,len(s)), s)
    plt.draw()

    #calculating average series with DBA
    average_series = performDBA(series)

    #plotting the average series
    plt.figure()
    plt.plot(range(0,len(average_series)), average_series)
    plt.show()


def main2():
    adjusted_series_weight_mat = [[1,2], [3,4]]
    series_mapping_mat = [[7,9], [19,24]]
    res = calSingleHorizontalVariance(adjusted_series_weight_mat, series_mapping_mat)
    # res = out(adjusted_series_weight_mat, series_mapping_mat)
    res

if __name__ == '__main__':
    main2()