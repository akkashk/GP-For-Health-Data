from datetime import datetime
from math import sqrt
import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp

METRIC_MAP = {'sleep': 10, 'bedin': 11, 'bedout': 12}

def preprocess(metric_data, num_data_pts=50):
    # Remove high anomaly value
    mask1 = metric_data < 15
    mask2 = metric_data > 2
    xs_values = np.where(np.logical_and(mask1, mask2))[0]
    ys_values = metric_data[xs_values]

    # Don't return array if too few data points. For now we have 1 metric per day of the week (7) for 12 months, so 84 ideally
    if len(ys_values) < num_data_pts:
        # print("Too few data points:", len(ys_values))
        return np.array([]), np.array([])

    return xs_values, ys_values


def get_metric(user_data, metric):
    """
    metric: Choose between steps, hr, sleep
    """
    metric = METRIC_MAP[metric]
    table_metric = user_data[user_data['DATATYPE'] == metric].copy()

    # Assign an index according to date, compared to first date in data. We do this since we have missing dates
    date_conversion = lambda new_date: datetime.strptime(new_date, "%Y-%m-%d")
    table_metric['DATE'] = table_metric['DATE'].map(date_conversion)
    table_metric['MONTH'] = table_metric['DATE'].map(lambda x: x.month)
    table_metric['DAY'] = table_metric['DATE'].map(lambda x: x.weekday())

    # For each month group by date and take mean
    answer = np.array([])
    for month in range(1, 13):
        month_data = table_metric[table_metric['MONTH'] == month]
        gbo = month_data.groupby('DAY')
        answer = np.append(answer, gbo.median()['DATAVALUE'].values)

    return preprocess(answer)

t = pd.read_csv('/home/aks73/PML/data.csv', header=None, names=['Row', 'UID', 'DATE', 'DATATYPE', 'DATAVALUE'],
                usecols=['UID', 'DATE', 'DATATYPE', 'DATAVALUE'])
uid = set(t['UID'])

v = np.exp(1.5)
l_rbf = np.exp(1)
l_per = np.exp(1)
p = np.exp(2)
sigma = np.exp(0.01)

for user_id in uid:
    user_data = t[t['UID'] == user_id]
    xs, ys = get_metric(user_data, 'sleep')
    if len(ys) == 0:
        # Preprocessing can lead to empty arrays being returned as an answer
        continue
    
    kernel1 = v**2 * gp.kernels.RBF(length_scale=l_rbf, length_scale_bounds=(np.exp(-2), np.exp(5)))
    kernel2 = gp.kernels.ExpSineSquared(length_scale=l_per, periodicity=p, length_scale_bounds=(np.exp(-2), np.exp(5)),
                                    periodicity_bounds=(np.exp(0), np.exp(4)))

    final_kernel = kernel1 * kernel2 + gp.kernels.WhiteKernel(noise_level=sigma, noise_level_bounds=(np.exp(-10), np.exp(0)))   
 
    model = gp.GaussianProcessRegressor(kernel=final_kernel, normalize_y=True, n_restarts_optimizer=100)

    model.fit(xs[:, np.newaxis], ys[:, np.newaxis])

    # Backup the params after every iteration
    with open('/home/aks73/PML/params_simple_kernel.txt', 'a') as fout:
        fout.write(user_id+'\n')
        fout.write(str(model.kernel_.theta)+'\n')
        fout.write(str(model.log_marginal_likelihood_value_ )+'\n')
