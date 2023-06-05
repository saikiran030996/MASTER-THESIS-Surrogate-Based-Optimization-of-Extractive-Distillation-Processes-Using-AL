import scipy.stats as st
from scipy.optimize import differential_evolution
from aspen_utils import *
import sklearn.gaussian_process as gp
from scipy.special import logit, expit
np.random.seed(42)
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from smt.sampling_methods import Random
from scipy import optimize
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint
import warnings
warnings.filterwarnings("ignore")
from aspen_utils import *
from scipy.special import logit, expit
from smt.applications.mixed_integer import MixedIntegerContext, FLOAT, ORD
Aspen_Plus = Aspen_Plus_Interface()
Aspen_Plus.load_bkp(r"simulation/ExtractiveDistillation.bkp", 0, 1)


def surrogate_model(X):
    # Obtain mean and std of predictions from GP model
    y_pred, y_pred_std = model.predict(X, return_std=True)
    return y_pred, y_pred_std


def acquisition_func(X):
    # Define the acquisition function (i.e., expected improvement) for optimization
    X = np.array(X).reshape(-1, 4)
    y_pred, y_pred_std = surrogate_model(X)
    MPI = st.norm.cdf((y_train_best - y_pred) / y_pred_std)
    minus_MPI = -MPI
    if ac_mode == "MPI":
        af_value = minus_MPI
    elif ac_mode == "UCB":
        af_value = (y_pred - 1.0 * y_pred_std)
    return af_value

def model_(X):
    X = np.array(X).reshape(-1,4)
    y_pred = model1.predict(X,return_std=False)
    return y_pred


xtypes = [ORD, FLOAT, FLOAT, FLOAT]
xlimits = [[40, 80], [1, 10], [3.5, 6], [1, 8]]
mixint = MixedIntegerContext(xtypes, xlimits)
ransamp = mixint.build_sampling_method(Random)
x = ransamp(50)
final_df = pd.DataFrame(
        {'NStage': pd.Series(dtype='float'), 'RR': pd.Series(dtype='float'), 'TopPres': pd.Series(dtype='float'),
         'StoF': pd.Series(dtype='float'), 'DIST_C4H8_T1': pd.Series(dtype='float'), 'RebDuty_T1': pd.Series(dtype='float'),
         'hasERROR': pd.Series(dtype='float')})
Sampling_start_time = time.time()

for i in range(len(x)):
    Aspen_Plus.re_initialization()
    next_X = x[i]
    Input = list(next_X)
    NStage, RR, TopPres, StoF = Input
    Input = [NStage, RR, TopPres, StoF]
    FeedRate=500
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\NSTAGE").Value = NStage
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\BASIS_RR").Value = RR
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\PRES1").Value = TopPres
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = StoF * FeedRate
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\FEED_STAGE\FEED").Value = np.ceil(0.5 * NStage)
    # the mixture is fed in the middle of the column

    # run the process simulation
    Aspen_Plus.run_simulation()
    Aspen_Plus.check_run_completion()

    # collect results
    DIST_C4H8 = Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H8").Value
    RebDuty = Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\REB_DUTY").Value
    hasERROR = Aspen_Plus.check_convergency()
    Input = Input
    DIST_C4H8 = DIST_C4H8
    RebDuty = RebDuty
    hasERROR = hasERROR
    lst = Input
    lst.append(DIST_C4H8)
    lst.append(RebDuty)
    lst.append(hasERROR)
    final_df.loc[len(final_df)] = lst
Sampling_end_time = time.time()
final_df.to_csv('data_constrain_high.csv')
Sampling_time = Sampling_end_time - Sampling_start_time
print("Sampling_time in seconds: ",Sampling_time )

kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel,
                                     optimizer="fmin_l_bfgs_b",
                                     n_restarts_optimizer=30,
                                     alpha=1e-4,
                                     normalize_y=True)
model1 = gp.GaussianProcessRegressor(kernel=kernel,
                                     optimizer="fmin_l_bfgs_b",
                                     n_restarts_optimizer=30,
                                     alpha=1e-4,
                                     normalize_y=True)


input_name = ["NStage", "RR", "TopPres", "StoF"]
dataset = pd.read_csv("data_constrain_high.csv")
count = len(dataset[dataset["DIST_C4H8_T1"] >= 0.995])
print(count)
new_data = dataset[dataset["DIST_C4H8_T1"] >= 0.995].iloc[:16,:]
X = new_data[input_name].values
y_puri, y_duty = new_data["DIST_C4H8_T1"].values, new_data["RebDuty_T1"].values
X_initial = X
y_initial = y_duty
y_puri_initial = logit(y_puri)
print(X_initial.shape)
print(y_initial.shape)


#Aspen_Plus.load_bkp(r"simulation/ExtractiveDistillation.bkp", 0, 1)
final_df = pd.DataFrame(
        {'NStage': pd.Series(dtype='float'), 'RR': pd.Series(dtype='float'), 'TopPres': pd.Series(dtype='float'),
         'StoF': pd.Series(dtype='float'), 'y_predicted': pd.Series(dtype='float'),'DIST_C4H8': pd.Series(dtype='float'), 'error': pd.Series(dtype='float'),'y_real': pd.Series(dtype='float'),
         'af_value':pd.Series(dtype='float'),'Mode':pd.Series(dtype='float'),'model_purity':pd.Series(dtype='float')})
iter_index = 0
ac_mode = "MPI"
restart = False
error= np.inf
error_tol = 0.003
Simulation_start_time= time.time()
while iter_index<=50:
    print(f"*------------------------------------------------------iteration:{iter_index}------------------------------------------------------*")

    if not restart:
        iter_index += 1

    if iter_index == 1:
        X_train, y_train,y_puri_ = X_initial, y_initial,y_puri_initial
    else:
        if not restart:
            X_train = np.concatenate((X_train, Output_arr.reshape(-1, 4)))
            y_train = np.concatenate((y_train.reshape(-1,1), next_y.reshape(-1,1)))
            y_puri_ = np.concatenate((y_puri_.reshape(-1,1),next_y_.reshape(-1,1)))
        restart = False
    # GP model training

    if not restart:
        model.fit(X_train, y_train)
        model1.fit(X_train,y_puri_)

    restart = False

    y_pred_train = model.predict(X_train)  # 6 points
    y_duty_filtered = [y_d_val for y_d_val,y_p_val in zip(y_train,y_puri_) if y_p_val >= logit(0.995)]
    y_train_best = min(y_duty_filtered)
    print(y_train_best)

    # Perform the optimization
    lb = logit(0.995)
    ul = logit(1.0)
    nlc = optimize.NonlinearConstraint(model_,lb, ul)
    bounds = [(40, 80), (1, 10), (3.5, 6), (1, 8)]
    integrality = ([1, 0, 0, 0])
    result = differential_evolution(acquisition_func, integrality=integrality, bounds=bounds,constraints = nlc)
    next_X = result.x  # next point to be evaluated
    if ac_mode == "MPI":
        af_value = -result.fun
    else:
        af_value = result.fun
    if af_value <= 0.5:
        restart = True
        ac_mode = "UCB"
        continue
    Aspen_Plus.re_initialization()
    Input = list(next_X)
    NStage, RR, TopPres, StoF = Input
    Input = [NStage, RR, TopPres, StoF]
    print("Next_value: ", Input)
    Output_array = []
    FeedRate = 500
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\NSTAGE").Value = NStage
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\BASIS_RR").Value = RR
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\PRES1").Value = TopPres
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value = StoF * FeedRate
    Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Input\FEED_STAGE\FEED").Value = np.ceil(0.5 * NStage)
    # the mixture is fed in the middle of the column

    # run the process simulation
    Aspen_Plus.run_simulation()
    # Aspen_Plus.check_run_completion()

    # collect results
    DIST_C4H8 = Aspen_Plus.Application.Tree.FindNode(r"\Data\Streams\DIST1\Output\MOLEFRAC\MIXED\C4H8").Value
    RebDuty = Aspen_Plus.Application.Tree.FindNode(r"\Data\Blocks\ED\Output\REB_DUTY").Value
    hasERROR = Aspen_Plus.check_convergency()
    Output = [DIST_C4H8, RebDuty, hasERROR]
    print("AspenPlus result-", Output)
    print("real_value", Output[1])
    real_value = Output[1]
    next_y = np.array(RebDuty)
    next_y_ = logit(np.array(DIST_C4H8))
    Output_arr = np.array(Input)
    y_predicted = model.predict(Output_arr.reshape(-1, 4))
    y_predicted_purity = model1.predict(Output_arr.reshape(-1,4))
    print("predicted-value", y_predicted)
    error = np.abs(y_predicted - Output[1])
    Input = Input
    output = y_predicted
    error = error
    real = real_value
    model_purity = expit(y_predicted_purity)
    lst = Input
    lst.append(output[0])
    lst.append(Output[0])
    lst.append(error[0])
    lst.append(real)
    lst.append(af_value)
    lst.append(ac_mode)
    lst.append(model_purity[0])
    final_df.loc[len(final_df)] = lst
KillAspen()
Simulation_end_time = time.time()
final_df.to_csv('result_constrain_high.csv')
Simulation_time = Simulation_end_time - Simulation_start_time

print("Simulation_time in seconds: ",Simulation_time)

