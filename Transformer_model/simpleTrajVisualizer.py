import numpy as np
import matplotlib.pyplot as plt

def vis_trajectory_scatter(best_model, data_src_inputs, data_src_targets, lag_time = 2.0):
  input_list, target_list, preds_list = get_predictions(best_model, data_src_inputs, data_src_targets)

  min_val, max_val = 0, 100
  
  for i in range(0, len(input_list)):
    # fig, ax = plt.subplots()
    plt.figure(figsize=(3,3))
    inputs = input_list[i]
    targets = target_list[i]
    preds = preds_list[i]

    xy_inputs = []
    xy_preds = []
    xy_target = []

    for inp in inputs:
      x = inp % 100
      y = inp / 100 

      xy_inputs.append([x, y])

    for inp in preds:
      x = inp % 100
      y = inp / 100 
      xy_preds.append([x, y])

    for inp in targets:
      x = inp % 100
      y = inp / 100 
      xy_target.append([x, y])

    x_inp, y_inp = zip(*xy_inputs)
    x_tar, y_tar = zip(*xy_target)
    x_pred, y_pred = zip(*xy_preds)

    plt.scatter(x_inp,y_inp)
    plt.scatter(x_tar,y_tar, color='r')
    plt.scatter(x_pred,y_pred)

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.show()
    print("Input: ", xy_inputs)
    print("Preds: ", xy_preds)
    print("Truth: ", xy_target)
    time.sleep(lag_time)
