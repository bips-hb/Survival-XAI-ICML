import numpy as np
from survlimepy import SurvLimeExplainer


def run_survlime(dat, model, num_samples = 50, model_type = "deephit"):
  
  import numpy as np
  import torch
  from survlimepy import SurvLimeExplainer
  from sksurv.linear_model import CoxPHSurvivalAnalysis
  
  # Get model
  model = model['model']
  
  # Get the data
  time = dat['time']
  status = dat['status'].astype('bool')
  x = dat.drop(['time', 'status'], axis=1).astype('float32')
  #wtype = np.dtype([('events', status.dtype), ('time', time.dtype)])
  #w = np.empty(len(time), dtype=wtype)
  #w['events'] = status
  #w['time'] = time
  
  if model_type == "deephit":
    model_output_times = model.duration_index
  elif model_type == "deepsurv":
    model_output_times =  np.array(model.compute_baseline_hazards())
  
  explainer = SurvLimeExplainer(
    training_features = x,
    training_events = status,
    training_times = time,
    model_output_times = model_output_times,
    random_state=10,
  )
  
  def pred_fun(x):
    return model.predict_surv(x.to_numpy(dtype = 'float32'))
  
  def calc_lime_weights(i):
    
    # Calc lime weights (i.e. the beta coefficients for a cox model)
    weights =  explainer.explain_instance(
      data_row = x.iloc[i].to_numpy(),
      predict_fn = pred_fun,
      num_samples = num_samples,
      verbose = False,
    )
    
    # Calculate the prediction
    #model_interpretable = CoxPHSurvivalAnalysis(alpha=0.0001)
    #model_interpretable.fit(x, w)
    #model_interpretable.coef_ = weights
    #preds_survlime = model_interpretable.predict_survival_function(x)
    
    #preds_surv_y = np.mean([x.y for x in preds_survlime], axis=0)
    
    return weights
  
  res = np.array([calc_lime_weights(i) for i in range(x.shape[0])])

  
  return res


#import pandas as pd
#import torch
#import torchtuples as tt
#import pycox
#import numpy as np

#dat = pd.read_csv('test.csv')
#net = torch.load('model.pt')
#cuts = np.load('cuts.npy')

#model = pycox.models.DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index = cuts)


#run_survlime(dat, model)
