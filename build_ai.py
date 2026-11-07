import numpy as np,pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
p='/mnt/data/pd_training.csv'; d=pd.read_csv(p)
X=d[['theta_deg','omega_deg_s']].to_numpy(); y=d.command.to_numpy()
# fixed physical scaling: no reliance on unavailable sklearn on Core2
scale=np.array([5.,200.]); x=X/scale; z=y/400.
idx=int(len(x)*.8)
best=None
for seed in range(8):
 m=MLPRegressor(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',alpha=0.00001,max_iter=3000,random_state=seed,tol=1e-9)
 m.fit(x[:idx],z[:idx]);pred=m.predict(x[idx:])*400
 err=mean_absolute_error(y[idx:],pred)
 if best is None or err<best[0]:best=(err,m,seed)
err,m,seed=best
print('MAE chronological holdout',err,'seed',seed,'max',np.max(np.abs(y[idx:]-m.predict(x[idx:])*400)))
print('train',mean_absolute_error(y[:idx],m.predict(x[:idx])*400))
print('weights',m.coefs_)
import json
weights={'scale':[5.,200.],'out_scale':400.,'w1':m.coefs_[0].tolist(),'b1':m.intercepts_[0].tolist(),'w2':m.coefs_[1][:,0].tolist(),'b2':float(m.intercepts_[1][0]),'validation_mae':float(err)}
open('/mnt/data/ai_weights.json','w').write(json.dumps(weights,indent=2))
