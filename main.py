import os
os.environ['DDE_BACKEND'] = 'tensorflow'

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import tensorflow as tf
import numpy as np
from scipy.integrate import quad, dblquad
from math import sqrt
import deepxde as dde


'''
Setting json encoder for converting numpy.integer to python's int when saving model data.
'''
class MyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    else:
      return super(MyEncoder, self).default(obj)

seed_value = 0 # setting seed to a fixed value
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
dde.config.set_random_seed(seed_value)

class PDE:
  def __init__(self, p, net_data):
    self.p = p
    self.xmin, self.xmax = [-40, 1], [40, 2]
    self.tmax = 30
    self.net_data = net_data
    self.x_num = 400
    self.x = np.linspace(self.xmin[0], self.xmax[0], self.x_num)
    self.lr=1e-3
    self.resample = False
    self.resample_period = 10

    self.vl2relerror_xt = np.vectorize(self.l2relerror_xt)
    self.vdde_l2relerror = np.vectorize(self.dde_l2relerror)
    self.eps = 1e-5
    self.error_data = {"eps" : self.eps}
    
  

  def set_data(self):
    self.geom = dde.geometry.geometry_2d.Rectangle(xmin=self.xmin, xmax=self.xmax)
    self.timeDomain = dde.geometry.TimeDomain(0, self.tmax)
    self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timeDomain)
    self.bc = dde.icbc.DirichletBC(self.geomtime, self.pde_sol, self.on_boundary, component=0)
    self.ic = dde.icbc.IC(self.geomtime, self.pde_sol, lambda _, on_initial: on_initial)
    self.data = dde.data.TimePDE(
                        self.geomtime,
                        self.pde,
                        [self.bc, self.ic],
                        num_domain = self.net_data["ND"],
                        num_boundary = self.net_data["Nb"],
                        num_initial = self.net_data["N0"]
                        )
    self.net = dde.nn.FNN([3] + [self.net_data["neurons"]] * self.net_data["layers"] + [1], "tanh", "Glorot normal")
    self.model = dde.Model(self.data, self.net)


  def on_boundary(self, x, on_boundary):
    return on_boundary and (dde.utils.isclose(x[0], self.xmin[0]) or dde.utils.isclose(x[0], self.xmax[0]))


  def pde(self, x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xxx = dde.grad.hessian(dy_x, x, i=0, j=0)
    pde = dy_t + y**self.p * dy_x + dy_xxx
    return pde


  def pde_sol(self, X):
    x_arr = X[:, 0:1]
    a_arr = X[:, 1:2]
    t_arr = X[:, 2:3]


    if self.p == 1:
      c = a_arr / 3
      k = np.sqrt(a_arr / 12)
      return a_arr*np.power(np.cosh(k*(x_arr - c*t_arr)), -2)

    if self.p == 2:
      c = a_arr**2 / 6
      k = a_arr / sqrt(6)
      return a_arr*np.power(np.cosh(k*(x_arr - c*t_arr)), -1)


  def train(self, iterations):
    display_every = 200
    checkpointer = dde.callbacks.ModelCheckpoint(
      f"model/ckpt/",
      period=200,
      verbose=1,
      save_better_only=True
    )
    if self.resample:
      resampler = dde.callbacks.PDEPointResampler(period=self.resample_period)
      return self.model.train(iterations=iterations,
                        display_every=display_every,
                        model_save_path=f"model/ckpt/",
                        callbacks=[checkpointer, resampler])


    return self.model.train(iterations=iterations,
                            display_every=display_every,
                            model_save_path=f"model/ckpt/",
                            callbacks=[checkpointer]
                            )

  def get_arr_X(self, t, a):
    t_array = np.full_like(self.x, t)
    a_array = np.full_like(self.x, a)
    X = np.stack((self.x, a_array, t_array), axis=1)
    return X

  def dde_l2relerror(self, t, a):
    X = self.get_arr_X(t, a)
    y_pred = self.model.predict(X)   #.reshape(self.x_num)
    y_true = self.pde_sol(X)
    return dde.metrics.l2_relative_error(y_true, y_pred)

  def l2relerror_xt(self, a):
    func = lambda x, t: (self.pde_sol(np.array([[x, a, t]])) - self.model.predict(np.array([[x, a, t]])))**2
    l2_norm_diff = sqrt(dblquad(func, self.xmin[0], self.xmax[0], lambda t: 0, lambda t: self.tmax, epsabs=self.eps, epsrel=self.eps)[0])
    l2_norm_true = sqrt(dblquad(lambda x, t: self.pde_sol(np.array([[x, a, t]]))**2, self.xmin[0], self.xmax[0], lambda t: 0, lambda t: self.tmax, epsabs=self.eps, epsrel=self.eps)[0])
    return l2_norm_diff / l2_norm_true


  def gif(self, a):
    Figure = plt.figure()
    predicted_line = plt.plot([], 'b-')[0]
    solution_line = plt.plot([], 'r--')[0]
    plt.xlim(self.xmin[0], self.xmax[0])
    plt.ylim(-0.5, 2.5)
    frames=120
    def AnimationFunction(frame):
        K= self.tmax/frames
        plt.title(f"t={K*frame:.2f}")
        X = self.get_arr_X(t=K*frame, a=a)
        y_pred = self.model.predict(X)
        y_sol = self.pde_sol(X)
        predicted_line.set_data((self.x, y_pred))
        solution_line.set_data((self.x, y_sol))

    anim_created = FuncAnimation(Figure, AnimationFunction, frames=frames, interval=25)
    writer = animation.PillowWriter(fps=24,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    anim_created.save(f'model/prediction_a{a}.gif', writer=writer)
    plt.close()

  def dde_l2relerror_scatter(self):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.cla()
    ax.set_xlim(self.xmin[1], self.xmax[1])
    ax.set_ylim(0, self.tmax)
    ax.set_xlabel("a")
    ax.set_ylabel("t")
    ax.set_zlabel("Error")
    plt.title("dde_l2relerror")

    A = np.linspace(self.xmin[1], self.xmax[1], 20)
    T = np.linspace(0, self.tmax, 20)
    A, T = np.meshgrid(A, T)
    Z = self.vdde_l2relerror(t=T, a=A)
    dde_l2relerror_max = np.max(Z)
    dde_l2relerror_mean = np.mean(Z)
    self.error_data["dde_l2relerror_max"] = dde_l2relerror_max
    self.error_data["dde_l2relerror_mean"] = dde_l2relerror_mean

    ax.scatter(A, T, Z, s = 5, marker = 'o', cmap=cm.inferno, c=Z)
    plt.savefig(f"model/dde_l2relerror.png")


  def l2relerror_xt_scatter(self):
    '''
    L2 Norm integrating in x and t.
    '''
    plt.clf()
    fig, ax = plt.subplots()
    a = np.linspace(self.xmin[1], self.xmax[1], 50)
    error = self.vl2relerror_xt(a)
    self.error_data["l2relerror_xt_max"] = np.max(error)
    self.error_data["l2relerror_xt_mean"] = np.mean(error)

    ax.plot(a, error)
    ax.set_xlabel("a")
    ax.set_ylabel("Error")
    plt.title("l2relerror_xt")
    plt.savefig(f"model/l2relerror_xt.png")

  def save_data(self):
    self.net_data["lr"] = self.lr
    self.net_data["resample"] = self.resample
    self.net_data["resample_period"] = self.resample_period
    self.net_data["xmin"] = self.xmin
    self.net_data["xmax"] = self.xmax
  

    try:
      os.makedirs(f"model/")
    except FileExistsError:
      pass

    try:
      with open(f"model/net_data.json", "x") as outfile:
        json.dump(self.net_data, outfile, indent=2)
    except FileExistsError:
      with open(f"model/net_data.json", "w") as outfile:
        json.dump(self.net_data, outfile, indent=2)

    try:
      with open(f"model/error_data.json", "x") as outfile:
        json.dump(self.error_data, outfile, indent=2)
    except FileExistsError:
      with open(f"model/error_data.json", "w") as outfile:
        json.dump(self.error_data, outfile, indent=2)


net_data = {
    "neurons" : 30,
    "layers" : 5,
    "ND" : 30_000,
    "Nb" : 6_000,
    "N0" : 1_200,
}
mkdv = PDE(p=2, net_data=net_data) # set p=1 for KdV and p=2 for mKdV

mkdv.xmin, mkdv.xmax = [-10, 1], [10, 2]
mkdv.tmax = 15
mkdv.set_data()
mkdv.model.compile("adam", lr=mkdv.lr)
mkdv.train(iterations=200_000)

mkdv.dde_l2relerror_scatter()
mkdv.gif(a=2)
#mkdv.l2relerror_xt_scatter() #slow to compute, don't use gpu
mkdv.save_data()
