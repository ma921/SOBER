import time
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt


class TS_GP():
	def __init__(self,x,F,GP,opt = "general", epsilon = 0.001, D = 1., multistart = 25, minimizer = "L-BFGS-B", grid = 100, verbose = False):
		'''
			Create the object

			Args:
				x - initial number of points
				F - lambda for evaluation
				GP - Gaussian Process object
		'''
		self.minimizer = minimizer
		self.x = x
		self.F = F
		self.y = F(x)
		self.GP = GP
		self.t = 1.0
		self.opt = opt
		self.grid = grid
		self.epsilon = epsilon
		self.multistart = multistart
		self.verbose = verbose
		self.fit_gp(self.x,self.y)

	def fit_gp(self,x,y, iterative = False):
		self.GP.fit_gp(x,y, iterative = iterative)
		return None

	def sample_point(self, xtest = None):
		if (self.opt == "first_order") and (self.GP.admits_first_order):
			(xnext,fxnext) = self.GP.sample_and_optimize(xtest, multistart = self.multistart, minimizer = self.minimizer, grid = self.grid)
			return (xnext,fxnext)

		elif (self.opt == "iterative"):
			(xnext, fxnext) = self.GP.sample_iteratively_max(xtest, multistart= self.multistart, minimizer= self.minimizer, grid = self.grid)
			return (xnext,fxnext)
		else:
			if xtest is None:
				raise AssertionError("Cannnot run a general kernel (an approximation) without specified grid")
			else:
				(xnext,value) = self.GP.sample_and_max(xtest)
				return (xnext,value)

	def isin(self,xnext):
		for v in self.x:
			if torch.norm(v-xnext) < self.epsilon:
				return True

		return False

	def step(self,xtest = None):
		start = time.time()
		self.fit_gp(self.x,self.y)
		(xnext,_) = self.sample_point(xtest)
		end = time.time()
		if self.verbose == 2:
			print ("Point found",end-start)
		xnext = xnext.view(-1,1)
		#detect whethe
		reward = self.F(torch.t(xnext))

		if self.verbose == 2:
			print ("Func evaluated",end-start)
		if not self.isin(xnext[:,0]):
			self.x = torch.cat((self.x, torch.t(xnext)), dim=0)
			self.y = torch.cat((self.y, reward),dim = 0)

		if self.verbose == 1:
			print (self.t, torch.t(xnext),reward)
		self.t +=1
		return (reward,torch.t(xnext))

	def eig_gap(self, xtest):
		(mu, s) = self.GP.mean_var(xtest)
		ss,_ = torch.max(s)
		return ss


	def plot_conf(self,xtest,d, empirical = True, F = None):
		'''
		Confidence
		'''
		from scipy.interpolate import griddata
		from mpl_toolkits.mplot3d import Axes3D
		no_samples = 1

		if d == 1:
			lw = 3
			ms = 12
			plt.rcParams.update({'font.size': 22})
			self.fit_gp(self.x[:-1],self.y[:-1])
			(mu,s) = self.GP.mean_var(xtest)
			plt.clf()
			plt.xlim([-0.5, 0.5])
			plt.ylim([-2.5, 2.5])
			index = np.argmax(self.F(xtest))
			plt.title("Bayesian Optimization Example")

			plt.plot(self.x[:-1].numpy(), self.y[:-1].numpy(), 'r+', ms=ms, marker="o", label = "evaluations")
			if F is None:
				plt.plot(xtest.numpy(), self.F(xtest).numpy(), 'b-', label = "true\nfunction", lw = lw)
			else:
				plt.plot(xtest.numpy(), F(xtest).numpy(), 'b-', label = "true\nfunction", lw = lw)
			#f = self.GP.sample(xtest,no_samples)
			if self.GP.temp is not None:
				plt.plot(xtest.numpy(), self.GP.temp.numpy(), 'g-', label = "acquisition\nfunction", lw = lw)
				value, index = torch.max(self.GP.temp, dim = 0)
				x = xtest[index]
				plt.plot([x.numpy().flatten(),x.numpy().flatten()],[value.numpy().flatten(),self.F(xtest)[index].numpy().flatten()],'--', lw = lw)
				plt.plot(x.numpy(), value.numpy(), 'g+', ms=ms, marker="o")
			plt.fill_between(xtest.numpy().flat, (mu-2*s).numpy().flat, (mu+2*s).numpy().flat , color="#dddddd", label = "2x std. dev.")
			plt.plot(xtest.numpy(), mu.numpy(), 'r--', lw=lw, label = "mean\nprediction")

			fig = plt.gcf()
			fig.set_size_inches(10, 6)
			ax = plt.subplot(111)
			# Shrink current axis by 20%
			box = ax.get_position()
			ax.set_position([0.08, box.y0, box.width * 0.8, box.height])
			# Put a legend to the right of the current axis
			ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

			#plt.legend(loc = "upper right")
			plt.draw()



		elif d == 2:


			self.fit_gp(self.x,self.y)

			(mu,s) = self.GP.mean_var(xtest)

			plt.clf()
			ax = plt.axes(projection = '3d')
			xx = xtest[:,0].numpy()
			yy = xtest[:,1].numpy()
			z = self.F(xtest).numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z = griddata((xx, yy), z[:,0], (grid_x, grid_y), method='linear')
			grid_z_mu = griddata((xx, yy), mu[:,0].numpy(), (grid_x, grid_y), method='linear')
			ax.plot_surface(grid_x, grid_y, grid_z, shade = True,alpha=0.1)

			ax.scatter(self.x[:,0].numpy(), self.x[:,1].numpy(), self.y.numpy(), c='r', s=100, marker="o", depthshade = False)
			ax.plot_surface(grid_x, grid_y, grid_z_mu, shade = True, alpha=0.2, color = 'r')
			# sample a couple of points
			f = self.GP.sample(xtest,1)
			grid_z_f = griddata((xx, yy), f[:,0].numpy(), (grid_x, grid_y), method='linear')
			ax.plot_surface(grid_x, grid_y, grid_z_f, shade = True, alpha=0.15, color = 'g')
			plt.draw()
		else:
			"nothing"

class Batched_BO():
	#Util class for Batched BO algorithms, regardless of being UCB or TS

	def isin_hal(self,xnext,x_hal):
		for v in x_hal:
			if torch.norm(v-xnext) < self.epsilon:
				return True

	def lower_confidence(self, xtest, beta=1.0): #LCB
		(ymean,yvar) = self.GP.mean_var(xtest, False)
		yvar = torch.nan_to_num(yvar)
		conf = ymean - torch.tensor(beta).sqrt()*yvar
		return conf

	def best_inference(self, xtest, beta=1.0):
		'''
		Best conservative guess for maximizer, used to compute inference regret
		'''
		if (xtest is None):
			#raise ValueError("To get inference regret, must use finite grid method")
			xtest = self.fake_xtest
		conf = self.lower_confidence(xtest, beta)
		print(conf.size())
		i = torch.argmax(conf)
		xnext = xtest[i]
		xnext = xnext.view(-1,1)
		return (xnext,conf[i])

	def eff_d(self, K, s):
		return torch.trace(K @ torch.inverse(K + s * torch.eye(K.shape[0], dtype=torch.double)))

	def step_update(self, xtest, x_batch, safe, get_inference, log_lik_hist = None):
		(_,first_point_var) = self.GP.mean_var(x_batch[0].view(-1,self.GP.d))

		batch_rewards = torch.tensor([], dtype=torch.double)
		#sample reward for the batch
		for xnext in x_batch:
			xnext = xnext[:,None]
			reward = self.F(torch.t(xnext))
			true_reward = self.true_F(torch.t(xnext))
			if not safe or not self.isin(xnext[:,0]):
				self.x = torch.cat((self.x, torch.t(xnext)), dim=0)
				self.y = torch.cat((self.y, reward), dim = 0)
			if batch_rewards.size()[0] == 0:
				batch_rewards = true_reward
			else:
				batch_rewards = torch.cat((batch_rewards, true_reward), dim = 0)

		res = {'batch_rewards': batch_rewards, 'x_batch': x_batch, 'first_p_var': first_point_var[0][0]}

		if log_lik_hist is not None:
			res['log_lik_hist'] = log_lik_hist

		#Get inference reward
		if (get_inference):
			(xinfer, _) = self.best_inference(xtest)
			infer_true_reward = self.true_F(torch.t(xinfer))
			res['inference_reward'] = infer_true_reward

		self.fit_gp(self.x, self.y)

		return res

	def check_start_K(self, xtest, fake_xtest_size = 512):
		if self.start_K is None:
			#If we are in the continuous setting, then xtest=None, but we may need an xtest for some metrics
			if xtest is None:
				bounds = np.array(self.GP.bounds)
				self.fake_xtest = torch.tensor(
									np.random.uniform(low = np.tile(bounds[:,0], (fake_xtest_size,1)), high = np.tile(bounds[:,1], (fake_xtest_size,1)), size = (fake_xtest_size, self.GP.d)),
									dtype=torch.double)
			(_, self.start_K) = self.GP.mean_var(xtest if xtest is not None else self.fake_xtest, full=True)

	def posterior_variance_stats(self, xtest, x_star, n_runs_pmax=50, eff_dim_s=False, compute_eig=False):
		(_, x_star_var) = self.GP.mean_var(x_star.view(-1,self.GP.d))
		if xtest is None:
			continuous = True
			xtest = self.fake_xtest
		else:
			continuous = False
		(_, post_K) = self.GP.mean_var(xtest, full=True)

		#reusing pmax sample computation (e.g. when DPP-TS samples from it)
		prev_samples_n = self.temp_pmax_samples.shape[0] if self.temp_pmax_samples is not None else 0
		if prev_samples_n < n_runs_pmax:
			new_n_runs_pmax = n_runs_pmax - prev_samples_n
			pmax_samples = torch.empty((new_n_runs_pmax, xtest.shape[1]), dtype=torch.double)
			for run_i in range(new_n_runs_pmax):
				(xnext,_) = self.sample_point(xtest if not continuous else None)
				pmax_samples[run_i] = xnext
			if self.temp_pmax_samples is not None:
				pmax_samples = torch.cat((self.temp_pmax_samples, pmax_samples), dim=0)
		else:
			pmax_samples = self.temp_pmax_samples

		#calculate estimated vector of Pmax probabilities for the domain
		pmax_freqs = torch.zeros(xtest.shape[0], dtype=torch.double)
		if continuous:
			distances = torch.cdist(pmax_samples, xtest)
			for pmax_s_dists in distances:
				s_idx = torch.argmin(pmax_s_dists)
				pmax_freqs[s_idx] += 1
		else:
			for pmax_s in pmax_samples:
				s_idx = (xtest == pmax_s).nonzero(as_tuple=True)
				pmax_freqs[s_idx[0]] += 1
		pmax_freqs /= pmax_samples.shape[0]

		var_diag = torch.diag(post_K)
		avg_var = torch.sum(var_diag / var_diag.shape[0])

		pmax_weighted_var = var_diag * pmax_freqs #element-wise mult
		pmax_avg_var = torch.sum(pmax_weighted_var)

		eff_d_s = self.GP.s if eff_dim_s else 1.
		post_eff_d = self.eff_d(post_K, eff_d_s)

		D_pmax_sqrt = torch.diag(torch.sqrt(pmax_freqs))
		pmax_start_K = D_pmax_sqrt @ self.start_K @ D_pmax_sqrt
		pmax_start_K_eff_d = self.eff_d(pmax_start_K, eff_d_s)
		pmax_post_K = D_pmax_sqrt @ post_K @ D_pmax_sqrt
		pmax_post_K_eff_d = self.eff_d(pmax_post_K, eff_d_s)

		res = {'x_star_var': x_star_var[0][0], 'avg_var': avg_var, 'pmax_avg_var': pmax_avg_var, 'post_eff_d': post_eff_d, 'pmax_start_K_eff_d': pmax_start_K_eff_d, 'pmax_post_K_eff_d': pmax_post_K_eff_d}

		if compute_eig and (self.t-2) % 10 == 0:
			K_eig = torch.norm(torch.eig(post_K)[0], dim=1)
			pmax_K_eig = torch.norm(torch.eig(pmax_post_K)[0], dim=1)
			plt.clf()
			fig = plt.gcf()
			fig.set_size_inches(10, 6)
			ax = plt.subplot(111)
			bins = np.linspace(0, max(torch.max(K_eig).numpy(), torch.max(pmax_K_eig).numpy()), 20)
			ax.hist(K_eig, bins=bins, alpha=0.5, label="K")
			ax.hist(pmax_K_eig, bins=bins, alpha=0.5, label="pmax_K")
			ax.legend(loc='upper right')
			plt.draw()
			plt.pause(5.0)

		return res

	def plot_conf_b(self,xtest,d, last_batch = None, empirical = True, F = None):
		'''
		Confidence
		last_batch is a dict containing the last selected batch, containing "x" and "y"
		'''
		if d == 1:
			self.one_d_plot(xtest, last_batch, F)
		else:
			super().plot_conf(xtest, d, empirical, F)

	def one_d_plot(self, xtest, last_batch = None, F = None):
		setup = self.one_d_plot_setup(xtest, last_batch)
		self.one_d_plot_common(xtest, setup, last_batch, F)
		self.one_d_plot_draw()

	def one_d_plot_setup(self, xtest, last_batch = None):
		setup = {}
		setup["lw"] = 3
		setup["ms"] = 12
		if last_batch is not None:
			setup["last_batch_size"] = last_batch["x"].shape[0]
		else:
			setup["last_batch_size"] = 0
		plt.rcParams.update({'font.size': 22})
		self.fit_gp(self.x[:-setup["last_batch_size"]],self.y[:-setup["last_batch_size"]])
		(setup["mu"],setup["s"]) = self.GP.mean_var(xtest)
		plt.clf()
		plt.xlim([-0.5, 0.5])
		plt.ylim([-2.5, 3.5])
		setup["index"] = np.argmax(self.F(xtest))
		plt.title("Bayesian Optimization Example")
		return setup

	def one_d_plot_common(self, xtest, setup, last_batch = None, F = None):
		plt.plot(self.x[:-setup["last_batch_size"]].numpy(), self.y[:-setup["last_batch_size"]].numpy(), 'r+', ms=setup["ms"], marker="o", label = "evaluations")
		if last_batch is not None:
			plt.plot(last_batch["x"].numpy(), last_batch["y"].numpy(), 'g+', ms=setup["ms"], marker="o")
		if F is None:
			plt.plot(xtest.numpy(), self.F(xtest).numpy(), 'b-', label = "true\nfunction", lw = setup["lw"])
		else:
			plt.plot(xtest.numpy(), F(xtest).numpy(), 'b-', label = "true\nfunction", lw = setup["lw"])

		plt.fill_between(xtest.numpy().flat, (setup["mu"]-2*setup["s"]).numpy().flat, (setup["mu"]+2*setup["s"]).numpy().flat , color="#dddddd", label = "2x std. dev.")
		plt.plot(xtest.numpy(), setup["mu"].numpy(), 'r--', lw=setup["lw"], label = "mean\nprediction")

	def one_d_plot_draw(self):
		fig = plt.gcf()
		fig.set_size_inches(10, 6)
		ax = plt.subplot(111)
		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([0.08, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

		#plt.legend(loc = "upper right")
		plt.draw()

class Batched_TS_GP(TS_GP, Batched_BO):

	def __init__(self, *args, true_F=None, **kw):
		super().__init__(*args, **kw)
		if (true_F is None):
			self.true_F = self.F
		else:
			self.true_F = true_F
		self.temp_pmax_samples = None
		self.start_K = None

	def sample_point(self, xtest = None):
		'''
		Add option of linear programming optimization
		'''
		if self.opt == "linprog":
			(xnext,fxnext) = self.GP.sample_and_optimize()
			return (xnext,fxnext)
		else:
			return super().sample_point(xtest)

	def step(self, xtest = None, batch_size = 1, safe = False, hal=True, get_inference=True):
		self.check_start_K(xtest)
		#work on "hallucinated data"
		x_hal = self.x
		y_hal = self.y
		x_batch = torch.tensor([], dtype=torch.double)
		for i in range(batch_size):
			start = time.time()
			if hal:
				self.fit_gp(x_hal,y_hal)
			#use sampled reward as hallucinated reward
			(xnext, _) = self.sample_point(xtest)
			(hal_reward, _) = self.GP.mean_var(xnext.view(1,-1))
			end = time.time()
			if self.verbose == 2:
				print ("Point found",end-start)
			xnext = xnext.view(-1,1)
			if x_batch.size()[0] == 0:
				x_batch = torch.t(xnext)
			else:
				x_batch = torch.cat((x_batch, torch.t(xnext)), dim=0)
			if hal and (not safe or not self.isin_hal(xnext[:,0], x_hal)):
				x_hal = torch.cat((x_hal, torch.t(xnext)), dim=0)
				y_hal = torch.cat((y_hal, hal_reward), dim = 0)

		if hal:
			self.fit_gp(self.x, self.y)

		res = self.step_update(xtest, x_batch, safe, get_inference)
		self.t += 1
		return res
