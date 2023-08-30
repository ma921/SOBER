# This code relates to paper "Diversified Sampling for Batched Bayesian Optimization with Determinantal Point Processes", Nava et al. 2021, AISTATS 2022. It is intended to be under "Creative Commons Attribution 4.0" license.
# This is a snippet of the DPP-TS specific code in our soon to be open-sourced Bayesian Opt codebase.
# It extends a class that assumes an underlying GP + Thompson Sampling implementation accessible with ".sample_point(xtest)". "xtest" is a matrix of points that corresponds to a finite domain for the BO problem. It is "None" if the domain is continuous and has been otherwise defined in the parent class.
# This class uses the MCMC "Algorithm 1" from the paper for "n_runs" iterations to sample from the mixture of DPP and TS distributions
# the DPP_lambda and lambda_mode parameters related to the lambda-parametrization of the kernel explored in Appendix E.2 of the paper. Default settings correspond to the results in the main paper text.
# the "cutoff_iter" parameter corresponds to the "DPP-TS as initialization" technique discussed in the second half of Appendix E.2 of the paper. It's "T_init" from the paper, if "DPP_lambda" is set to 1.
# "first_ts" set to True sets to algorithm to be DPP-TS-alt from the paper (Section 5.2)
# ".step_update(...)" at the end runs the actual data collection at the proposed points and updates the GP
import numpy as np
import torch

from .snippet_batched import Batched_TS_GP


class MCMC_DPP_Batched_TS_GP(Batched_TS_GP):
	'''
	DPP batched TS, but with point-by-point MCMC sampling for the DPP
	'''
	def step(self, xtest, batch_size = 1, n_runs = 50, DPP_lambda=1., cutoff_iter = None, lambda_mode='mult', first_ts = False, safe = False, get_inference=True):
		'''
		n_runs are the runs of the MCMC algorithm
		cutoff_iter: number of iterations after which we no longer DPP sample, just exploit with TS
		DPP_lambda: either a number, or a callable to get lambda a function of t (iter number)
		lambda_mode: 'mult' if (I + lambda sigma^-2 K), 'pow' if (I + sigma^-2 K)^lambda
		first_ts boolean: True if the first sample of the batch is sampled with regular TS (non-DPP)
		'''
		self.check_start_K(xtest)
		#get lambda value for callable
		if callable(DPP_lambda):
			DPP_lambda = DPP_lambda(self.t)

		#Sample first x_batch
		x_batch = torch.tensor([], dtype=torch.double)
		for i in range(batch_size):
			(xnext,_) = self.sample_point(xtest)
			xnext = xnext.view(-1,1)
			if x_batch.size()[0] == 0:
				x_batch = torch.t(xnext)
			else:
				x_batch = torch.cat((x_batch, torch.t(xnext)), dim=0)
		(_,post_K_S) = self.GP.mean_var(x_batch, full=True)
		if lambda_mode == 'mult':
			K_S = torch.eye(batch_size, dtype=torch.float64) + DPP_lambda * (self.GP.s ** -2) * post_K_S
			det_K_S = torch.det(K_S)
		elif lambda_mode == 'pow':
			K_S = torch.eye(batch_size, dtype=torch.float64) + (self.GP.s ** -2) * post_K_S
			det_K_S = torch.det(K_S) ** DPP_lambda
		else:
			raise ValueError("Unsupported lambda_mode")
		log_lik = torch.log(det_K_S)
		log_lik_hist = log_lik.view(1,1)

		#start temp saving the sampled pmax (if needed for some metrics)
		self.temp_pmax_samples = x_batch

		#MCMC
		if cutoff_iter is None or self.t < cutoff_iter:
			while n_runs > 0:
				switch_i = np.random.randint(0 if not first_ts else 1, batch_size) #If first was sampled with regular TS, cannot swap it during MCMC
				(xnext,_) = self.sample_point(xtest)
				x_batch_prop = x_batch.clone()
				x_batch_prop[switch_i] = torch.t(xnext.view(-1,1))
				(_,post_K_S) = self.GP.mean_var(x_batch_prop, full=True)
				if lambda_mode == 'mult':
					K_S = torch.eye(batch_size, dtype=torch.float64) + DPP_lambda * (self.GP.s ** -2) * post_K_S
					det_K_S_prop = torch.det(K_S)
				elif lambda_mode == 'pow':
					K_S = torch.eye(batch_size, dtype=torch.float64) + (self.GP.s ** -2) * post_K_S
					det_K_S_prop = torch.det(K_S) ** DPP_lambda
				else:
					raise ValueError("Unsupported lambda_mode")
				alpha = torch.min(torch.tensor(1, dtype=torch.double), det_K_S_prop / det_K_S)
				if torch.rand(1) < alpha:
					x_batch = x_batch_prop
					det_K_S = det_K_S_prop
					if self.verbose==2:
						print("Accepted with probability {}".format(alpha.numpy()))
				else:
					if self.verbose==2:
						print("Rejected with probability {}".format(alpha.numpy()))
				log_lik = torch.log(det_K_S)
				log_lik_hist = torch.cat((log_lik_hist, log_lik.view(1,1)), dim=0)
				self.temp_pmax_samples = torch.cat((self.temp_pmax_samples, xnext.view(1,-1)), dim=0)
				n_runs -= 1

		res = self.step_update(xtest, x_batch, safe, get_inference, log_lik_hist=log_lik_hist)
		self.t += 1
		return res
