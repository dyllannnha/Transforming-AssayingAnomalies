"""
	Running this program compiles IML modules needed to do Gibbs estimation of Roll model and 
	places them in a library called IMLStor.

	Note:
	There are three routines to draw (simulate) the trade direction indicators (the q(t)).
	qDraw		Is written for efficiency, but it may be difficult to follow.
				It is a block sampler.
				It can only be used when c and the variance of u are fixed.
	qDrawSlow	Is written for clarity. It draws the q one-at-a-time, sequentially.
				It can only be used when c and the variance of u are fixed.
	qDrawVec	This is like qDraw, but can be used when c and variance of u are time-varying.
				(THIS ROUTINE HAS NOT BEEN THOROUGHLY TESTED.)

"""

import numpy as np
from scipy.stats import truncnorm
import scipy.linalg as la
from scipy.special import expit

def BayesRegressionUpdate(prior_mu, prior_cov, y, X, d_var):
    # Compute prior precision matrix
    covi = la.inv(prior_cov)
    Di = (1/d_var) * X.T @ X + covi
    D = la.inv(Di)
    # Compute posterior mean
    dd = (1/d_var) * X.T @ y + covi @ prior_mu
    post_mu = D @ dd
    post_cov = D
    return post_mu, post_cov

def BayesVarianceUpdate(prior_alpha, prior_beta, u):
    post_alpha = prior_alpha + len(u)/2
    post_beta = prior_beta + (u @ u)/2
    return post_alpha, post_beta

def RandStdNormT(zlow, zhigh):
    if zlow == -np.inf and zhigh == np.inf:
        return np.random.normal()
    a = (zlow - 0) / 1  # Standard normal has mean 0, std 1
    b = (zhigh - 0) / 1
    return truncnorm.rvs(a, b)

def mvnrndT(mu, cov, vLower, vUpper):
    # Cholesky decomposition
    f = la.cholesky(cov, lower=True)
    n = len(mu)
    eta = np.zeros(n)
    for k in range(n):
        if k == 0:
            low = (vLower[k] - mu[k]) / f[k, k]
            high = (vUpper[k] - mu[k]) / f[k, k]
        else:
            etasum = (f[k, :k] @ eta[:k])
            mu_k = mu[k] + etasum
            denom = f[k, k]
            low = (vLower[k] - mu_k) / denom
            high = (vUpper[k] - mu_k) / denom
        # Generate truncated normal
        sample = RandStdNormT(low, high)
        eta[k] = sample
    # Back-transform
    return mu + f @ eta

def qDrawSlow(p, q, c, varu):
    T = p.shape[0]
    q_updated = q.copy()
    for s in range(T):
        if q[s] == 0:
            continue
        pr_exp = np.zeros(2)
        # Forward contribution
        if s < T - 1:
            u_ahead = (p[s+1] - p[s]) + c * np.array([1, -1]) - c * q[s+1]
            pr_exp -= (u_ahead ** 2) / (2 * varu)
        # Backward contribution
        if s > 0:
            u_back = (p[s] - p[s-1]) + c * q[s-1] - c * np.array([1, -1])
            pr_exp -= (u_back ** 2) / (2 * varu)
        # Compute log odds
        log_odds = pr_exp[0] - pr_exp[1]
        if abs(log_odds) > 100:
            q_updated[s] = np.sign(log_odds)
        else:
            p_buy = expit(log_odds)
            if np.random.uniform() > p_buy:
                q_updated[s] = -1
            else:
                q_updated[s] = 1
    return q_updated

def RollGibbsBeta(p, pm, q, nSweeps, regDraw=True, varuDraw=True, qDraw=True, varuStart=0, cStart=0, betaStart=0, printLevel=0):
    T = p.shape[0]
    if pm.shape[0] != T or q.shape[0] != T:
        raise ValueError("p, pm, and q must have the same number of rows.")
    
    # Initialize parameters
    varu = varuStart if varuStart > 0 else 0.001
    c = cStart if cStart > 0 else 0.01
    beta = betaStart if betaStart > 0 else 1
    
    # Initialize q if needed
    if qDraw:
        dp = p[1:] - p[:-1]
        q_initial = np.zeros_like(q)
        q_initial[1:] = np.sign(dp)
        q_initial[0] = 1  # or another initial value
        q = np.where(q != 0, q, q_initial)
    
    # Output storage
    parmOut = np.full((nSweeps, 3), np.nan)
    
    for sweep in range(nSweeps):
        dp = p[1:] - p[:-1] if len(p) >1 else np.array([0])
        dq = q[1:] - q[:-1] if len(q) >1 else np.array([0])
        dpm = pm[1:] - pm[:-1] if len(pm) >1 else np.array([0])
        
        if regDraw:
            prior_mu = np.array([0, 1])
            prior_cov = np.diag([1, 2])
            X = np.column_stack((dq, dpm)) if dq.size >0 else np.zeros((0, 2))
            post_mu, post_cov = BayesRegressionUpdate(prior_mu, prior_cov, dp, X, varu)
            # Ensure positive c
            lower = np.array([0, -np.inf])
            upper = np.array([np.inf, np.inf])
            coeffDraw = mvnrndT(post_mu, post_cov, lower, upper)
            c, beta = coeffDraw
        
        if varuDraw:
            u = dp - c*dq - beta*dpm
            prior_alpha = 1e-12
            prior_beta = 1e-12
            post_alpha, post_beta = BayesVarianceUpdate(prior_alpha, prior_beta, u)
            # Sample from inverse gamma
            gamma_var = np.random.gamma(post_alpha)
            varu = post_beta / gamma_var
        
        if qDraw:
            q = qDrawSlow(p, q, c, varu)
        
        parmOut[sweep] = [c, beta, varu]
    
    return parmOut

# Test the code
def test_example():
    np.random.seed(12345)
    nObs = 250
    sdu = 0.2 / np.sqrt(250)
    sdm = 0.3 / np.sqrt(250)
    beta_true = 1.1
    c_true = 0.04
    prob_zero = 0.6
    
    # Generate data
    v = np.random.normal(0, sdm, nObs)
    pm = np.cumsum(v)
    micro_noise = np.random.normal(0, sdu, nObs)
    m = 0
    p = np.zeros(nObs)
    q = np.sign(np.random.uniform(size=nObs) - 0.5)
    z = np.random.uniform(size=nObs)
    q = np.where(z > prob_zero, q, 0)
    
    for t in range(nObs):
        m += beta_true * v[t] + micro_noise[t]
        p[t] = m + c_true * q[t]
    
    # Run Gibbs sampler
    nSweeps = 1000
    nDrop = 200
    parmOut = RollGibbsBeta(p, pm, q, nSweeps, regDraw=True, varuDraw=True, qDraw=True, 
                             varuStart=0.001, cStart=0.01, betaStart=1, printLevel=0)
    parmOut_dropped = parmOut[nDrop:]
    post_means = np.mean(parmOut_dropped, axis=0)
    print("Posterior means (c, beta, varu):", post_means)

test_example()