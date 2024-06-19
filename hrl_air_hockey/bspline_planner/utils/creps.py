import numpy as np
from scipy.optimize import minimize


class ConstrainedREPS:
    """
    Episodic Relative Entropy Policy Search algorithm with constrained policy update.

    """
    def __init__(self, distribution, eps, kappa, x_con=np.array([0.57, 1.3]), y_con=np.array([-0.48535, 0.48535]), v_con=np.array([0, .3])):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa (float): the maximum admissible value for the entropy decrease
                between the new distribution and the 
                previous one at each update step. 

        """
        self._eps = eps
        self._kappa = kappa
        self.distribution = distribution
        
        self.x_constraints = x_con
        self.y_constraints = y_con
        self.v_constraints = v_con

    def _update(self, Jep, theta, context):
        eta_start = np.ones(1)

        res = minimize(ConstrainedREPS._dual_function, eta_start,
                       jac=ConstrainedREPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self._eps, Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)

        self.distribution.con_wmle(theta, d, self._eps, self._kappa)

        # clip mu (x and y component)
        self.distribution._mu[0] = np.clip(self.distribution._mu[0], self.x_constraints[0], self.x_constraints[1])
        self.distribution._mu[1] = np.clip(self.distribution._mu[1], self.y_constraints[0], self.y_constraints[1])
        self.distribution._mu[2] = np.clip(self.distribution._mu[1], self.v_constraints[0], self.v_constraints[1])

    @staticmethod
    def _dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J
        sum1 = np.mean(np.exp(r / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])