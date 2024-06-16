import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self):
        """
        Initialize the Geometric Brownian Motion (GBM) Monte Carlo simulator.

        Attributes:
        S0 (float): Initial value of the process.
        n_sym (int): Number of symbols.
        N (int): Number of time steps.
        T (float): Total time.
        drift (float or callable): Drift coefficient (mu).
        volatility (float or callable): Volatility coefficient (sigma).
        L (numpy.ndarray): Cholesky decomposition of the covariance matrix.
        paths (numpy.ndarray): Simulated paths.
        n_assets (int): Number of assets.
        """
        self.S0 = None
        self.n_sym = None
        self.N = None
        self.T = None
        self.drift = None
        self.volatility = None
        self.L = None
        self.paths = None
        self.n_assets = None

    def fit(self, data, date_range=1):
        """
        Calibrate the model parameters to the given data.

        Parameters:
        data (numpy.ndarray): Observed values.
        date_range (int): Range of data date for calibration.

        This method estimates the volatility as the covariance of returns
        and the drift as the mean of returns.
        """
        returns = np.diff(np.log(data), axis=1)
        self.volatility = np.cov(returns) * date_range ** 2
        self.drift = np.mean(returns, axis=1) * date_range
        self.L = np.linalg.cholesky(self.volatility)
        self.S0 = data[:, -1]

    def set_params(self, drift, volatility):
        """
        Set the drift and volatility parameters.

        Parameters:
        drift (numpy.ndarray): Drift coefficients.
        volatility (numpy.ndarray): Volatility coefficients.

        This method checks if the shapes of the drift and volatility parameters match,
        and sets the Cholesky decomposition of the volatility matrix.
        """
        if drift.shape[0] == volatility.shape[0]:
            self.drift = np.array(drift)
            self.volatility = np.array(volatility)
            self.L = np.linalg.cholesky(self.volatility)
        else:
            raise ValueError("Please provide appropriate drift and volatility parameters")

    def simulate_paths(self, S0=None, N=100, T=1, n_sym=10):
        """
        Simulate Monte Carlo paths for the multi-dimensional GBM.

        Parameters:
        S0 (numpy.ndarray, optional): Initial values of the process.
        N (int): Number of time steps.
        T (float): Total time.
        n_sym (int): Number of Monte Carlo paths.

        This method generates simulated paths using the provided parameters.
        """
        self.N = N
        self.T = T
        self.n_sym = n_sym
        self.n_assets = len(self.drift)
        paths = np.zeros((self.n_sym, N + 1, self.n_assets))
        if S0 is not None:
            paths[:, 0, :] = S0
        else:
            paths[:, 0, :] = self.S0

        dt = T / N
        for i in range(1, N + 1):
            Z = np.random.standard_normal((self.n_sym, self.n_assets))
            dW = np.dot(Z, self.L.T)
            paths[:, i, :] = paths[:, i - 1, :] * (1 + self.drift * dt + np.sqrt(dt) * dW)

        self.paths = paths

    def plot_paths(self, n_plots=1):
        """
        Plot the simulated paths.

        Parameters:
        n_plots (int): Number of paths to plot.

        This method generates a plot of the simulated paths.
        """
        if self.paths is None:
            raise ValueError("No paths simulated. Please run simulate_paths() first.")
        if self.n_sym < n_plots:
            raise ValueError("Number of generated simulations is too small. Please generate paths once again with different value of parameter n_sym") 

        plt.figure(figsize=(10, 6))
        for i in range(n_plots):
            plt.plot(np.linspace(0, self.T, self.N + 1), self.paths[i], lw=1.5, label=f'Path {i}')
        plt.title('Monte Carlo Simulation of GBM Paths')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.show()



# Example Usage
if __name__ == "__main__":
    # Sample data for calibration (e.g., stock prices)
    # Provided as numpy array, each row contains time series
    # of prices of given commodity
    data = np.array([[100, 102, 101, 105, 110, 108, 115],
                     [103, 104, 105, 102, 108, 110, 109]])

    # Initialize SDE with initial guess for drift and volatility
    p = GBM()

    # Calibrate the model to the data (drift, volatility, correlation)
    p.fit(data)

    # Simulate the paths
    p.simulate_paths()

    # Plot the simulated paths
    p.plot_paths()
