import torch
import math
import gpytorch
from matplotlib import pyplot as plt

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_Y, likelihood):
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Training data is 100 points in [0,1] inclusive regularly spaced
train_X = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_Y = torch.sin(train_X * (2 * math.pi)) + torch.randn(train_X.size()) * math.sqrt(0.04)
#train_X = torch.from_numpy(np.array([[0.1], [0.5], [0.9]], dtype=np.float32))
#train_Y = torch.from_numpy(np.array([0.5, -0.3, 0.4], dtype=np.float32))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_X, train_Y, likelihood)

model.train()
likelihood.train()

# Define the Marginal Log-Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Use the Adam optimizer with learning rate 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

training_iter = 50 # Train for 50 iterations

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, train_Y)
    loss.sum().backward()
    optimizer.step()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
model.eval()
likelihood.eval()

# Sample test points
test_X = torch.linspace(0, 1, 51)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_X))
    mean_ = observed_pred.mean
    std_ = observed_pred.stddev
    lower, upper = observed_pred.confidence_region()

print("Test X:", test_X)
print("Predicted Mean:", mean_)
print("Confidence Interval:", lower, upper)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_X.numpy(), train_Y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_X.numpy(), mean_.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_X.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
    # Plot test predictions with 1 standard deviation
    plt.errorbar(test_X.numpy(), mean_.numpy(), yerr=std_.numpy(), fmt='o', alpha=0.5)

    ax.set_ylim([-3, 3])
    ax.legend(['Trained Data', 'Test Mean', 'Test 95% Confidence Interval', 'Test 1 std'])
    plt.show()