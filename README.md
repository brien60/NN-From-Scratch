## Overview
- mlp.ipynb contains the code for a basic Multi-layer Perceptron. The forward and backward passes are coded entirely from scratch from mathematical fundamentals, with the only numpy usage being np.random functions. 

- The file also contains the result of an experiment with an MLP model achieving 97.79% accuracy.


## Challenges
- Deriving the back-propagation algorithm for the CNN was a challenge because it was difficult to even denote what value I want to refer to, let alone keep track of how that value influences value(s) that I know the gradient for. 
- It was also difficult to vectorize the back-propagation, but for most of the operations it was much easier to imagine what we do with a patch rather than a single value. 

## Mistakes
- Using reshape instead of transpose when a switching of dimensions was necessary, not a combination or separation.
- Forgetting to clear gradients after each backward pass / before each forward pass.
- Forgot to turn off dropout during validation


## Possible Improvements
- 

