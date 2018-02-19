function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i=1:m,
  r = X(i,:);
  z = theta'*r';
  J = J -1 * y(i, 1) * log (sigmoid(z)) - (1-y(i,1)) * log (1 - sigmoid(z));
end;

J = J / m;

nb_grads = (size(grad))(1, 1);
for t=1:nb_grads,
  cur_grad = 0;
  for i=1:m,
    z = theta'*X(i,:)';
    cur_grad = cur_grad + ((sigmoid(z))-y(i,1))*X(i, t);
  end;
  grad(t, 1) = cur_grad / m;
end;





% =============================================================

end
