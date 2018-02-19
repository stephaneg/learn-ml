function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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



for i=1:m,
  z = theta'*X(i,:)';
  J = J + y(i, 1) * log (sigmoid(z)) + (1-y(i,1)) * log (1 - sigmoid(z));
end;

tmp=0;
for i=2:size(theta, 1),
  tmp = tmp + theta(i, 1)^2;
end;
J = ((-1.0 * J) + (0.5 * lambda * tmp)) / m;

nb_grads = (size(grad))(1, 1);
for t=1:nb_grads,
  cur_grad = 0;
  for i=1:m,
    z = theta'*X(i,:)';
    cur_grad = cur_grad + ((sigmoid(z))-y(i,1))*X(i, t);
  end;
  if (t==1)
    grad(t, 1) = cur_grad / m;
  else
    grad(t,1) = (cur_grad / m) + (lambda / m)*theta(t,1);
end;




% =============================================================

end
