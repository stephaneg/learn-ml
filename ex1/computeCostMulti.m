function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

nb = length(X(1,:));

J = (1/2*m) * (((X * theta)-y)'*((X * theta)-y));


for i=1:m,
  h = 0;
  for j = 1:nb,
    h = h + theta(j, 1) * X(i, j);
  end ;
  J = J + (h-y(i))^2;

  % h = theta(1)+theta(2)*X(i, 2);
  % J = J + (h-y(i))^2;
end;

J = J / (2*m);

% =========================================================================

end
