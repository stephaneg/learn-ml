function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %printf('Running iteration %i\n', iter);
    temp0 = theta(1, 1);
    temp1 = theta(2, 1);

    %printf('theta 0 : %f, theta 1 : %f\n', temp0, temp1);



    % compute part. der. theta0
    der_theta0 = 0;
    for i=1:m,
      h = theta(1)+theta(2)*X(i, 2);
      der_theta0 = der_theta0 + (h-y(i));
    end;
    der_theta0 = der_theta0 / m;

    % compute part. der. theta1
    der_theta1 = 0;
    for i=1:m,
      h = theta(1)+theta(2)*X(i, 2);
      der_theta1 = der_theta1 + (h-y(i))*X(i, 2);
    end;
    der_theta1 = der_theta1 / m;

    %printf('der theta 0 : %f, der theta 1 : %f i\n', der_theta0, der_theta1);

    temp0 = theta(1, 1) - alpha * der_theta0;
    temp1 = theta(2, 1) - alpha * der_theta1;

    %printf('theta 0 : %f, theta 1 : %f\n', temp0, temp1);

    % update theta
    theta(1, 1) = temp0;
    theta(2, 1) = temp1;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    %fprintf('J : %f\n',J_history(iter));

end

end
