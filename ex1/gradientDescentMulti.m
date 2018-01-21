function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % printf('Running iteration %i\n', iter);
    temp = theta;

    %printf('theta : %f\n', temp, temp);

    for j=1 : length(temp(:,1)) ,

      h=0;
      for i=1 : m,
        h =  h + ( (X(i,:) * theta - y(i))*X(i,j));
      end;

    temp(j, 1) = theta(j, 1) - alpha * 1 / m * h;

    end;


    % update theta
    theta = temp;


    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

    % fprintf('J : %f\n',J_history(iter));

end

end
