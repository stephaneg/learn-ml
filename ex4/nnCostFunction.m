function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% init
newY = zeros(m, num_labels);
for i =1:m
  newY(i, y(i))=1;
end

%% compute a2, a3 : forward propagation
%fprintf("size of X is : %d %d\n", size(X));
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
%fprintf("size of Theta1 is : %d %d\n", size(Theta1));
%fprintf("size of Theta2 is : %d %d\n", size(Theta2));
%fprintf("size of a2 is : %d %d\n", size(a2));
z3 = [ones(m, 1) a2] * Theta2';
a3 = sigmoid(z3);
%fprintf("size of a3 is : %d %d\n", size(a3));
%fprintf("size of newY is : %d %d\n", size(newY));

%% compute backward propagation
for i=1:m
  delta3 = zeros(size(newY, 2), 1);
  delta2 = zeros(size(Theta1, 2), 1);

  delta3 = a3(i,:)' - newY(i, :)';
  newA2 =[1 a2(i, :)]';
  %fprintf("size of newA2 is : %d %d\n", size(newA2));
  delta2 = (Theta2' * delta3) .* (newA2 .* (ones(size(newA2))-newA2));

  %fprintf("size of Theta2' is : %d %d\n", size(Theta2'));

  %fprintf("size of delta3 is : %d %d\n", size(delta3));
  %fprintf("size of delta2 is : %d %d\n", size(delta2));

  %fprintf("size of Theta2_grad is : %d %d\n", size(Theta2_grad));
  %fprintf("size of Theta1_grad is : %d %d\n", size(Theta1_grad));
  %fprintf("size of delta2 trunc is : %d %d\n", size(delta2(2:end)));
  %fprintf("size of X(1) trunc is : %d %d\n", size(a1(i,:)'));


  Theta2_grad = Theta2_grad + delta3*newA2';
  Theta1_grad = Theta1_grad + delta2(2:end)*a1(i,:);

end

%fprintf("size of Theta2_grad is : %d %d\n", size(Theta2_grad));
%fprintf("size of Theta1_grad is : %d %d\n", size(Theta1_grad));
Theta2_grad = (1.0/m) * Theta2_grad  ;
Theta1_grad = (1.0/m) * Theta1_grad  ;



%% compute cost function
for i=1:m
  for l = 1:num_labels
    J = J - newY(i, l) * log(a3(i, l)) - ( (1-newY(i,l))*log(1-a3(i,l))  ) ;
  end
end
J =  (1.0/m) * J;


%% regularization
temp1 = Theta1;
temp1(:, 1) = zeros(size(Theta1, 1), 1);
temp2 = Theta2;
temp2(:, 1) = zeros(size(Theta2, 1), 1);
reg = 0;

for u=1:size(temp1, 1)
  for t=1:size(temp1,2)
    reg = reg + temp1(u, t) ^2;
  end
end

for u=1:size(temp2, 1)
  for t=1:size(temp2,2)
    reg = reg + temp2(u, t) ^2;
  end
end

J = J + 0.5 * lambda * reg / m ;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
