function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k=1 : K
  nb_assign=0;
  new_centroid = zeros(1, n);
  %fprintf("computing new centroid %d\n", k)
  for i=1:m
    if idx(i) == k
      %fprintf("element %d, idx = %d\n", i, idx(i));
      %fprintf("adding (%f, %f) ", X(i, :));
      nb_assign = nb_assign +1;
      new_centroid = new_centroid + X(i, :);
      %fprintf("new centroid = (%f, %f) \n", new_centroid);
    endif
  end
  centroids(k, :) = (1.0 / nb_assign) * new_centroid;

end






% =============================================================


end
