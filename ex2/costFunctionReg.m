function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculate cost function
sum1 = 0;
for i = 1 : m,
  h = sigmoid(theta' * X(i,:)');
  sum1 = sum1 + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
endfor

% Calculate regularization term
sum2 = 0;
for j = 2 : n,
   sum2 = sum2 + theta(j) * theta(j);
endfor
J = sum1 / m + lambda / (2 * m) * sum2;


% Calculate gradient for theta 0
for i = 1 : m,
  h = sigmoid(theta' * X(i,:)');
  grad(1) = grad(1) + (h - y(i)) * X(i, 1);
endfor
grad(1) = grad(1) / m;

% Calculate gradient for ttheta j, j > 1
for j = 2 : n,
  for i = 1 : m,
    h = sigmoid(theta' * X(i,:)');
    grad(j) = grad(j) + (h - y(i)) * X(i, j);
  endfor
  grad(j) = grad(j) / m + lambda / m * theta(j);
endfor

% =============================================================

end
