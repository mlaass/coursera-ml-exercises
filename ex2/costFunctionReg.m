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

ht = sigmoid(X *theta);
err = -y .* log(ht) - (1 - y) .* log(1 - ht);
delta = (ht - y) .* X;
delta_reg = (lambda/m) * theta;

reg = [0; ones(size(theta)-1, 1)];
thsqr = (theta .* reg) .^2;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = sum(err)/m + lambda * sum(thsqr)/(2*m);
grad = (sum(delta ) /m) + (reg .* delta_reg)';



% =============================================================

end
