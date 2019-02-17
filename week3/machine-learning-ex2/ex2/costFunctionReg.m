function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
featuresNum = size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hyp = sigmoid(X*theta);
newTheta = theta(2:featuresNum)
J = 1/m * sum(-y .* log(hyp) - (1-y) .* log(1-hyp)) + (lambda/(2*m))*sum(newTheta.^2)

grad(1) =  sum((hyp-y).*X(:,1))/m
for i = 2:featuresNum
	grad(i) = sum((hyp-y).*X(:,i))/m + (lambda/m)*theta(i)
end


% =============================================================

end
