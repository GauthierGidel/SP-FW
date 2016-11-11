function [ f ] = objective_function_SP( w, w_y,  ell )
%OBJECTIVE_FUNCTION returns the function value f(alpha)
%   returns the SVM dual objective function  f(alpha) , which is
%   -equation (4) in the paper.
%   The arguments are w = A*alpha and b_alpha = b'*alpha.

    f = -w'*w_y + ell;

end % objective_function

