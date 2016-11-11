function [X,Y,gkhistory] = spfw(W,n,epsilon,niter)
% 
% function [X,Y,gkhistory] = spfw(W,n,epsilon)
%
% n=4; N=nchoosek(n,2); W=rand(N,N); W=max(W,W');
% 
% 

if (nargin<3)
  epsilon=0.001;
end
if (nargin<4)
  niter = 1000;
end

N  = nchoosek(n,2);
X  = edmonds2(rand(n,n),1);
Y  = X;
gk = 2*epsilon;
k  = 1;
gkhistory = [];

% while (gk>epsilon)  % Disable stopping criterion for better plotting

for ii=1:niter
  x = zeros(N,1);
  y = zeros(N,1);
  ind = 1;
  for i=1:n
    for j=i+1:n
      x(ind) = X(i,j);  
      y(ind) = Y(i,j);  
      ind = ind+1;
    end
  end
  
  gy = W*x;
  gx = (y'*W)';
  
  GX = zeros(n,n);
  GY = zeros(n,n);
  ind = 1;
  for i=1:n
    for j=i+1:n
      GX(i,j)=gx(ind);
      GY(i,j)=gy(ind);
      ind = ind+1;
    end
  end
  GX =GX + GX';
  GY = GY + GY';
  SX = edmonds2(max(max(GX))-GX,1);
  SY = edmonds2(max(max(GY))-GY,1);
  
  sx = zeros(N,1);
  sy = zeros(N,1);
  ind = 1;
  for i=1:n
    for j=i+1:n
      sx(ind) = SX(i,j);  
      sy(ind) = SY(i,j);  
      ind = ind+1;
    end
  end
  
  gk = dot(x-sx,gx)+dot(y-sy,gy);
  gkhistory = [gkhistory; gk];
  gamma = 2/(2+k);
  x = (1-gamma)*x+gamma*sx;
  y = (1-gamma)*y+gamma*sy;
  X = zeros(n,n);
  Y = zeros(n,n);
  ind = 1;
  for i=1:n
    for j=i+1:n
      X(i,j) = x(ind);  
      Y(i,j) = y(ind);
      ind = ind+1;
    end
  end
  X = X + X';
  Y = Y + Y';
  
  k = k+1;
 
end

