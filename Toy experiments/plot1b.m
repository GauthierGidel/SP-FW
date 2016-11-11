% the library prettyplot needs to be loaded
addpath('~/Documents/MATLAB/export_fig-master')
addpath('~/Documents/MATLAB/prettyPlot')
% seed for reproductibility
rng(2)

% ==== INITIALIZATION ====
fEvals={};
fVals= {};
Kappas = {};
clear options
strongs = [50;60;100;300]; %The different values of the strong convexity
cst = .1;
dim = 30;
a = [-(sign(2*rand(dim/2,1)-1)-1)*.5; rand(dim/2,1)];
b = [-(sign(2*rand(dim/2,1)-1)-1)*.5; rand(dim/2,1)];
M = cst.* (2.*rand(dim,dim)-1);
adaptive = 3;
Kmax = 40000;
away = 1;
adaptive = 3;

strong = strongs(1);
[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
fEvals{end+1} = (1:(niter-1))./10^4; % resize the number of iterations
fVals{end+1}=G(1:niter-1);
Kappas{end+1} = strcat('$\nu =',num2str(kappa,1),' \gamma = \frac{2}{2+k(t)}$');
for i = 1:4
	adaptive = 1;
	strong = strongs(i);
	[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
	fEvals{end+1} = (1:(niter-1))./10^4;
	fVals{end+1}=G(1:niter-1);
	i
	if i == 1
		Kappas{end+1} = strcat('$\nu =',num2str(kappa,1),' \gamma$ adaptive');
	else
		Kappas{end+1} = strcat('$\nu=',num2str(kappa,2),'$ $  \gamma$ adaptive');
	end
end

options.legendLoc = 'NorthEast';
options.logScale = 2;
options.colors = {'c','b','g',[1 0.5 0],'r'};
options.lineStyles =  {'--','--','-','-','-'};
options.markers ={'s','o','d','p','d'};
options.markerSpacing = [4000 1500
	4000 1000
	4000 500
	4000 800
	4000 200
	4000 500];
options.lineSize = 8;
options.legend =Kappas;
options.ylabel = 'Duality Gap';
options.xlabel = 'Iteration ($\times 10^4$)';
options.legendFontSize=14;
options.labelLines = 0;
options.labelRotate = 1;
options.xlimits = [0 Kmax./10^4];
options.ylimits = [10^-3 0.2];
figure;
prettyPlot(fEvals,fVals,options);
