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
strongs = [11;12;21;40]; %The different values of the strong convexity
cst = .1;
dim = 30;
a = [rand(dim,1)./2 + .25];
b = [rand(dim,1)./2 + .25];
M = cst.* (2.*rand(dim,dim)-1);
Kmax = 40000;
away = 0;

for i = 1:4
	adaptive = 1;
	strong = strongs(i);
	[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
	fEvals{end+1} = (1:(niter-1))./10^4; % resize the number of iterations
	fVals{end+1}=G(1:niter-1);
	i
	if i == 1
		adaptive = 3;
		Kappas{end+1} = strcat('$\nu =',num2str(kappa,1),' \;\gamma$ adaptive');
		strong = strongs(1);
		[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
		fEvals{end+1} = (1:(niter-1))./10^4;
		fVals{end+1}=G(1:niter-1);
		Kappas{end+1} = strcat('$\nu =',num2str(kappa,1),' \;\gamma = \frac{2}{2+t}$');
	else
		Kappas{end+1} = strcat('$\nu=',num2str(kappa,2),' \;\gamma$ adaptive');
	end
end

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
options.legend = Kappas;
options.ylabel = 'Duality Gap';
options.xlabel = 'Iteration ($\times 10^4$)';
options.legendFontSize=14;
options.labelLines = 0;
options.labelRotate = 1;
options.xlimits = [0 Kmax./10^4];
options.ylimits = [10^-6 5];
figure;
prettyPlot(fEvals,fVals,options);
