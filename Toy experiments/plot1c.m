% seed for reproductibility
rng(2)

% ==== INITIALIZATION ====
fEvals={};
fVals= {};
Kappas = {};
clear options
strongs = [10;2;0.5;0.1]; %The different values of the strong convexity
cst = .1;
dim = 30;
a = [-(sign(2*rand(dim/2,1)-1)-1)*.5; rand(dim/2,1)];
b = [-(sign(2*rand(dim/2,1)-1)-1)*.5; rand(dim/2,1)];
M = cst.* (2.*rand(dim,dim)-1);
adaptive = 3;
Kmax = 20000;
away = 1;
for i = 1:4
	adaptive = 4;
	strong = strongs(i);
	[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
	fEvals{end+1} = (1:(niter-1))./10^4; % resize the number of iterations
	fVals{end+1}=G(1:niter-1);
	i
	if i ~= 3
		Kappas{end+1} = strcat('$\nu =',num2str(kappa,2),'\; \gamma$ heuristic');
	else i == 3
		Kappas{end+1} = strcat('$\nu=',num2str(kappa,2),'\;  \gamma$ heuristic');
		adaptive = 3;
		strong = strongs(i);
		[G,niter,kappa] = SP_FW(Kmax,strong,M,cst,dim,a,b,away,adaptive);
		fEvals{end+1} = (1:(niter-1))./10^4;
		fVals{end+1}=G(1:niter-1);
		Kappas{end+1} = strcat('$\nu =',num2str(kappa,2),' \gamma = \frac{2}{2+k(t)}$');
	end
end

options.logScale = 2;
options.colors = {'c','b','g',[1 0.5 0],'r'};
options.lineStyles =  {'--','--','-','-','-'};
options.markers ={'s','o','d','p','d'};
options.markerSpacing = [2000 1500
	2000 200
	2000 500
	2000 800
	2000 400
	2000 500];
options.lineSize = 8;
options.legend =Kappas;
options.ylabel = 'Duality Gap';
options.xlabel = 'Iteration ($\times 10^4$)';
options.legendFontSize=14;
options.labelLines = 0;
options.labelRotate = 1;
options.xlimits = [0 Kmax./10^4];
options.ylimits = [10^-5 .1];
figure;
prettyPlot(fEvals,fVals,options);
