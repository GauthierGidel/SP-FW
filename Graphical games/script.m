clear W
rng(2);
niter = 1000;
BIG = zeros(1000,2);
iter =1;
nn = [2.^(2:8)];
dd = nn .* (nn-1)./2;
for n = nn
	n; %number of students
	% We set random benefit for each student in each university
	mu = rand(n,1); % average value of the student
	C = mu*ones(1,2) + 0.05*randn(n,2);
	ranking = zeros(n,n);
	d = n*(n-1)/2;
	W = zeros(d);
    fprintf('initialize rankings\n')
	for i = 1:n
		ranking_i = rand(1,n-1);
		ranking(i,1:i-1) = ranking_i(1:i-1); %random ranking for each student
		ranking(i,i+1:n) = ranking_i(i:n-1); %%random ranking for each student
    end
    fprintf('set matrix M\n')
	for i = 1:n-1
		for j = i+1:n
			for k = i+1:n
				if (ranking(i,j) > ranking(i,k)) % university 1 wins
					ij = n*(i-1) + i*(1-i)/2 + j - i;
					ik = n*(i-1) + i*(1-i)/2 + k - i;
					W(ij,ik) = C(i,1);
				elseif (ranking(i,j) < ranking(i,k)) % university 2 wins
					ij = n*(i-1) + i*(1-i)/2 + j - i;
					ik = n*(i-1) + i*(1-i)/2 + k - i;
					W(ij,ik) = -C(i,2);
				end

			end

		end
		for j = 1:i-1
			for k = i+1:n
				if (ranking(j,i) > ranking(i,k)) % university 1 wins
					ji = n*(j-1) + j*(1-j)/2 + i - j;
					ik = n*(i-1) + i*(1-i)/2 + k - i;
					W(ji,ik) = C(i,1);
					W(ik,ji) = -C(i,2);
				elseif (ranking(j,i) < ranking(i,k)) % university 2 wins
					ji = n*(j-1) + j*(1-j)/2 + i - j;
					ik = n*(i-1) + i*(1-i)/2 + k - i;
					W(ji,ik) = -C(i,2);
					W(ik,ji) = C(i,1);
				end

			end

		end
	end
	epsilon = .01;
	[X,Y,gkhistory] = spfw(W,n,epsilon,niter);
	BIG(:,iter) = gkhistory;
	iter = iter +1;
end
plot(dd,times)