%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solving constraint optimization problem using Newton-Raphson method
%
% Eric Xing
% UC Berkeley
% Jan 15, 2002
%%%%%%%%%%%%%%%%%%%%%%%%%%
% revised by Lingxiao Zhao
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% modified to treat 0's in feature vectors as absent values.  

function A = PgdmAlgLearnDiagA(data, C, gamma)
% Input:
%   data: n*d X matrix
%   C: 3 column matrix
%        column 1, 2: index of constrained points.  Indexes between 1 and n
%        column 3: 1 if points are similar, -1 if dissimilar
%   gamma: default is 1
%

size_data=size(data);
N=size_data(1);
d=size_data(2);

a=ones(d,1);
X=data;

fudge = 0.000001;
threshold1 = 0.001;
reduction = 2;

% suppose d is a column vector
% sum(d'Ad) = sum(trace(d'Ad)) = sum(trace(dd'A)) 
%           = trace(sum(dd'A) = trace(sum(dd')A)

s_sum = zeros(1,d);
d_sum = zeros(1,d);

% LZ speed optimization (initial speed)
S = C(C(:,3) == 1, [1 2]);
D = C(C(:,3) == -1, [1 2]);
for sim_pair = S'
    i = sim_pair(1);
    j = sim_pair(2);
    d_ij = X(i,:) - X(j,:) + ~(X(i,:) | X(j,:));
    s_sum = s_sum + d_ij.^2;
end

for diff_pair = D'
    i = diff_pair(1);
    j = diff_pair(2);
    d_ij = X(i,:) - X(j,:) + ~(X(i,:) | X(j,:));
    d_sum = d_sum + d_ij.^2;
end

      
tt=1;
error=1;
% BP added outer loop constraint, it got stuck in an infinite loop once?
while error > threshold1 && tt < 100

  [fD0, fD_1st_d, fD_2nd_d] = D_constraint_LZ(X, D, a, N, d);
  obj_initial =  s_sum*a + gamma*fD0; 
  fS_1st_d = s_sum;                    % first derivative of the S constraints

  Gradient = fS_1st_d - gamma*fD_1st_d;            % gradient of the objective
  Hessian = - gamma*fD_2nd_d + fudge*eye(d);      % Hessian of the objective
%     invHessian = inv(Hessian);
%     step = invHessian*Gradient';
%   
  % BP - Matlab suggests using A\b instead of inv(A)*b
  step = Hessian\Gradient';
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Newton-Raphson update
  % search over optimal lambda
  
  lambda=1;        % initial step-size
  t=1;             % counter
  atemp = a - lambda*step;
  atemp = max(atemp, 0);

  obj = s_sum*atemp + gamma*D_objective_LZ(X, D, atemp, N, d);
  
  % BP - this doesn't always work?
  obj_previous = obj * 1.1;    %  just to get the while loop started  

  % BP without initialization this can bug out
  a_previous = atemp;
  
  while obj < obj_previous
    lambda_previous = lambda;
    obj_previous = obj;
    a_previous = atemp;
    lambda = lambda/reduction; 
    atemp = a - lambda*step;
    atemp = max(atemp, 0);
    obj = s_sum*atemp + gamma*D_objective_LZ(X, D, atemp, N, d);
    t=t+1;   % inner counter
  end    % line search for lambda that minimize obj
  
  a = a_previous;
  
  error = abs((obj_previous - obj_initial)/obj_previous);
  tt = tt + 1;   % outer counter
  
end
a(a~=0);
A = a;
%A=diag(a);  
