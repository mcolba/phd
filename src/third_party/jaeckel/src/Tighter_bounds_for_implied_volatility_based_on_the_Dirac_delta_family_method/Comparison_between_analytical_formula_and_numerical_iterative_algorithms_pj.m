%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Tighter bounds for implied volatility based on the Dirac delta family method
% Author: Zhenyu Cui,Yanchu Liu and Yuhang Yao
% Date: August 13, 2023
% Note: This document corresponds to section 3.2 of the paper, with figure(8)-(10) and table 2
% MATLAB version: Matlab2020a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adapted to GNU Octave by Peter Jäckel, 2023-11-09
% Adapted to using
%    "letsberational.oct" -> "LetsBeRational.xll"
% or via
%    "letsberational_via_swig.oct" -> "LetsBeRational.xll"
% in GNU Octave by Peter Jäckel, 2023-11-09
%
% "Let's Be Rational" (LetsBeRational.xll) provides
%    the Black function as:                      Black(double F, double K, double sigma, double T, double q /* q=±1 */)
%    the Black implied volatility function as:   ImpliedBlackVolatility(double value, double F, double K, double T, double q /* q=±1 */)
%

clear; clc; format long g; clear all;
addpath(fileparts(mfilename('fullpath')));

n_per_dimension = 39;

figure_width=1400;
figure_height=350;
figure_head_height=35;
figure_height_offset=50;

%if (isunix())
%  graphics_toolkit("gnuplot")
%endif

start_tic = tic;

global isOctave use_letsberational_black use_letsberational_implied_black_volatility use_swig_generated_letsberational_api keep_vector_form_of_option_function_calls;

use_letsberational_black = 1;
use_letsberational_implied_black_volatility = 1;
keep_vector_form_of_option_function_calls = 1;
use_swig_generated_letsberational_api = 0;

printf("\nComparison between analytical formula and numerical iterative algorithms (section 3.2).\n");

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if (ispc())
  cpu = winqueryreg('HKEY_LOCAL_MACHINE', 'HARDWARE\DESCRIPTION\System\CentralProcessor\0', 'ProcessorNameString');
else
  cpu = "";
endif

if (isOctave)
  this_program = [ "GNU Octave " version() ];
  saved_warning_state = warning();
  warning("off","Octave:shadowed-function");
% pkg load statistics % implicitly loaded by financial.
  pkg load financial;
  warning(saved_warning_state);
else
  use_letsberational_black = 0;
  use_letsberational_implied_black_volatility = 0;
  [v d] = version();
  this_program = [ "Matlab " v ];
endif

function [b] = use_letsberational()
  global isOctave use_letsberational_black use_letsberational_implied_black_volatility;
  b = isOctave && ( use_letsberational_black || use_letsberational_implied_black_volatility );
end

if (use_letsberational())
  if (use_swig_generated_letsberational_api)
    printf("\nUsing Let's Be Rational (via SWIG-generated Octave interface).\n\n");
    letsberational_via_swig
	letsberational_xll_api = "SWIG-generated";
  else
    printf("\nUsing Let's Be Rational (via hand-written Octave interface).\n");
    letsberational
	letsberational_xll_api = "hand-written";
  endif
endif

% Replacement for European_call(S,K,r,T,sigma,q)
function [value] = BlackScholesCall(S,K,r,T,sigma,q)
  global use_letsberational_black;
  if (use_letsberational_black)
    F = S*exp((r-q)*T);
    value = exp(-r*T) * Black(F,K,sigma,T,1.0);
  else
    value = European_call(S,K,r,T,sigma,q);
  endif
end

% Replacement for `` blsimpv(S,K,r,T,value[,,'Method','jackel2016' 'Class',{'Call'}]); % Jaeckel’s method '' -- Note that 'jackel2016' and 'Call' are the default values in MATLAB.
function [sigma] = LetsBeRationalForCall(S,K,r,T,value)
  global use_letsberational_implied_black_volatility;
  if (use_letsberational_implied_black_volatility)
    inv_df = exp(r*T); 
    sigma = ImpliedBlackVolatility(value*inv_df,S*inv_df,K,T,1.0);
  else
    % Under MATLAB, 'Method','jackel2016' is the default.
    % Octave's blsimpv() implementation does not allow for 'Method','jackel2016' or 'Method','search' -  hence, under Octave,
    % we automatically fall back to its fzero based implementation to solve by gradient (Newton or quasi-Newton).
    sigma = blsimpv(S,K,r,T,value);
  endif
end

global colours;
colours = [[0,1,1];[0,0,1];[1,0,0];[1,0.5,0];[1,1,0];[0,0.95,0]];
function [cm] = mycolourmap(n,gamma=1.25)
  global colours;
  cm = zeros(n,3);
  n_colours = rows(colours);
  for i = 1:n
	w = 1+((i-1.0)/(n-1.0))^gamma*(n_colours-1);
	l = min(floor(w),n_colours-1);
	cm(i,:) = (l+1-w)*colours(l,:) + (w-l)*colours(l+1,:);
  endfor
end

function [h] = heatmapplot(X,Y,data,x_label,y_label,the_title)
  global isOctave;
  if (isOctave)
%   clims = [min(min(data)),max(max(data))];
    h = imagesc(X,Y,data);%,clims);
    colorbar();
  else
    % The 'heatmap' function is not yet implemented in Octave.
    x = num2cell(round(X,2)); % Matlab syntax for rounding
    y = num2cell(round(Y,0)); % Matlab syntax for rounding
%   z_min = min(min(data));
    z_max = max(max(data));
    h = heatmap(x,y,data,'ColorLimits',[-z_max z_max]);
    colormap jet;
  endif
  xlabel(x_label);
  ylabel(y_label);
  title(the_title);
end

function [s] = TrueFalse(b)
  if (b)
    s = "True";
  else
    s = "False";
  endif
end

printf("Configuration\n");
printf("=============\n\n");
printf("   %-117s:  %s\n","CPU",CPUName());
printf("   %-117s:  %s\n","Matlab/GNU Octave?",this_program);
printf("   %-117s:  %s\n","Using \"Let's Be Rational\"'s Black() function",TrueFalse(use_letsberational_black));
printf("   %-117s:  %s\n","Using \"Let's Be Rational\"'s ImpliedBlackVolatility() function",TrueFalse(use_letsberational_implied_black_volatility));
if (use_letsberational())
  printf("   %-117s:  %s\n","Octave interface to \"Let's Be Rational\"'s functions (in LetsBeRational.xll)",letsberational_xll_api);
  printf("   %-117s:  %s\n","Using SWIG-generated Octave interface to \"Let's Be Rational\"'s functions",TrueFalse(use_swig_generated_letsberational_api));
endif
printf("   %-117s:  %s\n","Keeping vectorised evaluation [where applicable, thus using 'European_call()' function in DBNR, (2), and (2)* method]",TrueFalse(keep_vector_form_of_option_function_calls));
printf("\n");

% initial parameter
S0 = 100;r = 0.03;q = 0;

for type = 1:3

%type = 1;%1,2,3

switch type
    case 1  % figure(8) and table 2
% generate the three dimension dataset
num_K = n_per_dimension; num_tau = num_K; num_sigma = num_K;
num_vols = (num_K+1)*(num_tau+1)*(num_sigma+1);
printf("Preparing data for table 2 (page 15) and figure 8 (page 25) for %g strikes, %g times-to-expiry, and %g input volatilities (%g in total).\n\n",num_K+1,num_tau+1,num_sigma+1,num_vols);
K0 = 105:(800-105)/num_K:800;
tau0 = 0.01:(2-0.01)/num_tau:2;
sigma0 = 0.01:(0.99-0.01)/num_sigma:0.99;
[x,y,z] = ndgrid(K0,tau0,sigma0);
K = reshape(x,1,num_vols);
tau = reshape(y,1,num_vols);
sigma = reshape(z,1,num_vols);
C_real = linspace(0,0,num_vols);
for i = 1:num_vols
  C_real(i) = BlackScholesCall(S0,K(i),r,tau(i),sigma(i),q);
endfor
%rel_diffs = European_call(S0,K,r,tau,sigma,q)./C_real-1; printf("Minimum / maximum relative difference: %g / %g\n",min(rel_diffs),max(rel_diffs));
%C_real = European_call(S0,K,r,tau,sigma,q);
%
% filter out option prices that are too small to match market prices
%
printf("Filtering option values below 1E-20...\n");
temp0 = find(C_real<1e-20);
C_real(temp0) = [];
tau(temp0) = [];
K(temp0) = [];
sigma(temp0) = [];
printf("Retained %g option values.\n",columns(C_real));

    case 2 % figure(9)
sigma = 0.2;
num_K = n_per_dimension; num_tau = num_K; num_sigma = 0;
printf("Preparing data for figure 9 (page 26) for %g strikes, %g times-to-expiry, and input volatility sigma = %g.\n\n",num_K+1,num_tau+1,sigma);
tau0 = 0.1:(2-0.1)/num_tau:2;
K0 = 105:(180-105)/num_K:180;
[x,y] = ndgrid(K0,tau0);
num_vols = (num_K+1)*(num_tau+1);
K = reshape(x,1,num_vols);
tau = reshape(y,1,num_vols);
C_real = linspace(0,0,num_vols);
for i = 1:num_vols
  C_real(i) = BlackScholesCall(S0,K(i),r,tau(i),sigma,q);
endfor
%rel_diffs = European_call(S0,K,r,tau,sigma,q)./C_real-1; printf("Minimum / maximum relative difference: %g / %g\n",min(rel_diffs),max(rel_diffs));
%C_real = European_call(S0,K,r,tau,sigma,q);
temp0 = find(C_real<1e-20);
C_real(temp0) = [];
tau(temp0) = [];
K(temp0) = [];

    case 3 % figure(10)
sigma = 0.8;
num_K = n_per_dimension; num_tau = num_K; num_sigma = 0;
printf("Preparing data for figure 10 (page 26) for %g strikes, %g times-to-expiry, and input volatility sigma = %g.\n",num_K+1,num_tau+1,sigma);
tau0 = 0.1:(2-0.1)/num_tau:2;
K0 = 105:(800-105)/num_K:800;
[x,y] = ndgrid(K0,tau0);
num_vols = (num_K+1)*(num_tau+1);
K = reshape(x,1,num_vols);
tau = reshape(y,1,num_vols);
C_real = linspace(0,0,num_vols);
for i = 1:num_vols
  C_real(i) = BlackScholesCall(S0,K(i),r,tau(i),sigma,q);
endfor
%rel_diffs = European_call(S0,K,r,tau,sigma,q)./C_real-1; printf("Minimum / maximum relative difference: %g / %g\n",min(rel_diffs),max(rel_diffs))
%C_real = European_call(S0,K,r,tau,sigma,q);
temp0 = find(C_real<1e-20);
C_real(temp0) = [];
tau(temp0) = [];
K(temp0) = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (1 == type)
  printf("\nBeginning timing of DBNR method...\n");
endif
tic
N = 500; % the number of abscissas for calculating epsilon
N2 = 1999; % the number of abscissas for calculating boundary
sigma_down = 1e-3;sigma_up = 1; % naive boundary
epsilon0 = 1e-4; % naive limiting parameter
k1 = 1e-3; % adjusted coefficient related to boundary

% compute tighter bound
[sigma_min_dirac,sigma_max_dirac,epsilon] = Dirac_delta_bound(S0,K,r,q,tau,C_real,...
    N,N2,sigma_down,sigma_up,epsilon0,k1);

% midpoint of the upper and lower boundaries is chosen as the initial point of Newton-Raphson method
N_iterative = 101;
num_vols = columns(C_real);

IV_DBNR = zeros(N_iterative,num_vols);% Pre allocate memory to related variables
IV_DBNR(1,:) = (sigma_min_dirac + sigma_max_dirac)./2;
if (!keep_vector_form_of_option_function_calls)
  C_Dirac_bound = linspace(0,0,num_vols);
endif
for i = 2:N_iterative
    sigma_Dirac_bound = IV_DBNR(i-1,:);
	if (keep_vector_form_of_option_function_calls)
      C_Dirac_bound = European_call(S0,K,r,tau,sigma_Dirac_bound,q);
    else
	  for j = 1:num_vols
	    C_Dirac_bound(j) = BlackScholesCall(S0,K(j),r,tau(j),sigma_Dirac_bound(j),q);
      endfor
    endif
    vega_Dirac_bound = (exp((-((tau.^2.*(4.*(q-r).^2+4.*(q+r).*(sigma_Dirac_bound).^2+(sigma_Dirac_bound).^4)+...
        4.*log(S0./K).^2)./(8.*tau.*(sigma_Dirac_bound).^2)))).*K.*(S0./K).^(1./2+(q-r)./(sigma_Dirac_bound).^2).*...
        sqrt(tau))./sqrt(2.*(pi));
    % first-order form of Newton-Raphson method
    IV_DBNR(i,:) = sigma_Dirac_bound - (C_Dirac_bound - C_real)./vega_Dirac_bound;
end
elapsed_time = toc;
if (1 == type)
  printf("Elapsed time is %g seconds. ",elapsed_time);
  printf("Average time for DBNR method per implied volatility calculation: %g seconds.\n\n",elapsed_time/num_vols);
endif
IV_DBNR_average_time = elapsed_time/num_vols;
% absolute error and its statistics
t2_a = abs(IV_DBNR - sigma);
t2_a_statistics = [mean(t2_a(end,:)),std(t2_a(end,:)),max(t2_a(end,:)),min(t2_a(end,:)),IV_DBNR_average_time];
% average error of each iteration step
t2_b = mean(abs(IV_DBNR - sigma),2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choi method
if (1 == type)
  printf("Beginning timing of Choi method...\n");
endif
tic
C_real_2 = exp((r - q).*tau).*C_real;
c = (C_real_2 - max(S0.*exp((r-q).*tau) - K,0))./min(K,S0.*exp((r-q).*tau));
k = abs(log(S0.*exp((r-q).*tau)./K));
temp1 = min((1+c)./2,c + exp(k).*normcdf(-sqrt(2.*k)));
sigma_max_Choi = (norminv(temp1) - norminv((temp1 - c)./exp(k)));
d1 = -k./sigma_max_Choi + sigma_max_Choi./2;
d2 = -k./sigma_max_Choi - sigma_max_Choi./2;
temp2 =  1 - exp(k).*normcdf(d2)./normcdf(d1);
temp3 = norminv(c./temp2);
sigma_min_Choi = (temp3 + sqrt(temp3.^2 + 2.*k))./sqrt(tau);
temp0 = find(isnan(sigma_min_Choi)==1);
sigma_min_Choi(temp0) = 0;
sigma_max_Choi = (norminv(temp1) - norminv((temp1 - c)./exp(k)))./sqrt(tau);

N_iterative = 101;
IV_Choi = zeros(N_iterative,columns(K));
IV_Choi(1,:) = sigma_min_Choi;
for i = 2:N_iterative
    sigma_normal = IV_Choi(i-1,:).*sqrt(tau);
    d1 = -k./sigma_normal + sigma_normal./2;
    d2 = -k./sigma_normal - sigma_normal./2;
    CV = (normcdf(d1,0,1) - exp(k).*normcdf(d2,0,1))./normpdf(d1,0,1);
    IV_Choi(i,:) = 1./sqrt(tau).*...
        (sigma_normal + (d1.^2/2 - log(CV) + log(c.*sqrt(2.*pi))).*CV);
end
elapsed_time = toc;
if (1 == type)
  printf("Elapsed time is %g seconds. ",elapsed_time);
  printf("Average time for Choi method per implied volatility calculation: %g seconds.\n\n",elapsed_time/num_vols);
endif
IV_Choi_average_time = elapsed_time/num_vols;

% absolute error and its statistics
t3_a = abs(IV_Choi - sigma);
t3_a_statistics = [mean(t3_a(end,:)),std(t3_a(end,:)),max(t3_a(end,:)),min(t3_a(end,:)),IV_Choi_average_time];
% average error of each iteration step
t3_b = mean(abs(IV_Choi - sigma),2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inversion formula 2
for type2 = 1:2
%type2 = 1;%1,2
switch type2
    case 1 % improved bound and epsilon
	  if (1 == type)
		printf("Timing trapezoidal integration evaluation of equation (2) via 'improved bound and epsilon'.\n");
	  endif
	  k2 = 1e-4;
	  epsilon_inverse = epsilon.*k2;
	  N = N2;
    case 2 % naive bound and epsilon
	  if (1 == type)
		printf("Timing trapezoidal integration evaluation of equation (2) via 'naive bound and epsilon'.\n");
	  endif
	  epsilon_inverse = 1e-4;
	  N = 3999;
	  sigma_min_dirac = 1e-3;sigma_max_dirac = 1;
end

tic
% generate discrete points
i = (1:N+1)';
sigma_all = sigma_min_dirac + (i-1).*(sigma_max_dirac - sigma_min_dirac)./N;

% compute IV according to formula 2
vega = (exp(-r.*tau-1./2.*(sqrt(tau).*(sigma)-(tau.*(-q+r+(sigma).^2./2)+...
    log(S0./K))./(sqrt(tau).*(sigma))).^2).*K.*(tau.*(-q+r+(sigma).^2./2)+...
    log(S0./K)))./(sqrt(2.*(pi)).*sqrt(tau).*(sigma).^2)-(exp(-q.*tau-(tau.*...
    (-q+r+(sigma).^2./2)+log(S0./K)).^2./(2.*tau.*(sigma).^2)).*S0.*...
    (-(sqrt(tau)./sqrt(2))+(tau.*(-q+r+(sigma).^2./2)+log(S0./K))./(sqrt(2).*...
    sqrt(tau).*(sigma).^2)))./sqrt((pi));

num_cols = columns(K);
num_vols = num_cols;
if (keep_vector_form_of_option_function_calls)
  values = European_call(S0,K,r,tau,sigma_all,q);
else
  assert( num_vols == columns(sigma_all) || 1 == columns(sigma_all) );
  assert( num_cols , columns(tau) , 0);
  num_rows = rows(sigma_all);
  values = zeros(num_rows,num_cols);
  sigma_all_has_single_column = 1 == columns(sigma_all);
  for i_row = 1:num_rows
    if (sigma_all_has_single_column)
      for i_col = 1:num_cols
        values(i_row,i_col) = BlackScholesCall(S0,K(i_col),r,tau(i_col),sigma_all(i_row),q);
      endfor;
	else
      for i_col = 1:num_cols
        values(i_row,i_col) = BlackScholesCall(S0,K(i_col),r,tau(i_col),sigma_all(i_row,i_col),q);
      endfor;
	endif
  endfor;
endif

f = sigma_all.*exp(-(values - C_real).^2./...
    (4.*epsilon_inverse)).*...
    (vega);
IV_inverse = (1./(2.*sqrt(pi.*epsilon_inverse))).*(sigma_max_dirac - sigma_min_dirac)/N.*...
    (0.5.*(f(1,:) + f(end,:)) + sum(f(2:end,:)));

elapsed_time = toc;
if (1 == type)
  printf("Elapsed time is %g seconds. ",elapsed_time);
  printf("Average time for trapezoidal integration evaluation of equation (2) per implied volatility calculation: %g seconds.\n\n",elapsed_time/num_vols);
endif
IV_formula_2_average_time = elapsed_time/num_vols;
% absolute error
t4_a = abs(IV_inverse - sigma);
% statistics
switch type2
    case 1 % improved bound and epsilon
		t41_a_statistics = [mean(t4_a),std(t4_a),max(t4_a),min(t4_a),IV_formula_2_average_time];
    case 2 % naive bound and epsilon
		t42_a_statistics = [mean(t4_a),std(t4_a),max(t4_a),min(t4_a),IV_formula_2_average_time];
end
endfor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab algorithms

num_vols = columns(C_real);

if (1 == type)
  printf("Timing %g implied volatility calculations with blsimpv() function provided by %s.\n",num_vols,this_program)
endif

tic
if (isOctave)
  % Octave's blsimpv() implementation does not allow for 'Method','jackel2016' or 'Method','search' -  hence, under Octave,
  % we fall back to its fzero based implementation to solve by gradient (Newton or quasi-Newton).,'Method','search'
  IV_brent = blsimpv(S0,K,r,tau,C_real); % Octave uses fzero to solve by gradient (Newton or quasi-Newton).
else
  IV_brent = blsimpv(S0,K,r,tau,C_real,'Method','search'); % Dekker-Brent method
endif
elapsed_time = toc;
if (1 == type)
  printf("Elapsed time is %g seconds. ",elapsed_time);
  printf("Average time per implied volatility calculation: %g seconds.\n\n",elapsed_time/num_vols);
endif
IV_brent_average_time = elapsed_time/num_vols;

tic
if (!isOctave && keep_vector_form_of_option_function_calls)
  if (1 == type)
    printf("Timing %g implied volatility calculations with Let's Be Rational function [vectorised, invoked via MATLAB's blsimpv()].\n",num_vols);
  endif
  IV_jackel = blsimpv(S0,K,r,tau,C_real,'Method','jackel2016');% Jackel’s method
else
  if (1 == type)
    if (use_letsberational_implied_black_volatility)
      printf("Timing %g implied volatility calculations with Let's Be Rational function [looping, invoked in XLL].\n",num_vols);
    elseif (!isOctave)
      printf("Timing %g implied volatility calculations with Let's Be Rational function [looping, invoked via MATLAB's blsimpv()].\n",num_vols);
    else
      printf("Timing %g implied volatility calculations with Octave's fzero-based implementation.\n",num_vols);
    endif
  endif
  IV_jackel = linspace(0,0,num_vols);
  for j = 1:num_vols
    IV_jackel(j) = LetsBeRationalForCall(S0*exp(-q*tau(j)),K(j),r,tau(j),C_real(j));
  endfor;
end
elapsed_time = toc;
if (1 == type)
  printf("Elapsed time is %g seconds. ",elapsed_time);
  printf("Average time per implied volatility calculation: %g seconds.\n\n",elapsed_time/num_vols);
endif
IV_jaeckel_average_time = elapsed_time/num_vols;

% absolute error and its statistics
t5_a = abs(IV_brent - sigma);
t5_a_statistics = [mean(t5_a),std(t5_a),max(t5_a),min(t5_a),IV_brent_average_time];

t6_a = abs(IV_jackel - sigma);
t6_a_statistics = [mean(t6_a),std(t6_a),max(t6_a),min(t6_a),IV_jaeckel_average_time];

function ShowStatsLine(name, stats)
  printf("%-17s", name)
  for i = 1:columns(stats)
    printf("\t%8.2E",stats(i));
  endfor
  printf("\n");
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch type

    case 1

printf("%g x %g x %g => %g, out of which %g option values above 1E-20\n\n",num_K+1,num_tau+1,num_sigma+1,(num_K+1)*(num_tau+1)*(num_sigma+1),num_vols);
printf("Table 2:\n");
printf("========\n");
printf("Method           \tMean       \tStd.Dev.   \tMaximum    \tMinimum    \ttime\n");
printf("------------------------------------------------------------------------------------------------\n");
ShowStatsLine("Formula (2)*",t42_a_statistics);
ShowStatsLine("Formula (2)",t41_a_statistics);
ShowStatsLine("DBNR",t2_a_statistics);
ShowStatsLine("Choi",t3_a_statistics);
ShowStatsLine("Dekker-Brent",t5_a_statistics);
if (use_letsberational_implied_black_volatility || !isOctave)
  ShowStatsLine("Let's Be Rational",t6_a_statistics);
endif
printf("------------------------------------------------------------------------------------------------\n\n");

figure(8,'toolbar','none','menubar','none','position',[0,4*(figure_height+figure_head_height)+figure_height_offset,figure_width,figure_height]);
clf;
num_iterative = 0:N_iterative-1;
subplot(1,2,1)
axis([0,10,1e-20,1])
plot(num_iterative(1:11),t2_b(1:11),'r-*','markersize',4,'linewidth',1)
hold on
plot(num_iterative(1:11),t3_b(1:11),'b-o','markersize',4,'linewidth',1)
legend('DBNR method','Choi method')
xlabel('Number of Iterations')
ylabel('Average AEIV')
set(gca,'YScale','log');

subplot(1,2,2)
axis([num_iterative(end-9),num_iterative(end),3e-16,5.0e-16])
plot(num_iterative(end-9:end),t2_b(end-9:end),'r-*','markersize',4,'linewidth',1)
hold on
plot(num_iterative(end-9:end),t3_b(end-9:end),'b-o','markersize',4,'linewidth',1)
legend('DBNR method','Choi method')
xlabel('Number of Iterations')
ylabel('Average AEIV')

    otherwise % figure(9)-(10)
% Increased AEIV of Dekker-Brent method relative to DBNR method(of which error
% of each output IV in the final iteration)
accuracy_up_brent = t5_a - t2_a(end,:);
figure(7+type,'toolbar','none','menubar','none','position',[0,(7-2*type)*(figure_height+figure_head_height)+figure_height_offset,figure_width,figure_height]);
clf;
if (isOctave) % See tps://techniex.com/octave-data-visualization/
  colormap(mycolourmap(128));
endif
subplot(1,2,1)
accuracy_up_brent_reshape = reshape(accuracy_up_brent,num_K+1,num_tau+1);
heatmapplot(tau0,K0,accuracy_up_brent_reshape,'\itT','\itK',['Increased AEIV of Dekker-Brent method (\sigma = ',num2str(sigma),')']);
% Increased AEIV of Jackel’s method relative to DBNR method(of which error
% of each output IV in the final iteration)
accuracy_up_jackel = t6_a - t2_a(end,:);
temp1 = find(accuracy_up_jackel>1e-1);
accuracy_up_jackel((temp1)) = nan;
subplot(1,2,2)
accuracy_up_jackel_reshape = reshape(accuracy_up_jackel,num_K+1,num_tau+1);
h = heatmapplot(tau0,K0,accuracy_up_jackel_reshape,'\itT','\itK',["{\\it'Increased' (?) } AEIV of Jaeckel's \"Let's Be Rational\" method (\sigma = ",num2str(sigma),')']);
if (!isOctave)
  h.MissingDataLabel='0.1';
endif

figure(9+type,'toolbar','none','menubar','none','position',[0,(6-2*type)*(figure_height+figure_head_height)+figure_height_offset,figure_width,figure_height],'NumberTitle','off','Name',['Figure ',num2str(7+type),'b (not in the original paper)']);
clf;
if (isOctave) % See tps://techniex.com/octave-data-visualization/
  colormap(mycolourmap(128,0.5));
endif
subplot(1,3,1);
heatmapplot(tau0,K0,reshape(t5_a,num_K+1,num_tau+1),'\itT','\itK',['AEIV of Dekker-Brent method (\sigma = ',num2str(sigma),')']);
subplot(1,3,2)
heatmapplot(tau0,K0,reshape(t2_a(end,:),num_K+1,num_tau+1),'\itT','\itK',['AEIV of DBNR method (\sigma = ',num2str(sigma),')']);
subplot(1,3,3)
heatmapplot(tau0,K0,reshape(t6_a,num_K+1,num_tau+1),'\itT','\itK',["AEIV of Jaeckel's \"Let's Be Rational\" method (\sigma = ",num2str(sigma),')']);
end

endfor

total_time = toc(start_tic);
printf("\nTotal time: %g seconds.\n\n",total_time);

printf("Hit Enter to end program.\n")
pause
