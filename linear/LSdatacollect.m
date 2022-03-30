clear all
close all
rng(2141444)

%%%                     \dot{x1} = a*x1+omega*x2; \dot{x2} = a*x2-omega*x1        %%%

%% *************************** Dynamics ***********************************
omega = 3;
a = 0.1;
f_u =  @(t,x)([ a.*x(1,:) + omega.*x(2,:); a.*x(2,:) - omega.*x(1,:)] );
n = 2;% number of states


%% ************************** Discretization ******************************

deltaT = 0.01;
%Runge-Kutta 4
k1 = @(t,x) (  f_u(t,x) );
k2 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT/2) );
k3 = @(t,x) ( f_u(t,x + k2(t,x)*deltaT/2) );
k4 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT) );
f_ud = @(t,x) ( x + (deltaT/6) * ( k1(t,x) + 2*k2(t,x) + 2*k3(t,x) + k4(t,x)  )   );

%% ************************** Collect data ********************************
tic
disp('Starting data collection')
MaxT = 6; Nsim_traj = 30;% 200 sampled point for each traj for input
Nsim = Nsim_traj*MaxT/deltaT; % collect point for each traj
Ntraj = 1000; % 1000 trajectories

Xcurrent = (rand(n,Ntraj)*2 - 1);%generate size of (n,Ntraj) random initial condition [-1,1]

X = []; Y = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent);
    X = [X Xcurrent];
    Y = [Y Xnext];
    Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %1.2f s \n', toc);
save('LSdata_omega=3_a=posi01.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');