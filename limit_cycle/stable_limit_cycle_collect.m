clear all
close all
rng(2141444)


%% *************************** Dynamics ***********************************
    mu = 3;
    f_u =  @(t,x)([ mu.*x(2,:) - x(1,:).*(x(1,:).^2+x(2,:).^2-1); -mu.*x(1,:) - x(2,:).*(x(1,:).^2+x(2,:).^2-1)] );
%f_u =  @(t,x)([ x(2,:) ; x(2,:) - x(1,:).^2.*x(2,:) - x(1,:)] );
    n = 2; %number of states 

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
MaxT = 5;%deltaT; 
Nsim_traj = 30;%2000;% 200 sampled point for each traj for input
Nsim = Nsim_traj*MaxT/deltaT; % collect point for each traj
Ntraj = 1000;%1000; % 1000 trajectories

Xcurrent = (rand(n,Ntraj)*3-1.5);%generate size of (n,Ntraj) random initial condition

X = []; Y = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent);
    X = [X Xcurrent];
    Y = [Y Xnext];
    Xcurrent = Xnext;
end
t = 0:deltaT: (Nsim_traj-1)*deltaT;
fprintf('Data collection DONE, time = %1.2f s \n', toc);
save(['stable_limitcycle_mu=', num2str(mu),'_initial_1p5.mat'],'X','Y','MaxT','Nsim_traj','Ntraj','deltaT');