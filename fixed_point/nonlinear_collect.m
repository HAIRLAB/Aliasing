clear all
close all
rng(2141444)

%%%                     \dot{x1} = -x2-x1(x1^2+x2^2); \dot{x2} = x1-x2(x1^2+x2^2)       %%%

%% *************************** Dynamics ***********************************

mu = 1;
    %f_u =  @(t,x)([ x(1,:); x(2,:)-x(1,:).^2] );
    f_u =  @(t,x)([ -3.*x(2,:)-mu.*x(1,:).*(x(1,:).^2+x(2,:).^2); 3.*x(1,:)-mu.*x(2,:).*(x(1,:).^2+x(2,:).^2)] );
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
MaxT = deltaT; Nsim_traj = 500;% 200 sampled point for each traj for input
Nsim = Nsim_traj*MaxT/deltaT; % collect point for each traj
Ntraj = 1000; % 1000 trajectories

Xcurrent = ([1;1]);%generate size of (n,Ntraj) random initial condition [-1,1]

X = []; Y = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent);
    X = [X Xcurrent];
    Y = [Y Xnext];
    Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %1.2f s \n', toc);
t = 0:deltaT: (Nsim_traj-1)*deltaT;
lw = 3;
plot(t, X(1,:),'linewidth',lw); hold on;
    %annotation('doublearrow',XY2Norm('X',[0,1.3]),XY2Norm('Y',[0,0]),'LineStyle','-','color','k','LineWidth',2,'HeadStyle','Plain');
    %text(pi/2,sin(pi/2),'T');
xlabel('Time [s]','interpreter','latex');
ylabel('State $$x_1$$', 'interpreter', 'latex');
set(gca,'fontsize',25);
h1 = legend({'$$\mu=1$$','$$\mu=2$$','$$\mu=3$$'},'interpreter','latex','fontsize',25);
set(h1,'Orientation','horizon');%,'Box','off')
save('Nonlineardata_mu=1.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');