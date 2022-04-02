clear all
close all
rng(2141444)
addpath('./figure');


%%%   \dot{x1} = a.*x(1,:) + omega.*x(2,:);\dot{x2} =a.*x(2,:) - omega.*x(1,:);         %%%

%% ************************** Data ******************************
load('LSdat_omega=3_a=posi01.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');
%load('LSdat.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');

n = 2;
Nsim = 500;
Xinitial = ([0.5;0.5]);%generate size of (n,Ntraj) random initial condition [-1,1]
%% ************************** data for predict********************************
omega = 3;
a = 0.1;

states = sym('x', [1:n]);
states_name = {'x1','x2'};

f_u =  @(t,x)([ a.*x(1,:) + omega.*x(2,:); a.*x(2,:) - omega.*x(1,:)] );
vec_f1 = @(x1,x2)([ a.*x1 + omega.*x2] );
vec_f2 = @(x1,x2)([ -omega.*x1 + a.*x2 ] ); 

%Runge-Kutta 4
k1 = @(t,x) (  f_u(t,x) );
k2 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT/2) );
k3 = @(t,x) ( f_u(t,x + k2(t,x)*deltaT/2) );
k4 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT) );
f_ud = @(t,x) ( x + (deltaT/6) * ( k1(t,x) + 2*k2(t,x) + 2*k3(t,x) + k4(t,x)  )   );

tic
disp('Starting data collection')
Xcurrent = Xinitial;
X_p = []; Y_p = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent);
    X_p = [X_p Xcurrent];
    Y_p = [Y_p Xnext];
    Xcurrent = Xnext;
end
lw = 3;

%% ************************** Sampling for input ********************************
MaxT = 5;
T_sample = [1:MaxT/deltaT];% sampled by T_sample*0.01 respectively
Nsim_traj = 30;
Nsample = Nsim_traj*Ntraj;
Npolyorder = 1;%only 1

NdeltaT = length(T_sample);
mea = zeros(NdeltaT,1);
NRMSE = zeros(NdeltaT,Npolyorder);
%imag_max = zeros(NdeltaT,1);

polyorder = Npolyorder
for h = 1:NdeltaT %If you just want the picture of state prediction T=0.5,1.1 or 2.8, change it to h=[50,110,280].
    fprintf('Starting sampling, period = %1.2f s \n', T_sample(h)*deltaT);
    Xtmp = [];
    Ytmp = [];
    %Nh = Nsim_traj*h;
    for i = 1:Nsim_traj
        Xtmp = [Xtmp X(:,Ntraj*h*(i-1)+1:Ntraj*h*(i-1)+Ntraj)];
        Ytmp = [Ytmp Y(:,Ntraj*h*i-999:Ntraj*i*h)];
    end

    [input] = make_input_fd(Xtmp);        
    [output] = make_output_fd(Ytmp);

    instates = input.variable; %x1(t),x2(t)
    instates_name = input.name;%'x1','x2'

    outstates = output.variable;%y1(t),y2(t)
    outstates_name = output.name;%'y1','y2'



%% ************************** Basis functions and Lifting ***********************
%disp('Starting LIFTING')
tic
[Xlift, xbasis_name] = build_basis(instates, instates_name, polyorder);
[Ylift, ~] = build_basis(outstates, outstates_name, polyorder);
%fprintf('Lifting DONE, time = %1.2f s \n', toc);
N = size(Xlift,2);% number of basis functions


%% ********************** System identification *********************************

%disp('Starting REGRESSION')
tic
U = pinv(Xlift) * Ylift;
L = logm(U)./(T_sample(h)*deltaT);
%fprintf('Regression done, time = %1.2f s \n', toc);


%% ********************** Print result *********************************
    delta = 1e-8;
    w_estimate = zeros(N,n);

    for i=1:n
        y_name = strcat('\dot{x(',num2str(i),')}');
        fprintf('\n%s = ', y_name);
        w_estimate(:,i) = L(:,i);
        for j = 1:N
            if   w_estimate(j,i).^2/norm(w_estimate(:,i))^2<delta
                w_estimate(j,i)=0;
            else
                if w_estimate(j,i)< 0
                    fprintf('%.4f%s', w_estimate(j,i),xbasis_name{j});
                else 
                    fprintf('+');
                    fprintf('%.4f%s', w_estimate(j,i),xbasis_name{j});
                end
            end 
        end
    end


%% %% ****************************  Print error  *******************************

    w_true = zeros(N,n);    Nw_nonzero = 4;
    w_true(1,1) = a; w_true(2,1) = omega; w_true(1,2) = -omega; w_true(2,2) = a; 
    err = abs(w_estimate-w_true);
    mea(h) = mean(err(:));
    NRMSE(h,polyorder) = sum(sum(err.^2))*Nw_nonzero/(N*N*n*n*mean(abs(w_true(:))));
    fprintf('\nerror-polyorder = %.0f \n: mea = %.4f%%, NRMSE = %.4f%%\n',polyorder, mea(h),NRMSE(h,polyorder));

%% ****************************  Plots  ***********************************
if (h==280) + (h==110) + (h==50)
    t = [0:0.01:deltaT*(Nsim-1)];

    basis = [];
    basis_name = {};

    basis = [basis, states];
    basis_name = [basis_name,states_name];
    if polyorder > 1
        for i = 2:polyorder
            tmp = ones(1,i);
            comb = [];        
            [comb] = comb_with_rep(comb,tmp,size(states,2),i); %generate all possible combinations
            for j = 1:size(comb,1)
                nametmp = states_name(comb(j,1));
                basistmp = states(:,comb(j,1));
                for k = 2:size(comb,2) 
                    nametmp = strcat(nametmp,'*',states_name(comb(j,k)));
                    basistmp = basistmp.*states(:,comb(j,k));
                end
                basis = [basis,basistmp];
                basis_name = [basis_name,nametmp];            
            end
        end
    end 
    g = matlabFunction(basis);    

figure('Units','centimeter','Position',[5 10 20 15]);

t = [0:0.01:deltaT*(Nsim-1)];
%d = 0.2;

plot(X_p(1,:),X_p(2,:),'-','linewidth',lw); 
hold on;
% [x,y]=meshgrid(min(X_p(1,:)):d:max(X_p(1,:)), min(X_p(2,:)):d:max(X_p(2,:)));
% u = vec_f1(x,y);
% v = vec_f2(x,y);
% quiver(x,y,u,v);

x_1 = zeros(length(t),1); %x1 is sym. do not use it. 
x_2 = zeros(length(t),1);
for m = 1:length(t)
    Uti = expm(t(m).*L);%expmdemo1(t(i).*L);
    x_1(m) = g(Xinitial(1),Xinitial(2))*Uti(:,1);
    x_2(m) = g(Xinitial(1),Xinitial(2))*Uti(:,2);
end
plot(x_1,x_2,'-','linewidth',lw); 
hold on;

    if a>0
    set(gca,'xlim',[-1.5,1.5]);
    set(gca,'ylim',[-1.5,1.5]);
    else
    set(gca,'xlim',[-1,1]);
    set(gca,'ylim',[-1,1]);
    end
    

% [x,y]=meshgrid(min(x_1):d:max(x_1), min(x_2):d:max(x_2));
% G = g(x,y);
% G1 = reshape(G,size(x,1)*size(x,2),size(G,2)/size(x,2));
% u = reshape(G1 *L(:,1),size(x,1), size(x,2));
% v = reshape(G1 *L(:,2),size(x,1), size(x,2));
% quiver(x,y,u,v);

    sampleTs = h*deltaT;
    dt = [0:sampleTs:deltaT*(Nsim-1)];
    dx_1 = zeros(length(dt),1);
    dx_2 = zeros(length(dt),1);
    for mm = 1:length(dt)
        dx_1(mm) = X_p(1,(mm-1)*h+1);
        dx_2(mm) = X_p(2,(mm-1)*h+1);
    end
    plot(dx_1,dx_2,'^','color','k','linewidth',4,'Markersize',20); 
    xlabel('$x_1$','interpreter','latex');
    ylabel('$x_2$', 'interpreter', 'latex');
    set(gca,'fontsize',33);
if a>0
    set(gca,'xlim',[-1.5,1.5]);
    set(gca,'ylim',[-1.5,1.5]);
else
    set(gca,'xlim',[-1,1]);
    set(gca,'ylim',[-1,1]);
end
LEG = legend('CT ${x(t)}$','CT $\hat{{x}}(t)$','DT $x(kT)$');
set(LEG,'interpreter','latex');
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$', 'interpreter', 'latex');
set(gca,'fontsize',30);

end

end
%% ****************************  Plots  ***********************************
lw = 3;
x = pi/omega;

figure('Units','centimeter','Position',[5 10 20 15]);
plot(T_sample(1:h-1)*deltaT,NRMSE(1:h-1,polyorder),'linewidth',lw); hold on

%title('Linear', 'interpreter','latex'); 
xlabel('Sampling period [s]','interpreter','latex');
ylabel('NRMSE', 'interpreter', 'latex');
set(gca,'fontsize',30);set(gca,'xlim',[0,5]);
ylim =get(gca,'Ylim');
plot([x,x], ylim,'r--','linewidth',lw);


function [comb] = comb_with_rep(comb,tmp,N,K)

% @author: Ye Yuan
%This recursive function generates combinations with replacement.
%
%Inputs:
%      comb: combinations that have been created
%      tmp : next combination to be included in comb
%      N   : number of all possible elements
%      K   : number of elements to be selected 
%
%Output:
%      comb: updated combinations


if tmp(1)>N
    return;
end

for i = tmp(K):N
    comb = [comb;tmp];
    tmp(K) = tmp(K) + 1;
end
tmp(K) = tmp(K) - 1;
for i = K:-1:1
    if tmp(i)~=N
        break;
    end
end
tmp(i) = tmp(i)+1;
for j = i+1:K
    tmp(j)=tmp(i);
end
[comb] = comb_with_rep(comb,tmp,N,K);
end



