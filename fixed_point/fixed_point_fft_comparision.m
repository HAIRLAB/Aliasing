clear all
close all
rng(2141444)
addpath('./figure');


%%%                   \dot{x1} = -3x2-mu*x1(x1^2+x2^2); \dot{x2} = 3x1-mu*x2(x1^2+x2^2)      %%%

%% ************************** Data ******************************
mu=1; ii = 1;
load('Nonlineardat_mu=1.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');
n = 2; %number of states 

Nsim = 500;
Xinitial = ([0.5;0.5]);%generate size of (n,Ntraj) random initial condition [-1,1]
Xcurrent = Xinitial;

%% ************************** original data for predict********************************
f_u =  @(t,x)([ -3.*x(2,:)-mu.*x(1,:).*(x(1,:).^2+x(2,:).^2); 3.*x(1,:)-mu.*x(2,:).*(x(1,:).^2+x(2,:).^2)] );

states = sym('x', [1:n]);
states_name = {'x1','x2'};

vec_f1 = @(x1,x2)([ -3.*x2-mu.*x1.*(x1.^2+x2.^2)] );
vec_f2 = @(x1,x2)([  3.*x1-mu.*x2.*(x1.^2+x2.^2)]); 

%f_u =  @(t,x)([ x(1,:) + x(2,:); x(2,:) - x(1,:)] );

%Runge-Kutta 4
k1 = @(t,x) (  f_u(t,x) );
k2 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT/2) );
k3 = @(t,x) ( f_u(t,x + k2(t,x)*deltaT/2) );
k4 = @(t,x) ( f_u(t,x + k1(t,x)*deltaT) );
f_ud = @(t,x) ( x + (deltaT/6) * ( k1(t,x) + 2*k2(t,x) + 2*k3(t,x) + k4(t,x)  )   );

tic
disp('Starting data collection')
Xcurrent = Xinitial;
X_o = []; Y_o = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent);
    X_o = [X_o Xcurrent];
    Y_o = [Y_o Xnext];
    Xcurrent = Xnext;
end
lw = 3;

%% ************************** Sampling for input ********************************
MaxT = 4;
T_sample = [1:MaxT/deltaT];% sampled by T_sample*0.01 respectively
Nsim_traj = 30;
Nsample = Nsim_traj*Ntraj;
Npolyorder = 13;%only 1

NdeltaT = length(T_sample);
mea = zeros(NdeltaT,1);
NRMSE = zeros(NdeltaT,Npolyorder);
err_fft = zeros(n,NdeltaT);
%imag_max = zeros(NdeltaT,1);

for polyorder = Npolyorder
for h = 1:NdeltaT %separate by h 
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
    %polyorder = 1;  %maximum power of polynomial function to be included in basis function
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

%if isreal(L)==1
%    imag_max(h) = max(abs(imag(eig(L))));

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

    w_true = zeros(N,n);    Nw_nonzero = 6;
    w_true(2,1) = -3; w_true(6,1) = -mu; w_true(8,1) = -mu; w_true(1,2) = 3; w_true(7,2) = -mu; w_true(9,2) = -mu; 
    err = abs(w_estimate-w_true);
    mea(h) = mean(err(:));
    NRMSE(h,polyorder) = sum(sum(err.^2))*Nw_nonzero/(N*N*n*n*mean(abs(w_true(:))));
    fprintf('\nerror-polyorder = %.0f \n: mea = %.4f%%, NRMSE = %.4f%%\n',polyorder, mea(h),NRMSE(h,polyorder));
%else 
    %fprintf('deltaT=',deltaT(h));
%    break;
%end

%% ****************************  predict the states based on \hat{f}  ***********************************
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
    
    t = [0:0.01:deltaT*(Nsim-1)];
    X_p = zeros(n,length(t)); %x1 is sym. do not use it. 
for m = 1:length(t)
    Uti = expm(t(m).*L);%expmdemo1(t(i).*L);
    X_p(1,m) = g(Xinitial(1),Xinitial(2))*Uti(:,1);
    X_p(2,m) = g(Xinitial(1),Xinitial(2))*Uti(:,2);
end
    
    %% ****************************  fft of X_p and X_o  ***********************************
fe = 1/0.01; % frequency 
f = fe*(0:(Nsim/2))/Nsim;

%N_sim: number of samples
PH1_o = fft(X_o(1,:))/Nsim;
PH2_o = fft(X_o(2,:))/Nsim;
P1_o = PH1_o(1:Nsim/2+1);
P1_o(2:end-1) = 2*P1_o(2:end-1);
P2_o = PH2_o(1:Nsim/2+1);
P2_o(2:end-1) = 2*P2_o(2:end-1);

PH1_p = fft(X_p(1,:))/Nsim;
PH2_p = fft(X_p(2,:))/Nsim;
P1_p = PH1_p(1:Nsim/2+1);
P1_p(2:end-1) = 2*P1_p(2:end-1);
P2_p = PH2_p(1:Nsim/2+1);
P2_p(2:end-1) = 2*P2_p(2:end-1);

err_fft(1,h) = abs(sum(sum((P1_o-P1_p).^2)));
err_fft(2,h) = abs(sum(sum((P2_o-P2_p).^2)));

%% %%%%%%%%%%%%%%%%%%%% plot %%%%%%%%%%%%%%%%%%%%%
if (h==100)+(h==110)

lw = 3;
figure('Units','centimeter','Position',[5 10 20 15]);
plot(f,abs(P1_o),f,abs(P1_p),'linewidth',lw);
set(gca,'fontsize',30);
xlabel('$\omega$ (Hz)','interpreter','latex')
ylabel('$|P_1(f)|$','interpreter','latex')
%set(gca,'ylim',[0,1.6]);
%set(gca,'xlim',[0,25]);

figure('Units','centimeter','Position',[5 10 20 15]);
plot(f,abs(P2_o),f,abs(P2_p),'linewidth',lw);
xlabel('$\omega$ (Hz)','interpreter','latex')
ylabel('$|P_2(\omega)|$','interpreter','latex')
set(gca,'fontsize',30);
%set(gca,'ylim',[0,1.6]);
%set(gca,'xlim',[0,25]);


d = 0.2;
% [x,y]=meshgrid(min(x_1):d:max(x_1), min(x_2):d:max(x_2));
% G = g(x,y);
% G1 = reshape(G,size(x,1)*size(x,2),size(G,2)/size(x,2));
% u = reshape(G1 *L(:,1),size(x,1), size(x,2));
% v = reshape(G1 *L(:,2),size(x,1), size(x,2));
% quiver(x,y,u,v);

end
%saveas(gcf, ['mu=1_T_',num2str(sampleTs),'.fig'])

end
end
%% ****************************  Plots  ***********************************
lw = 3;
x = pi/3;

figure('Units','centimeter','Position',[5 10 20 15]);
plot(T_sample(1:h-1)*deltaT,err_fft(1,1:h-1),'linewidth',lw); hold on
plot(T_sample(1:h-1)*deltaT,err_fft(2,1:h-1),'linewidth',lw); hold on
%title('Linear', 'interpreter','latex'); 
xlabel('Sampling period [s]','interpreter','latex');
ylabel('Spectral error', 'interpreter', 'latex');
set(gca,'fontsize',30);%set(gca,'xlim',[0,5]);
ylim =get(gca,'Ylim');
hold on
plot([x,x], [0,0.8],'r--','linewidth',lw);
LEG = legend('state ${x_1(t)}$','state $x_2(t)$');
set(LEG,'interpreter','latex');




% open('compute-linear.fig');
% l = findall(gca,'type','line');
% TMAX = get(l,'YData');
% close;
% 
% lw = 3;
% 
% figure('Units','centimeter','Position',[5 15 20 15]);
% for polyorder = 1:Npolyorder
%     x=TMAX(polyorder);
%     plot(T_sample(1:h-1)*deltaT,NRMSE(1:h-1,polyorder),'linewidth',lw); hold on
%     ylim =get(gca,'Ylim');
%     hold on
%     plot([x,x], ylim,'r','linewidth',lw-1);
% 
% end
%title('Linear', 'interpreter','latex'); 
set(gca,'fontsize',30);

% figure
% plot([0:Nsim]*deltaT,x_true(2,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(2,:), '--r','linewidth',lw)
% plot([0:Nsim]*deltaT,X_loc_x0(2,:), '--g','linewidth',lw-1)
% plot([0:Nsim]*deltaT,X_loc_0(2,:), '--k','linewidth',lw-1)
% axis([0 Tmax min(x_koop(2,:))-0.15 max(x_koop(2,:))+0.15])
% title('Predictor comparison - $x_2$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','Local at $x_0$','Local at 0','location','southwest');
% set(LEG,'interpreter','latex')
% 
% figure
% plot([0:Nsim]*deltaT,x_true(1,:),'linewidth',lw); hold on
% plot([0:Nsim]*deltaT,x_koop(1,:), '--r','linewidth',lw)
% plot([0:Nsim]*deltaT,X_loc_x0(1,:), '--g','linewidth',lw-1)
% plot([0:Nsim]*deltaT,X_loc_0(1,:), '--k','linewidth',lw-1)
% axis([0 Tmax min(x_koop(1,:))-0.1 max(x_koop(1,:))+0.1])
% title('Predictor comparison - $x_1$','interpreter','latex'); xlabel('Time [s]','interpreter','latex');
% set(gca,'fontsize',20)
% LEG = legend('True','Koopman','Local at $x_0$','Local at 0','location','southwest');
% set(LEG,'interpreter','latex')


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



