clear all
close all
rng(2141444)


%%%          \dot{x1} = -3x2-mu*x1(x1^2+x2^2); \dot{x2} = 3x1-mu*x2(x1^2+x2^2)      %%%

%% ************************** Data ******************************
mu=1; 
load('Nonlineardat_mu=1.mat','X','Y','MaxT','Nsim_traj','Ntraj','deltaT');
n = 2; %number of states 


%% ************************** Sampling for input ********************************
MaxT = 4; 
T_sample = [1:MaxT/deltaT];% sampled by T_sample*0.01 respectively
Nsim_traj = 30;
Nsample = Nsim_traj*Ntraj;
Npolyorder = 13;

NdeltaT = length(T_sample);
mea = zeros(NdeltaT,1);
NRMSE = zeros(NdeltaT,Npolyorder);

int_err = zeros(NdeltaT, n);

%% **********************************************************

for polyorder = [7,10,13]
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

    %% ********************** Compute integral error *********************************

    states = sym('x', [1:2]);
states_name = {'x1','x2'};
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


%% ****************************  Plots  ***********************************
if (polyorder==13) && ((h==50) + (h==110) + (h==280))
        
    figure('Units','centimeter','Position',[5 10 20 15]);
    t = [0:0.01:deltaT*(Nsim-1)];

plot(X_p(1,:),X_p(2,:),'linewidth',lw); 
hold on;
x_1 = zeros(length(t),1);
x_2 = zeros(length(t),1);
for m = 1:length(t)
    Uti = expm(t(m).*L);%expmdemo1(t(i).*L);
    x_1(m) = g(Xinitial(1),Xinitial(2))*Uti(:,1);
    x_2(m) = g(Xinitial(1),Xinitial(2))*Uti(:,2);
end
plot(x_1,x_2,'linewidth',lw); 

hold on;

    sampleTs = h*deltaT;
    dt = [0:sampleTs:deltaT*(Nsim-1)];
    dx_1 = zeros(length(dt),1);
    dx_2 = zeros(length(dt),1);
    for mm = 1:length(dt)
        dx_1(mm) = X_p(1,(mm-1)*h+1);
        dx_2(mm) = X_p(2,(mm-1)*h+1);
    end
    plot(dx_1,dx_2,'^','linewidth',3,'Markersize',18); hold on;
    set(gca,'xlim',[-1.5,1.5]);
    set(gca,'ylim',[-1.5,1.5]);
    LEG = legend('CT ${x(t)}$','CT $\hat{{x}}(t)$','DT $x(kT)$');
    set(LEG,'interpreter','latex');
    xlabel('$$x_1$$','interpreter','latex');
    ylabel('$$x_2$$', 'interpreter', 'latex');
    set(gca,'fontsize',33);
    %or
    %plot(dt,dx_1,'--^','linewidth',2.5,'Markersize',10); 
end
end
end
figure('Units','centimeter','Position',[5 10 20 15]);
lw =3;
x = pi/3;
for k = 7:3:13
plot(T_sample(1:h-1)*deltaT,(NRMSE(1:h-1,k)).^(1/4),'linewidth',lw); 
hold on
end
hold on
ylim =get(gca,'Ylim');
plot([x,x], ylim,'r--','linewidth',lw);

LEG = legend('$m=7$','$m=10$','$m=13$');
set(LEG,'interpreter','latex')

%title('Linear', 'interpreter','latex'); 
xlabel('Sampling period [s]','interpreter','latex');
ylabel('$NRMSE^{1/4}$', 'interpreter', 'latex');
set(gca,'fontsize',30);


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



