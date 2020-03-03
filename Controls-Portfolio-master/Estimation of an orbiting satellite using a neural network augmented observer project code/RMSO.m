function [Xhat_hist,est_hist,W_hist] = RMSO(X_hist,Y_hist,Xhat0,X0,Y0,T,A,B,...
    K,C,D,F,sigma,emax,phi)
%This matlab function approximates a function or state history using a reduced
%order modified state observer.The following are the required inputs
%state or function value history (with noise included)
%measurement history (with noise included)
%initial conditions
%time step
%A and B matrices, use xhat1 as the symbolic variable
%MSO gain
%C matrix, assumed scalar
%D, tuning parameter
%F tuning matrix
%sigma, tuning parameter
%emax, tuning parameter

%get number of steps
[r,c]=size(X_hist);  %assuming each state is a column vector
num_steps=c;

%create storage for important variable histories
Xhat_hist=zeros(r,num_steps);
est_hist=zeros(3,num_steps-1);
W_hist = zeros(7,3,num_steps);

%RMSO loop
for i = 1:num_steps
    %initialization
    if i==1
        %Update states and estimates
        %Get measurement at time t=i-1
        Y=Y0;
        %initial weights = 0
        W0=zeros(7,3);
        %assign measured states to variables
        x = Y(1);
        y = Y(2);
        z = Y(3);
        %assign initial estimates to variables
        Xhat=Xhat0;
        xhat = Xhat(1);
        yhat = Xhat(2);
        zhat = Xhat(3);
        xhatdot = Xhat(4);
        yhatdot = Xhat(5);
        zhatdot = Xhat(6);
        %define values at t=i-1
        %A matrix
        X1hat=[x; y; z; xhatdot; yhatdot; zhatdot];
        A_val=eval(A);
        %robustifying term
        denom = sqrt((Y(1)-Xhat(1))^2 + (Y(2)-Xhat(2))^2 + ...
            (Y(3)-Xhat(3))^2);
        v = -D*((Y-C*Xhat) / denom) - emax*(Y-C*Xhat);
        %evaluate basis function vector 
        phi_val=eval(phi);
        %calculate uncertainty estimate at time t=i-1
        f=W0'*phi_val;
        %update values at t=i
        %update weights
        W = W0 + T*F*(phi_val*(Y - C*Xhat)' - sigma*W0);
        %update estimates (note that f: t=i-t, v: t=i-1) 
        Xhat = Xhat + T*(A_val*X1hat + B*(f-v) + K*(Y-C*Xhat));
        %store values at t=i
        Xhat_hist(:,i)=Xhat;
        W_hist(:,:,i) = W;
    else
        %general case
        %update states and estimates
        %get measurement at t=i-1
        Y=Y_hist(:,i-1);
        %assign measured states to variables
        x = Y(1);
        y = Y(2);
        z = Y(3);
        %assign estimated states to variables
        xhat = Xhat(1);
        yhat = Xhat(2);
        zhat = Xhat(3);
        xhatdot = Xhat(4);
        yhatdot = Xhat(5);
        zhatdot = Xhat(6);
        %define values at t=i-1
        %A matrix
        X1hat=[x; y; z; xhatdot; yhatdot; zhatdot];
        A_val=eval(A);  %calculate A using previously defined variables
        %robustifying term
        denom = sqrt((Y(1)-Xhat(1))^2 + (Y(2)-Xhat(2))^2 + ...
            (Y(3)-Xhat(3))^2);
        v= -D*((Y-C*Xhat) / denom) - emax*(Y-C*Xhat);
        %calculate basis function values
        phi_val=eval(phi);
        %calculate uncertainty estimate term
        f=W'*phi_val;
        %update values at t=i
        W = W + T*F*(phi_val*(Y - C*Xhat)' - sigma*W);
        Xhat = Xhat + T*(A_val*X1hat + B*(f-v) + K*(Y-C*Xhat));
        %store values
        Xhat_hist(:,i)=Xhat;
        est_hist(:,i-1)=f;
    end
end
end

