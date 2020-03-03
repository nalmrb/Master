function [X_hist,Y_hist,Y0,f_X_hist] = num_int(f,f_X,X0,T,t,C)
%This is a numerical integration function designed to receive a symbolic
%function f, an initial condition x0, a step size T, and a time value t.
%This function integrates the symbolic function f from 0 to t given the
%initial conditions and step size value
%This function is designed to be used with filters or other estimation
%algorithms and thus adds random noise to the state. It also returns a
%measurement with added noise

%determine number of integration steps
num_steps=length(0:T:t);

%get dimensions of c
[meas, ~]=size(C);

%create storage for important variables
%state history
X_hist=zeros(length(X0),num_steps);
%measurement history
Y_hist=zeros(meas,num_steps);
%J2 effect matrix
f_X_hist=zeros(3,num_steps-1);

%initial measurement
Y0=C*X0;

%perform integration
for i =1:num_steps
    %initialization
    if i==1
        %generate values at t=i-1
        %assign state values
        x = X0(1);
        y = X0(2);
        z = X0(3);
        xdot = X0(4);
        ydot = X0(5);
        zdot = X0(6);
        %calculate equation values
        f_val=eval(f);
        %update variables at time t=i
        X=X0 + T*(f_val);
        Y=C*X0;
        %store variables
        X_hist(:,i)=X;
        Y_hist(:,i)=Y;
    else
        %general
        %generate values at t=i-1
        %assign state values
        x = X(1);
        y = X(2);
        z = X(3);
        xdot = X(4);
        ydot = X(5);
        zdot = X(6);
        %calculate values
        f_val=eval(f);
        f_X_val=eval(f_X);
        %update values at t=i
        X=X + T*(f_val);
        Y=C*X;
        %store variables
        X_hist(:,i)=X;
        Y_hist(:,i)=Y;
        f_X_hist(:,i-1)=f_X_val;
    end
end
end

