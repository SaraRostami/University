clc;clear all;close all;

morris_lecar;



function morris_lecar

%parameter values
C = 1 ; 
gCa=20;  
VCa=60;
gK=10; 
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;
tau = 1;
Iext=0;
%-------------------------------------------------part1-------------------------------------------%
%V-nullclines and W-nullclines
figure;

m_inf = @(V) (1./(1+exp((Vhalf_m - V)/k_m)));
n_inf = @(V) (1./(1+exp((Vhalf_n - V)/k_n)));

Vnc = @(V) (Iext - gCa * m_inf(V) .*(V-VCa) - gL .* (V-VL)) ./ (gK .* (V-VK));
wnc = @(V) (n_inf(V));
fplot(@(V) Vnc(V),'r', [-80 80]);
hold on
fplot(@(V) wnc(V),'b', [-80 80]);
xlabel('V(mv)');
ylabel('w');
title('Phase Plane');

%calculate equilibrium points
syms V w
Vnc1_eqn = (1/C)*(Iext - gCa * m_inf(V) .*(V-VCa) - gK * w .*(V-VK) - gL .* (V-VL)) == 0;
wnc1_eqn = (n_inf(V) - w) == 0;
soln1 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-66, 0.03]);
soln2 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-56, 0.2]);
soln3 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-27, 39]);

V_eq1 = double(soln1.V);
w_eq1 = double(soln1.w);
plot(V_eq1, w_eq1);
text(V_eq1, w_eq1, ['(' num2str(round(V_eq1,3)) ',' num2str(round(w_eq1,3)) ')']);
grid on;

V_eq2 = double(soln2.V);
w_eq2 = double(soln2.w);
plot(V_eq2, w_eq2);
text(V_eq2, w_eq2, ['(' num2str(round(V_eq2,3)) ',' num2str(round(w_eq2,3)) ')']);
grid on;

V_eq3 = double(soln3.V);
w_eq3 = double(soln3.w);
plot(V_eq3, w_eq3);
text(V_eq3, w_eq3, ['(' num2str(round(V_eq3,3)) ',' num2str(round(w_eq3,3)) ')']);
grid on;

fprintf('Equilibrium points : \n 1. (%f, %f) \n 2. (%f, %f) \n 3. (%f, %f)\n', ...
        V_eq1, w_eq1, V_eq2, w_eq2, V_eq3, w_eq3);

% Simulate Trajectories

[V_quiv,w_quiv] = meshgrid(-80:80, -1.5:0.05:1.5);

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V_quiv)/k_m))).*(VCa-V_quiv) + gK*w_quiv .*(VK-V_quiv) + gL*(VL-V_quiv) + Iext);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V_quiv)/k_n)))-w_quiv);
quiver(V_quiv,w_quiv,dV_dt,dw_dt);
legend('V nullcline', 'w nullcline');


% Stability of the Jacobian matrix
syms V w

V_eq = [V_eq1, V_eq2, V_eq3];
w_eq = [w_eq1, w_eq2, w_eq3];

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V)/k_m))).*(VCa-V) + gK*w .*(VK-V) + gL*(VL-V) + Iext);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V)/k_n)))-w);

JSymbolic = jacobian([dV_dt, dw_dt],[V,w]);
eigVectors = 0;

for i = 1:3
    V = V_eq(i);
    w = w_eq(i);
Jmatrix = zeros(2,2);
Jmatrix(1,1) = subs(JSymbolic(1,1));
Jmatrix(1,2) = subs(JSymbolic(1,2));
Jmatrix(2,1) = subs(JSymbolic(2,1));
Jmatrix(2,2) = subs(JSymbolic(2,2));
eigenValues = eig(Jmatrix);

fprintf('Equilibrium point %d : The eigen values :  %f%+fi , %f%+fi \n', i, real(eigenValues(1)), imag(eigenValues(1)), ...
            real(eigenValues(2)), imag(eigenValues(2)));
end

%-------------------------------------------------part2-------------------------------------------%
%finding minimum amplitude of the step input --> Action Potential

options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('Action potential');
grid on;

options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage1(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('voltage for a current below I = 5');
grid on;

options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage2(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('voltage for a current more than I = 5');
grid on;

%-------------------------------------------------part3-------------------------------------------%
% for Ib
Ib = 4;
figure;
hold on
Vnc = @(V) (Ib - gCa * m_inf(V) .*(V-VCa) - gL .* (V-VL)) ./ (gK .* (V-VK));
wnc = @(V) (n_inf(V));
fplot(@(V) Vnc(V), [-80 80]);
fplot(@(V) wnc(V), [-80 80]);
xlabel('V(mv)');
ylabel('w');
title('Phase Plane');

% calculate equilibrium points
syms V w
Vnc1_eqn = (1/C)*(Ib - gCa * m_inf(V) .*(V-VCa) - gK * w .*(V-VK) - gL .* (V-VL)) == 0;
wnc1_eqn = (n_inf(V) - w) == 0;
soln1 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-62, 0.03]);
soln2 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-59, 0.2]);
soln3 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-27, 39]);

V_eq1 = double(soln1.V);
w_eq1 = double(soln1.w);
plot(V_eq1, w_eq1);
text(V_eq1, w_eq1, ['(' num2str(round(V_eq1,3)) ',' num2str(round(w_eq1,3)) ')']);
grid on;

V_eq2 = double(soln2.V);
w_eq2 = double(soln2.w);
plot(V_eq2, w_eq2);
text(V_eq2, w_eq2, ['(' num2str(round(V_eq2,3)) ',' num2str(round(w_eq2,3)) ')']);
grid on;

V_eq3 = double(soln3.V);
w_eq3 = double(soln3.w);
plot(V_eq3, w_eq3);
text(V_eq3, w_eq3, ['(' num2str(round(V_eq3,3)) ',' num2str(round(w_eq3,3)) ')']);
grid on;

fprintf('Equilibrium points : \n 1. (%f, %f) \n 2. (%f, %f) \n 3. (%f, %f)\n', ...
        V_eq1, w_eq1, V_eq2, w_eq2, V_eq3, w_eq3);
    
[V_quiv,w_quiv] = meshgrid(-80:80, -1.5:0.05:1.5);

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V_quiv)/k_m))).*(VCa-V_quiv) + gK*w_quiv .*(VK-V_quiv) + gL*(VL-V_quiv) + Ib);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V_quiv)/k_n)))-w_quiv);
quiver(V_quiv,w_quiv,dV_dt,dw_dt);
legend('V nullcline', 'w nullcline');    
% Stability of the Jacobian matrix
syms V w

V_eq = [V_eq1, V_eq2, V_eq3];
w_eq = [w_eq1, w_eq2, w_eq3];

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V)/k_m))).*(VCa-V) + gK*w .*(VK-V) + gL*(VL-V) + Ib);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V)/k_n)))-w);

JSymbolic = jacobian([dV_dt, dw_dt],[V,w]);
eigVectors = 0;

for i = 1:3
    V = V_eq(i);
    w = w_eq(i);
Jmatrix = zeros(2,2);
Jmatrix(1,1) = subs(JSymbolic(1,1));
Jmatrix(1,2) = subs(JSymbolic(1,2));
Jmatrix(2,1) = subs(JSymbolic(2,1));
Jmatrix(2,2) = subs(JSymbolic(2,2));
eigenValues = eig(Jmatrix);

fprintf('Equilibrium point %d : The eigen values :  %f%+fi , %f%+fi \n', i, real(eigenValues(1)), imag(eigenValues(1)), ...
            real(eigenValues(2)), imag(eigenValues(2)));
end

% for Ia
Ia = 6;
figure;
hold on
Vnc = @(V) (Ia - gCa * m_inf(V) .*(V-VCa) - gL .* (V-VL)) ./ (gK .* (V-VK));
wnc = @(V) (n_inf(V));
fplot(@(V) Vnc(V), [-80 80]);
fplot(@(V) wnc(V), [-80 80]);
xlabel('V(mv)');
ylabel('w');
title('Phase Plane');

% calculate equilibrium points
syms V w
Vnc1_eqn = (1/C)*(Ia - gCa * m_inf(V) .*(V-VCa) - gK * w .*(V-VK) - gL .* (V-VL)) == 0;
wnc1_eqn = (n_inf(V) - w) == 0;
soln1 = vpasolve([Vnc1_eqn, wnc1_eqn], [V, w], [-27, 0.39]);

V_eq1 = double(soln1.V);
w_eq1 = double(soln1.w);
plot(V_eq1, w_eq1);
text(V_eq1, w_eq1, ['(' num2str(round(V_eq1,3)) ',' num2str(round(w_eq1,3)) ')']);
grid on;

fprintf('equilibrium point : (%d,%d) \n', V_eq1, w_eq1);
    
[V_quiv,w_quiv] = meshgrid(-80:80, -1.5:0.05:1.5);

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V_quiv)/k_m))).*(VCa-V_quiv) + gK*w_quiv .*(VK-V_quiv) + gL*(VL-V_quiv) + Ia);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V_quiv)/k_n)))-w_quiv);
quiver(V_quiv,w_quiv,dV_dt,dw_dt);
legend('V nullcline', 'w nullcline');    
% Stability of the Jacobian matrix
syms V w

dV_dt = (1/C)*(gCa*(1./(1+exp((Vhalf_m - V)/k_m))).*(VCa-V) + gK*w .*(VK-V) + gL*(VL-V) + Ia);
dw_dt = (1./tau)*((1./(1+exp((Vhalf_n - V)/k_n)))-w);

JSymbolic = jacobian([dV_dt, dw_dt],[V,w]);

    V = V_eq1;
    w = w_eq1;
Jmatrix = zeros(2,2);
Jmatrix(1,1) = subs(JSymbolic(1,1));
Jmatrix(1,2) = subs(JSymbolic(1,2));
Jmatrix(2,1) = subs(JSymbolic(2,1));
Jmatrix(2,2) = subs(JSymbolic(2,2));
eigenValues = eig(Jmatrix);
i=1;
fprintf('Equilibrium point %d : The eigen values are  %f%+fi , %f%+fi \n', i, real(eigenValues(1)), imag(eigenValues(1)), ...
            real(eigenValues(2)), imag(eigenValues(2)));
 
%-------------------------------------------------part4-------------------------------------------%

% impulse input
options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage3(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('minimum amplitude of input impulse --> AP');
grid on;

options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage4(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('voltage for a current below I = 28');
grid on;

options = odeset('RelTol',1e-6,'AbsTol',1e-6, 'refine',5);

tSpan = [0, 100];
initial = [-66,0];

[t1, S1] = ode15s(@(t,S)plot_voltage5(t,S),tSpan, initial, options);

figure;
hold on;
plot(t1,S1(:,1));

xlabel('Time(ms)');
ylabel('Volatage(mv)');
title('voltage for a current more than I = 28');
grid on;
            
end

%----------------------------------------------------------------------------------%
%----------------------------------------------------------------------------------%
%% Morris Lecar dynamics equation solver
function dS = plot_voltage(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;

             
tau = 1;
Iext = 5;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));



ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end

%% Morris Lecar dynamics equation solver
function dS = plot_voltage1(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;

            
tau = 1;
Iext=4;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));



ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end

%% Morris Lecar dynamics equation solver
function dS = plot_voltage2(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;

            
tau = 1;
Iext = 5;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));



ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end

%% Morris Lecar dynamics equation solver
function dS = plot_voltage3(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;

             
tau = 1;
Iext=28;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));

  if t>0.4
      Iext = 0;
  end

ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end

%% Morris Lecar dynamics equation solver
function dS = plot_voltage4(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;

            
tau = 1;
Iext=27;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));

  if t>0.4
      Iext = 0;
  end

ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end

%% Morris Lecar dynamics equation solver
function dS = plot_voltage5(t,S)
C = 1 ; 
gCa=20; 
VCa=60; 
gK=10;
VK=-90; 
gL=8;
VL=-80;
Vhalf_n  = -25; k_n = 5;
Vhalf_m  = -20; k_m = 15;


tau = 1;
Iext=29;

%locally define state variables:
V=S(1);
w=S(2);

%local functions:
m_inf = (1./(1+exp((Vhalf_m - V)/k_m)));
w_inf = (1./(1+exp((Vhalf_n - V)/k_n)));

  if t>0.4
      Iext = 0;
  end

ddt_V = (1/C)*(gCa*m_inf*(VCa-V) + gK*w*(VK-V) + gL*(VL-V)+Iext);
ddt_w = (1/tau)*(w_inf-w);

dS=[ddt_V; ddt_w];

end