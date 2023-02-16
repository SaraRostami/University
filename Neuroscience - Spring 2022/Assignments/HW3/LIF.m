clc;clear all;close all;

V_reset = -0.075;

V_EL = -0.075;                
V_th = -0.055; 

R = 1e8; 

tau_m = 1e-2; 

dt = 0.0001;

T = 0:dt:0.4; 

Vm(1) = V_reset; 

v_mem = zeros(1,50);

for f=10:20:1000
    Im =4e-10 .* sin(2*pi*f*T);
    for t=1:length(T)-1

        if Vm(t) > V_th

            Vm(t+1) = V_reset;

        else

            Vm(t+1) = Vm(t) + dt * ( -(Vm(t) - V_EL) + Im(t) * R) / tau_m;

        end

    end
    v_mem(ceil(f/20)) = max(Vm);
end
f = 10:20:1000;
figure;
plot(f,v_mem)
xlabel('stimulus frequency (Hz)');
ylabel('amplitude of the membrane potential (V)');

%------------------------------------------------------------%

f=10;
Im =4e-10 .* sin(2*pi*f*T);
abs_ref = 0.002;
ref = 0;
Vm(1) = V_reset;
V_trace = [];

for t = 1:length(T)-1
 if ref==0
 Vm(t+1) = Vm(t) + dt * ( -(Vm(t) - V_EL) + Im(t) * R) / tau_m;
 else
 ref = ref - 0.001;
 Vm(t+1) = V_reset;
 end
 if (Vm(t) > V_th)    
 Vm(t) = 0.01;
 ref = abs_ref;
 end
 V_trace = [V_trace Vm(t)];
end
V_trace = [V_th , V_trace];
V_th_vector = zeros(1,length(T));
V_th_vector = V_th_vector + V_th;
figure;
hold on
plot(T,V_trace);
plot(T,V_th_vector);
xlabel('T(s)');
ylabel('Membrane Potential(V)');

%--------------------------------------------------------------

abs_ref = 0.002;
ref = 0;
Vm(1) = V_reset;
V_trace = [];
spike_num=0;
s = zeros(1,100);
for i = 4e-12:4e-12:4e-10
    Vm(1) = V_reset;
    ref=0;
for t = 1:length(T)-1
 if ref==0
 Vm(t+1) = Vm(t) + dt * ( -(Vm(t) - V_EL) + i * R) / tau_m;
 else
 ref = ref - 0.001;
 Vm(t+1) = V_reset;
 end
 if (Vm(t) > V_th)
     spike_num = spike_num+1;
 Vm(t) = 0.01;
 ref = abs_ref;
 end
end
j = i/4e-12;
j= round(j);
s(j) = ceil(spike_num/2);
spike_num=0;
end
i =4e-12:4e-12:4e-10;
figure;
plot(i,s)