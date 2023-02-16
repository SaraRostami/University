close all
clear all
clc
%%
% 1.1 part a
load('extracellular.mat');
plot(all_data_with_noise_and_line)
xlabel('time');
ylabel('amplitude');
% 1.1 part b
figure;
histogram(all_data_with_noise_and_line);
% What can be inferred about the background noise by looking at this diagram (distribution, etc.)?
%%
% 1.2 part c
order = 7;
fs = 2400;
fc = 300;
wc = 2*pi*fc;
wn = wc / fs;
% [b,a] = butter(3,wn,'high');
[z,p,k] = butter(7,2*fc/fs,'high');
sos = zp2sos(z,p,k);
g = 1;
y = filtfilt(sos,g,all_data_with_noise_and_line);
% 1.2 part d
figure;
subplot(2,1,1)
plot(all_data_with_noise_and_line)
title('unfiltered data');

subplot(2,1,2)
plot(y)
title('filtered data');
%%
% 1.3 part e
sigma_n = median((abs(y)/0.6745));
tetha = 5 * sigma_n;

% 1.3 part f

count=0;
peak = zeros(1,length(y));
for i = 1:length(y)-2
    if (y(i) < y(i+1)) && (y(i+1) > y(i+2))
           count = count + 1;
            peak(i) = y(i+1);
    end
    if (y(i) > y(i+1)) && (y(i+1) < y(i+2))
        count = count + 1;
        peak(i) = y(i+1);
    end
end

% 1.3 part g
count = 0;
for i = 1:length(peak)
        if abs(peak(i)) >= tetha
            count = count + 1;
        end     
end
index_spike = find(abs(peak) >= tetha);
w = zeros(1,length(y));
for i = 1:length(index_spike)
%        w(index_spike(i)-2:index_spike(i)+2) = y(index_spike(i));
         w(index_spike(i)-2) = y(index_spike(i)-2);
         w(index_spike(i)-1) = y(index_spike(i)-1);
         w(index_spike(i)) = y(index_spike(i));
         w(index_spike(i)+1) = y(index_spike(i)+1);
         w(index_spike(i)+2) = y(index_spike(i)+2);
         for j=1:5
         waveform_matrix(i,j) = w(index_spike(i)+j-3);
         end
end

% 1.3 part h
figure;
plot(w)
title('waveforms for detected spikes');
% Are there any noticeable difference among the overall shapes of these waveforms?
%%
% 1.4 part i
[coeff,score,latent] = pca(waveform_matrix);

% 1.4 part j
% Each column of score corresponds to one principal component. The vector, latent, stores the variances of the four principal components.
PC1 = score(:,1);
PC2 = score(:,2);
PC3 = score(:,3);
%%
% 1.5 part k
score_new = score(:,1:3);
coeff_new = coeff(1:3,1:3);
waveform_matrix_new = score_new * coeff_new';

[idx,C] = kmeans(waveform_matrix_new,3);

% 1.5 part L
figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,2),idx,'bgm');
hold on;
plot(C(:,1),C(:,2),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid');
title('for pair of PC1 and PC2');

figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,3),idx,'bgm');
hold on;
plot(C(:,1),C(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid');
title('for pair of PC1 and PC3');

figure;
gscatter(waveform_matrix_new(:,2),waveform_matrix_new(:,3),idx,'bgm');
hold on;
plot(C(:,2),C(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid');
title('for pair of PC2 and PC3');


% 1.5 part m
% for example k=2

[idx1,C1] = kmeans(waveform_matrix_new,2);

figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,2),idx1,'bg');
hold on;
plot(C1(:,1),C1(:,2),'kx');
legend('Cluster 1','Cluster 2','Cluster Centroid');
title('for pair of PC1 and PC2');

figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,3),idx1,'bg');
hold on;
plot(C1(:,1),C1(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster Centroid');
title('for pair of PC1 and PC3');

figure;
gscatter(waveform_matrix_new(:,2),waveform_matrix_new(:,3),idx1,'bg');
hold on;
plot(C1(:,2),C1(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster Centroid');
title('for pair of PC2 and PC3');

% for example k=4

[idx2,C2] = kmeans(waveform_matrix_new,4);

figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,2),idx2,'bgmr');
hold on;
plot(C2(:,1),C2(:,2),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster Centroid');
title('for pair of PC1 and PC2');

figure;
gscatter(waveform_matrix_new(:,1),waveform_matrix_new(:,3),idx2,'bgmr');
hold on;
plot(C2(:,1),C2(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster Centroid');
title('for pair of PC1 and PC3');

figure;
gscatter(waveform_matrix_new(:,2),waveform_matrix_new(:,3),idx2,'bgmr');
hold on;
plot(C2(:,2),C2(:,3),'kx');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster Centroid');
title('for pair of PC2 and PC3');

% Which value of K gives the best results? Explain why.

%% second section

% 1.5 part n

load('spikes.mat');
% SpikeInds vector's length shows number of spikes. It is 4779. my result (count) is 4801. sounds good!
intersect_of_SpikeInds_and_index_spike = intersect(SpikeInds,index_spike);
w_new_related_to_part_n = zeros(1,length(y));
for i = 1:length(SpikeInds)
         w_new_related_to_part_n(SpikeInds(i)-2) = y(SpikeInds(i)-2);
         w_new_related_to_part_n(SpikeInds(i)-1) = y(SpikeInds(i)-1);
         w_new_related_to_part_n(SpikeInds(i)) = y(SpikeInds(i));
         w_new_related_to_part_n(SpikeInds(i)+1) = y(SpikeInds(i)+1);
         w_new_related_to_part_n(SpikeInds(i)+2) = y(SpikeInds(i)+2);
end

figure;
plot(w_new_related_to_part_n)
title('waveforms for part n spikes');

% 1.5 part o

% 'y' is the filtered data.

tetha_new = 0.9 * max(y);

count_new = 0;
for i = 1:length(peak)
        if abs(peak(i)) >= tetha_new
            count_new = count_new + 1;
        end     
end

index_spike_new = find(abs(peak) >= tetha_new);

w_new = zeros(1,length(y));

for i = 1:length(index_spike_new)
         w_new(index_spike_new(i)-2) = y(index_spike_new(i)-2);
         w_new(index_spike_new(i)-1) = y(index_spike_new(i)-1);
         w_new(index_spike_new(i)) = y(index_spike_new(i));
         w_new(index_spike_new(i)+1) = y(index_spike_new(i)+1);
         w_new(index_spike_new(i)+2) = y(index_spike_new(i)+2);
         for j=1:5
         waveform_matrix_new(i,j) = w_new(index_spike_new(i)+j-3);
         end
end

figure;
plot(w_new)
title('new waveforms for detected spikes');

% Do you think choosing the new threshold (𝜃𝑛𝑒𝑤 ) improved the results of spike-sorting? Justify your answer.

Y = tsne(waveform_matrix_new);
gscatter(Y(:,1),Y(:,2));