%% Neuroevolution for Car-vgood dataset
% Author: ChatGPT (MATLAB translation)
clc; clear; close all;

%% ---------------------------
%% Configurable hyperparameters
%% ---------------------------
DATA_PATH = 'car-vgood.dat';
RANDOM_SEED = 1;
POP_SIZE = 120;
N_GENERATIONS = 200;
TOURNAMENT_K = 3;
CROSSOVER_RATE = 0.9;

MUTATE_ARCH_PROB = 0.20;
MUTATE_WEIGHT_PROB = 0.15;
WEIGHT_MUTATION_SCALE = 0.5;

MAX_HIDDEN = 4;
MAX_NEURONS = 200;

ACTIVATIONS = {'relu','tanh','sigmoid'};
USE_F1 = true;

rng(RANDOM_SEED);

%% ---------------------------
%% Load and preprocess dataset
%% ---------------------------
% Read ARFF-like file (skip header until @data)
fid = fopen(DATA_PATH);
line = fgetl(fid);
while ischar(line)
    if startsWith(lower(strtrim(line)),'@data')
        break;
    end
    line = fgetl(fid);
end

data = {};
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line) || startsWith(line,'%')
        continue;
    end
    parts = strsplit(line,',');
    if numel(parts) == 7
        data(end+1,:) = parts; %#ok<SAGROW>
    end
end
fclose(fid);

% Convert to table
cols = {'Buying','Maint','Doors','Persons','Lug_boot','Safety','Class'};
df = cell2table(data,'VariableNames',cols);

% Encode labels
y_raw = strcmp(df.Class,'positive');
y_raw = double(y_raw);

% One-hot encode categorical inputs
X_raw = df(:,1:end-1);
X_enc = [];
for c = 1:width(X_raw)
    [G, categories] = findgroups(X_raw{:,c});
    X_enc = [X_enc dummyvar(G)]; %#ok<AGROW>
end

n_inputs = size(X_enc,2);

% Train/validation/test split
cv = cvpartition(y_raw,'HoldOut',0.25);
X_train_full = X_enc(training(cv),:); y_train_full = y_raw(training(cv));
X_test = X_enc(test(cv),:); y_test = y_raw(test(cv));

cv2 = cvpartition(y_train_full,'HoldOut',0.2);
X_train = X_train_full(training(cv2),:); y_train = y_train_full(training(cv2));
X_val = X_train_full(test(cv2),:); y_val = y_train_full(test(cv2));

%% ---------------------------
%% Neural network helpers
%% ---------------------------
activation_forward = @(z,act) ...
    (strcmp(act,'relu') .* max(0,z) + ...
     strcmp(act,'tanh') .* tanh(z) + ...
     strcmp(act,'sigmoid') .* (1./(1+exp(-z))));

%% ---------------------------
%% Genotype encoding / decoding
%% ---------------------------
% Max weight length
function [max_len, per_layer_len] = compute_max_weight_length(n_inputs,max_hidden,max_neurons)
    prev = n_inputs;
    per_layer_len = zeros(max_hidden+1,1);
    for i=1:max_hidden
        per_layer_len(i) = prev*max_neurons + max_neurons;
        prev = max_neurons;
    end
    % output layer
    per_layer_len(end) = prev*1 + 1;
    max_len = sum(per_layer_len);
end

[MAX_WEIGHT_LEN, per_layer_weight_len] = compute_max_weight_length(n_inputs,MAX_HIDDEN,MAX_NEURONS);

%% Random individual generator
function arch = random_arch_genes(MAX_HIDDEN,MAX_NEURONS)
    arch.num_layers = randi([1,MAX_HIDDEN]);
    arch.neurons = randi([1,MAX_NEURONS],1,MAX_HIDDEN);
    arch.acts = randi([0,2],1,MAX_HIDDEN); % 0:relu,1:tanh,2:sigmoid
end

function ind = random_individual(MAX_WEIGHT_LEN,MAX_HIDDEN,MAX_NEURONS)
    ind.arch = random_arch_genes(MAX_HIDDEN,MAX_NEURONS);
    ind.weights = randn(1,MAX_WEIGHT_LEN)*0.5;
end

%% Decode architecture
function [num_layers, neurons, acts] = decode_arch(arch,ACTIVATIONS)
    num_layers = arch.num_layers;
    neurons = arch.neurons(1:num_layers);
    acts = ACTIVATIONS(arch.acts(1:num_layers)+1);
end

%% Slice weights for architecture
function layers = slice_weights_for_arch(weights,arch,n_inputs,MAX_HIDDEN,MAX_NEURONS)
    [num_layers, neurons, acts] = decode_arch(arch,{'relu','tanh','sigmoid'});
    layers = cell(num_layers+1,1);
    idx = 1;
    in_units = n_inputs;
    for l=1:num_layers
        out_units = neurons(l);
        need = in_units*out_units + out_units;
        seg = weights(idx:idx+need-1);
        W = reshape(seg(1:in_units*out_units),in_units,out_units);
        b = seg(in_units*out_units+1:end);
        layers{l} = {W,b,acts{l}};
        idx = idx + need;
        in_units = out_units;
    end
    % output layer
    out_units = 1;
    need = in_units*out_units + out_units;
    seg = weights(idx:idx+need-1);
    W = reshape(seg(1:in_units*out_units),in_units,out_units);
    b = seg(in_units*out_units+1:end);
    layers{end} = {W,b,'sigmoid'};
end

%% Forward pass
function probs = forward_with_genotype(weights,arch,X_batch,n_inputs,MAX_HIDDEN,MAX_NEURONS)
    layers = slice_weights_for_arch(weights,arch,n_inputs,MAX_HIDDEN,MAX_NEURONS);
    a = X_batch;
    for l=1:length(layers)
        W = layers{l}{1}; b = layers{l}{2}; act = layers{l}{3};
        z = a*W + repmat(b,size(a,1),1);
        if strcmp(act,'relu')
            a = max(0,z);
        elseif strcmp(act,'tanh')
            a = tanh(z);
        elseif strcmp(act,'sigmoid')
            a = 1./(1+exp(-z));
        end
    end
    probs = min(max(a,1e-6),1-1e-6);
end

%% Fitness function
function f = fitness_of_ind(ind,X_val,y_val,n_inputs,MAX_HIDDEN,MAX_NEURONS)
    probs = forward_with_genotype(ind.weights,ind.arch,X_val,n_inputs,MAX_HIDDEN,MAX_NEURONS);
    preds = probs>=0.5;
    tp = sum(preds==1 & y_val==1);
    tn = sum(preds==0 & y_val==0);
    fp = sum(preds==1 & y_val==0);
    fn = sum(preds==0 & y_val==1);
    prec = tp/(tp+fp+eps);
    rec = tp/(tp+fn+eps);
    f = 2*prec*rec/(prec+rec+eps); % F1-score macro not exact but approximate for binary
end

%% ---------------------------
%% GA operators
%% ---------------------------
function parent = tournament_selection(pop,TOURNAMENT_K)
    idxs = randsample(length(pop),TOURNAMENT_K);
    fitnesses = arrayfun(@(i) pop{i}.fitness,idxs);
    [~,bestIdx] = max(fitnesses);
    parent = pop{idxs(bestIdx)};
end

function [child1, child2] = crossover(parent1,parent2,CROSSOVER_RATE,MAX_HIDDEN)
    child1.arch = parent1.arch; child2.arch = parent2.arch;
    if rand<0.5
        slot = randi([1,MAX_HIDDEN]);
        tmp_n = child1.arch.neurons(slot);
        tmp_a = child1.arch.acts(slot);
        child1.arch.neurons(slot) = child2.arch.neurons(slot);
        child2.arch.neurons(slot) = tmp_n;
        child1.arch.acts(slot) = child2.arch.acts(slot);
        child2.arch.acts(slot) = tmp_a;
    end
    if rand<0.3, child1.arch.num_layers = parent2.arch.num_layers; end
    if rand<0.3, child2.arch.num_layers = parent1.arch.num_layers; end

    w1 = parent1.weights; w2 = parent2.weights;
    if rand<CROSSOVER_RATE
        alpha = rand(size(w1));
        c1w = alpha.*w1 + (1-alpha).*w2;
        c2w = alpha.*w2 + (1-alpha).*w1;
    else
        c1w = w1; c2w = w2;
    end
    child1.weights = c1w; child2.weights = c2w;
end

function ind = mutate(ind,MUTATE_ARCH_PROB,MUTATE_WEIGHT_PROB,WEIGHT_MUTATION_SCALE,MAX_HIDDEN,MAX_NEURONS)
    if rand<MUTATE_ARCH_PROB
        if rand<0.5
            ind.arch.num_layers = max(1, ind.arch.num_layers-1);
        else
            ind.arch.num_layers = min(MAX_HIDDEN, ind.arch.num_layers+1);
        end
    end
    if rand<MUTATE_ARCH_PROB
        slot = randi([1,MAX_HIDDEN]);
        if rand<0.6
            delta = randi([-4,4]);
            ind.arch.neurons(slot) = min(MAX_NEURONS,max(1,ind.arch.neurons(slot)+delta));
        else
            ind.arch.acts(slot) = randi([0,2]);
        end
    end
    mask = rand(size(ind.weights))<MUTATE_WEIGHT_PROB;
    ind.weights(mask) = ind.weights(mask) + randn(sum(mask),1)'*WEIGHT_MUTATION_SCALE;
end

%% ---------------------------
%% Initialize population
%% ---------------------------
population = cell(POP_SIZE,1);
for i=1:POP_SIZE
    population{i} = random_individual(MAX_WEIGHT_LEN,MAX_HIDDEN,MAX_NEURONS);
    population{i}.fitness = fitness_of_ind(population{i},X_val,y_val,n_inputs,MAX_HIDDEN,MAX_NEURONS);
end

best_history = zeros(N_GENERATIONS,1);
mean_history = zeros(N_GENERATIONS,1);
best_overall = [];

%% ---------------------------
%% GA main loop
%% ---------------------------
for gen=1:N_GENERATIONS
    % Elitism: keep top 2
    fitness_vals = cellfun(@(x) x.fitness,population);
    [~,idxs] = sort(fitness_vals,'descend');
    new_pop = population(idxs(1:2));

    while length(new_pop)<POP_SIZE
        p1 = tournament_selection(population,TOURNAMENT_K);
        p2 = tournament_selection(population,TOURNAMENT_K);
        [c1,c2] = crossover(p1,p2,CROSSOVER_RATE,MAX_HIDDEN);
        c1 = mutate(c1,MUTATE_ARCH_PROB,MUTATE_WEIGHT_PROB,WEIGHT_MUTATION_SCALE,MAX_HIDDEN,MAX_NEURONS);
        c2 = mutate(c2,MUTATE_ARCH_PROB,MUTATE_WEIGHT_PROB,WEIGHT_MUTATION_SCALE,MAX_HIDDEN,MAX_NEURONS);
        c1.fitness = fitness_of_ind(c1,X_val,y_val,n_inputs,MAX_HIDDEN,MAX_NEURONS);
        c2.fitness = fitness_of_ind(c2,X_val,y_val,n_inputs,MAX_HIDDEN,MAX_NEURONS);
        new_pop{end+1} = c1;
        if length(new_pop)<POP_SIZE
            new_pop{end+1} = c2;
        end
    end

    population = new_pop;

    fitness_vals = cellfun(@(x) x.fitness,population);
    best_history(gen) = max(fitness_vals);
    mean_history(gen) = mean(fitness_vals);

    % Track best overall
    [~,idx] = max(fitness_vals);
    if isempty(best_overall) || population{idx}.fitness>best_overall.fitness
        best_overall = population{idx};
    end

    if mod(gen,10)==0 || gen==1
        fprintf('Gen %3d | Best val fitness: %.4f | Mean: %.4f | Overall best: %.4f\n',...
            gen,max(fitness_vals),mean(fitness_vals),best_overall.fitness);
    end
end

%% ---------------------------
%% Final evaluation on test set
%% ---------------------------
final = best_overall;
fprintf('\nBest validation fitness (final): %.4f\n',final.fitness);
[num_layers, neurons, acts] = decode_arch(final.arch,ACTIVATIONS);
for i=1:num_layers
    fprintf(' Hidden layer %d: neurons=%d, activation=%s\n',i,neurons(i),acts{i});
end
fprintf('Output: 1 neuron (sigmoid)\n');

probs_test = forward_with_genotype(final.weights,final.arch,X_test,n_inputs,MAX_HIDDEN,MAX_NEURONS);
preds_test = probs_test>=0.5;

accuracy = mean(preds_test==y_test);
tp = sum(preds_test==1 & y_test==1);
tn = sum(preds_test==0 & y_test==0);
fp = sum(preds_test==1 & y_test==0);
fn = sum(preds_test==0 & y_test==1);
f1 = 2*tp/(2*tp+fp+fn+eps);

fprintf('Test accuracy: %.4f\n',accuracy);
fprintf('Test F1 (approx macro): %.4f\n',f1);

%% ---------------------------
%% Plot fitness progression
%% ---------------------------
figure;
plot(best_history,'LineWidth',1.5); hold on;
plot(mean_history,'LineWidth',1.5);
xlabel('Generation'); ylabel('Validation Fitness (F1 approx)');
title('GA Neuroevolution Progress');
grid on; legend('Best','Mean');

