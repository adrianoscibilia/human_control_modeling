% CREATE IDDATA
% values_input = zeros(400, 7500);
% values_output = zeros(400, 7500);
% 
% for exp_idx=1:400
%     values_input(exp_idx, :) = x{1,exp_idx}.data;
%     values_output(exp_idx, :) = y{1,exp_idx}.data; 
% end
% 
% data = iddata(transpose(values_output),transpose(values_input), 0.008);
% save('err_force_delsecsint_norm_iddata', "data");


load('err_force_del_norm_iddata', "data")
train_end = 6250;
test_steps = 1250;


% ARMAX INIT ONLY
% DoPEM = false;
% opt = armaxOptions;
% opt.Focus = 'simulation';
% opt.SearchOptions.MaxIterations = 100;
% opt.SearchOptions.Tolerance = 1e-5;
% na = 5; 
% nb = 5;
% nc = 5;
% models = cell(1, 400);
% 
% for i = 1:400
%     nk = delayest(data(:, i, i));
%     orders = [na nb nc nk];
%     models{i} = armax(data(1:train_end, i, i), orders, opt);
% end

% save('error_force_delsecsint_norm_models_delayest_pred', "models");

% load("error_force_delsecsint_norm_models_delayest_pred", "models");


% PEM ONLY 
% DoPEM = true;
% sys = cell(1, 400);
% opt = armaxOptions;
% opt.Focus = 'prediction';
% opt.SearchOptions.MaxIterations = 100;
% opt.SearchOptions.Tolerance = 1e-5;
% for i = 1:400
%     sys{i} = pem(data(1:train_end,i,i), models{i}, opt);
% end

% save("error_force_delsecsint_norm_models_delayest_pred_pem2", "sys");
% load("error_force_delsecsint_norm_models_delayest_pred_pem2", "sys");


%ITERATIVE MIXED APPROACH
DoPEM = true;
% opt = armaxOptions;
% opt.Focus = 'simulation';
% opt.SearchOptions.MaxIterations = 100;
% opt.SearchOptions.Tolerance = 1e-5;
% na = 5; 
% nb = 5;
% nc = 5;
% 
% for i = 1:400
%     nk = delayest(data(:, i, i));
%     orders = [na nb nc nk];
%     model = armax(data(1:train_end, i, i), orders, opt);
%     sys = pem(data(1:train_end,i,i), model, opt);
% end
% save("armax_ref_force_iter", "sys");

load("armax_error_force_iter", "sys");

% SIMULATE, FORECAST, PLOT
if DoPEM == true
    armax_model = sys;
else
    armax_model = model;
end
test_end = 7500;

idx = randi(400);
time = linspace(0,(test_end*0.008), test_end);
sim_out = sim(armax_model, data.u(1:test_end, idx));

plot(time, data.y(1:test_end, idx));
hold on;
plot(time, sim_out);
xline(50, '--r');
legend('measured', 'simulated', 'train/test division');
grid on;

% yout = forecast(armax_model, data(1:train_end, idx), test_steps);
% test_truth = data((train_end+1):7500, idx);
% figure;
% compare(test_truth, yout(:,1))
