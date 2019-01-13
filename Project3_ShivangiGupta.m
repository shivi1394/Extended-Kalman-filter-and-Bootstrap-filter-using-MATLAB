clc; clear all; format compact;

%% Problem 3 
%   SHIVANGI GUPTA
clc; clear all; format compact;

delta_t = 0.01; % Time Step
mu      = 1.0;  % Van der Pol Parameter

f = @(x) [x(1) + x(2) * delta_t , x(2) + delta_t * (mu * (1 - power(x(1),2)) * x(2) - x(1))];% PUT YOUR PROPAGATION FUNCTION HERE
F = @(x) [1 , delta_t ; -delta_t * (2 * mu * x(1) * x(2) + 1) , 1 + delta_t * (mu * (1 - power(x(1),2)))]; % PUT THE JACOBIAN OF THE PROPAGATION FUNCTION HERE

H = [1 0]; % Measurement Matrix

Q = 0.001 * eye(2); % Propagation Noise
R = 10;             % Measurement Noise

num_samps = 2000;

t_vec = (1:num_samps)*delta_t;

truth = [1 0];
for ii = 2:num_samps
   
    truth(ii, :) = f(truth(ii-1, :));
    
end

measurements = H * truth.' + sqrt(R) * randn(1, num_samps);

%% Problem 4

% Initialization
P = 1 * eye(2);
m = [1 0].';

est_states = [];
est_covars = {};
for ii = 1:num_samps
   
    % Prediction
    m_pred = f(m);
    P_pred = F(m) * P * F(m).' + Q;
    
    % Update
    v = measurements(:, ii) - H * m_pred.';
    S = H * P_pred * H.' + R;
    K = P_pred * H.' * inv(S);
    
    m = m_pred.' + K * v;
    P = P_pred - K * S * K.';
    
    % Storage for analysis
    est_states(ii,:) = m;
    est_covars{ii} = P;
end
 


%% Problem 5 (Bootstrap Filter)

num_samples = 2000;
num_particles = 1000;

particle_states  = repmat([1 0], num_particles, 1);
particle_history = [];

mean_history = [];

for ii = 1:num_samples
   
    for jj = 1:num_particles
       
        particle_states(jj, :) = f(particle_states(jj, :)) + mvnrnd([0 0], Q);
        particle_weights(jj)   = mvnpdf(measurements(ii), H * particle_states(jj, :).', R);     
              
    end
    
    particle_weights = particle_weights / sum(particle_weights);
    
    mean_history(ii, :) = [sum(particle_weights(:) .* particle_states(:, 1)) ...
                           sum(particle_weights(:) .* particle_states(:, 2))];
       
    %neff = 1 / sum(particle_weights.^2);
    
    %if neff < num_particles / 10
        w      = cumsum(particle_weights);    
        newInd = arrayfun(@(x) sum(w < x),  rand(num_particles, 1)) + 1;

        particle_states = particle_states(newInd, :);   
    %end
    
    particle_history(end + 1, :) = particle_states(:, 1);
end

%% Problem 6 (Plot)
figure(1)
plot(truth(:, 1), 'b', 'LineWidth', 2)
hold on
plot(measurements(1, :), 'yx')
plot(est_states(:, 1), 'k', 'LineWidth', 2)
plot(mean_history(:,1),'--r', 'LineWidth', 2)
hold off
legend('Truth', 'Measurements', 'EKF Output', 'Particle Filter')

%% Problem 6 (Mean Error)
Kalman_Error = mean(abs(truth(:,1) - est_states(:,1)));
Particle_Error = mean(abs(truth(:,1) - mean_history(:,1)));
fprintf('Mean Error of Extended Kalman Filter : %f \n',Kalman_Error);
fprintf('Mean Error of Bootstrap Filter Error : %f \n',Particle_Error);





