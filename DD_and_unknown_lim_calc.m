N = 500;
K = 2:2:500;
T = K .* log2(N);
p = log(2)./K;
exactPD = K + (N-K) .* (1-p.*(1-p).^K).^T;
Ck = 1;
limPD = Ck .* K;

exactDD = K .* (1 - (1 - p .* (1-p) .^ (exactPD-1)) .^ T);
limDD = K .* (1 - 0.5 .^ (0.5*Ck .* log2(N)));

exactUnknown = exactPD - exactDD;

limUnknown = limPD - limDD;
limUnknown2 = exactPD - limDD;


figure;
subplot(411)
plot(K, exactPD, '--.b')
hold on
plot(K, limPD, '--.g')
hold off
legend({'exactPD', 'estPD = C(k)*K'}, 'FontSize', 16)
grid on
xlabel('K', 'FontSize', 16)
ylabel('#PD', 'FontSize', 16)
ylim([min(K), max(K)])
subplot(412)
plot(K, exactDD, '--.b')
hold on
plot(K, limDD, '--.g')
hold off
grid on
legend({'exactDD', 'limDD'}, 'FontSize', 16)
xlabel('K', 'FontSize', 16)
ylabel('#DD', 'FontSize', 16)
ylim([min(K), max(K)])
subplot(413)
plot(K, exactUnknown, '--.b')
hold on
plot(K, limUnknown, '--.g')
plot(K, limUnknown2, '--.r')
hold off
grid on
legend({'exact unknown', 'lim(unknown)1 = limPD-limDD', 'lim(unknown)1 = exactPD-limDD'}, 'FontSize', 16)
xlabel('K', 'FontSize', 16)
ylabel('#unknown', 'FontSize', 16)
ylim([min(K), max(K)])

subplot(414)
plot(K, K-limDD, '--.b')
hold on
plot(K, K,'--k')
hold off
grid on
legend({'K-#DD', 'K'}, 'FontSize', 16)
xlabel('K', 'FontSize', 16)
ylabel('K-#DD', 'FontSize', 16)
ylim([0, max(K)])
%%
Ts = [];
avgUnknowns = [];
numOfTestsFactors = 0.25:0.1:2;
for numOfTestsFactor=numOfTestsFactors
    N = 500;
    K = 2:2:500;
    T = numOfTestsFactor * K .* log2(N);
    Ts = [Ts T];
    
    p = log(2)./K;
    exactPD = K + (N-K) .* (1-p.*(1-p).^K).^T;
    Ck = 1;
    limPD = Ck .* K;

    exactDD = K .* (1 - (1 - p .* (1-p) .^ (exactPD-1)) .^ T);
    limDD = K .* (1 - 0.5 .^ (0.5*Ck .* log2(N)));

    exactUnknown = exactPD - exactDD;

    limUnknown = limPD - limDD;
    limUnknown2 = exactPD - limDD;

    avgUnknown = mean(limUnknown2);
    avgUnknowns = [avgUnknowns avgUnknown];
end
figure;
plot(numOfTestsFactors, avgUnknowns, '--.b')
xlabel('\alpha (T=\alpha\cdotT_{ml})', 'FontSize', 16)
ylabel('avg(#unknown)', 'FontSize', 16)
grid on

%% K estimation using DD algo with Tml
N = 500;
K = 2:2:500;
figure;
legends = {};
% ii = 1;
numOfTestsFactors = [0.5 0.7 0.9 1 1.2 1.5];
for numOfTestsFactor=numOfTestsFactors 
T = numOfTestsFactor * K .* log2(N); 
p = log(2)./K;
exactPD = K + (N-K) .* (1-p.*(1-p).^K).^T;
Ck = 1;
limPD = Ck .* K;

exactDD = K .* (1 - (1 - p .* (1-p) .^ (exactPD-1)) .^ T);

plot(exactDD, K,'--.', 'DisplayName', ['T = ' num2str(numOfTestsFactor) 'T_{ML}'])
hold on
% ii = ii + 1;
end
hold off
grid on
xlabel('#DD', 'FontSize', 16)
ylabel('K', 'FontSize', 16)
title('K estimation using DD algo with T_{ML} - calculation only', 'FontSize', 16)
legend();

%% defective left vs. T
figure;
exactDefectivesLefts = [];
N = 500;
Ks = 5:25:200;
for K=Ks
Tml = K .* log2(N); 
Ts = (0.5:0.1:1.7) .* Tml;
p = log(2)./K;
Ck = 1;
exactDefectivesLeft = (N^(-0.5*Ck)/log2(N)) .* Ts;
exactDefectivesLefts = [exactDefectivesLefts exactDefectivesLeft];
plot(Ts, exactDefectivesLeft , '--.')
hold on
end

hold off
grid on;
xlabel('T')
ylabel('#Defective not detected')