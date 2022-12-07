function compare2Runs(workspacePath1, workspacePath2)
run1 = load(workspacePath1);
run2 = load(workspacePath2);

num_of_Ks = length(run1.vecK);
lineColors = ['r', 'g', 'b', 'c', 'k', 'y', 'm'];

figure; 
subplot(2,2,1)
legends = {};
for ii=1:num_of_Ks 
    K = run1.vecK(ii);
    plot(run1.vecTs(ii,:), run1.count_success_DD(ii, :), [lineColors(ii), ':'], 'LineWidth', 2.5)
    legends{3*(ii-1)+1} = ['success DD | K = ' num2str(K)];
    hold on
    plot(run1.vecTs(ii,:), run1.count_success_exact_tot(ii, :), [lineColors(ii), '-'], 'LineWidth', 1.5)
    legends{3*(ii-1)+2} = ['run1 | success Tot | K = ' num2str(K)];
    plot(run2.vecTs(ii,:), run2.count_success_exact_tot(ii, :), [lineColors(ii), '-.'], 'LineWidth', 1.5)
    legends{3*(ii-1)+3} = ['run2 | success Tot | K = ' num2str(K)];
end
legend(legends, 'FontSize', 16);
grid on
ylim([0 100])
xlabel('T')
ylabel('Psuccess')
title({'Compare exact analysis:', [run1.experiment_str ' vs. '], run2.experiment_str}, 'FontSize', 14)


subplot(2,2,2)
legends = {};
for ii=1:num_of_Ks 
    K = run1.vecK(ii);
    plot(run1.vecTs(ii,:), run1.iter_until_detection_tot(ii, :), [lineColors(ii), '-'], 'LineWidth', 1.5)
    hold on
    legends{2*(ii-1)+1} = ['run1 | success Tot | K = ' num2str(K)];
    plot(run2.vecTs(ii,:), run2.iter_until_detection_tot(ii, :), [lineColors(ii), '-.'], 'LineWidth', 1.5)
    legends{2*(ii-1)+2} = ['run2 | success Tot | K = ' num2str(K)];
end
legend(legends, 'FontSize', 16);
grid on
xlabel('T')
title({'Compare #iterations until detection:', [run1.experiment_str ' vs. '], run2.experiment_str}, 'FontSize', 14)


subplot(2,2,3)
legends = {};
for ii=1:num_of_Ks 
    K = run1.vecK(ii);
    plot(run1.vecTs(ii,:), run1.count_success_DD_non_exact(ii, :), [lineColors(ii), ':'], 'LineWidth', 2.5)
    legends{3*(ii-1)+1} = ['success DD | K = ' num2str(K)];
    hold on
    plot(run1.vecTs(ii,:), run1.count_success_non_exact_tot(ii, :), [lineColors(ii), '-'], 'LineWidth', 1.5)
    legends{3*(ii-1)+2} = ['run1 | success Tot | K = ' num2str(K)];
    plot(run2.vecTs(ii,:), run2.count_success_non_exact_tot(ii, :), [lineColors(ii), '-.'], 'LineWidth', 1.5)
    legends{3*(ii-1)+3} = ['run2 | success Tot | K = ' num2str(K)];
end
legend(legends, 'FontSize', 16);
grid on
xlabel('T')
ylabel('Psuccess')
ylim([0 100])
title({'Compare non-exact analysis:', [run1.experiment_str ' vs. '], run2.experiment_str}, 'FontSize', 14)

subplot(2,2,4)
legends = {};
for ii=1:num_of_Ks 
    K = run1.vecK(ii);
    plot(run1.vecTs(ii,:), run1.iter_until_detection_third_step_full(ii, :), [lineColors(ii), '-'], 'LineWidth', 1.5)
    hold on
    legends{2*(ii-1)+1} = ['run1 | success Tot | K = ' num2str(K)];
    plot(run2.vecTs(ii,:), run2.iter_until_detection_third_step_full(ii, :), [lineColors(ii), '-.'], 'LineWidth', 1.5)
    legends{2*(ii-1)+2} = ['run2 | success Tot | K = ' num2str(K)];
end
legend(legends, 'FontSize', 16);
grid on
xlabel('T')
title({'Compare #iterations until detection:', [run1.experiment_str ' vs. '], run2.experiment_str}, 'FontSize', 14)


if 0
    %%
    workspacePath1 = '/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N100_nmc100_methodDD_Normal_perm_factor50__thirdStep_MAP_typical_Tbaseline_ML_25112022_184734/workspace.mat';
    workspacePath2 = '/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N100_nmc100_methodDD_Normal_perm_factor10__thirdStep_MAP_typical_Tbaseline_ML_25112022_181058/workspace.mat';
    compare2Runs(workspacePath1, workspacePath2)
end

