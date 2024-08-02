clear all
clc
res_all = [];
addpath('C:\CONE_CODE+DATA\DATA\real_data')
for data_num = 1
    get_dnames_cdata;
    dnames = [dname];
    disp('********* load data **********')
    tot_data = load([dnames '.mat']);
    
    data_only = tot_data.X;
    data_labels = tot_data.y;
    data_label = data_labels;
    
    %% cross fold 5
    count = 0;
    res.dname = [dname];
    normal_data = data_only(data_label==0,:);
    normal_data = cat(2, normal_data, ones(size(normal_data,1),1));
    outlier_data = data_only(data_label==1,:);
    outlier_data = cat(2, outlier_data, ones(size(outlier_data,1),1).*-1);
    
    %%
    v_array = 0.1;
    t_array = 0.1;
    s_array = 0.1;
    kk=1;
    for v = 1:length(v_array)
        for t = 1:length(t_array)
            for s = 1:length(s_array)
                [normal_num,normal_dim] = size(normal_data);
                normal_indices=crossvalind('Kfold',normal_data(1:normal_num,normal_dim),5);
                [outlier_num,outlier_dim] = size(outlier_data);
                outlier_indices=crossvalind('Kfold',outlier_data(1:outlier_num,outlier_dim),5);
                for f = 1:5
                    test_normalind = (normal_indices==f);
                    train_normalind =~ test_normalind;
                    test_outlierind = (outlier_indices==f);
                    train_outlierind =~ test_outlierind;
                    
                    test_normal = normal_data(test_normalind,:);
                    train_normal = normal_data(train_normalind,:);
                    test_outlier = outlier_data(test_outlierind,:);
                    train_outlier = outlier_data(train_outlierind,:);
                    
                    disp('********* train test val **********')
                    train_data = train_normal(:,1:end-1);
                    train_lbls = train_normal(:,end);
                    test_data = cat(1,test_normal(:,1:end-1),test_outlier(:,1:end-1));
                    test_lbls = cat(1,test_normal(:,end),test_outlier(:,end));
                    val_data = train_outlier(:,1:end-1);
                    val_lbls = train_outlier(:,end);
                    
                    disp('********* begin **********')
                    
                    temp_ind = 0;
                    sigm_array = power(2,-3:3);
                    for sigm = 1:length(sigm_array)
                        
                        tic
                        count = count + 1;
                        [data_num f v t sigm s]
                        kernel = Kernel('type','gaussian','gamma',sigm_array(sigm));
                        theta_value = double(rand(1,size(train_data,1)) > 0.5)';
                        theta_value = theta_value.*(-1/(v_array*size(train_data,1)));
                        CONEParmeter = struct('nu',v_array,'tao',t_array(t),'svalue',s_array(s), 'theta', theta_value, 'kernelFunc',kernel);
                        CONE = BaseCONE(CONEParmeter);
                        CONE.train(train_data, train_lbls);
                        traval_data = [train_data;val_data];
                        traval_lbls = [train_lbls; val_lbls];
                        val_results = CONE.test(traval_data, traval_lbls);
                        test_results = CONE.test(test_data, test_lbls);
                        
                        toc
                        temp_ind = temp_ind +1;
                        %%
                        val_acc(temp_ind) = val_results.performance.accuracy; val_auc(temp_ind) = val_results.performance.AUC;
                        val_errorRate(temp_ind) = val_results.performance.errorRate; val_f1(temp_ind) = val_results.performance.F1score;
                        val_gmean(temp_ind) = val_results.performance.gmean; val_aupr(temp_ind) = val_results.performance.aupr;
                        val_sen(temp_ind) = val_results.performance.sensitive; val_spe(temp_ind) = val_results.performance.specificity;
                        val_rec(temp_ind) = val_results.performance.recall; val_pre(temp_ind) = val_results.performance.precision;
                        val_nu(temp_ind) = v_array; val_tao(temp_ind) = t_array(t); val_s(temp_ind) = s_array(s);
                        %%
                        test_acc(temp_ind) = test_results.performance.accuracy; test_auc(temp_ind) = test_results.performance.AUC;
                        test_errorRate(temp_ind) = test_results.performance.errorRate; test_f1(temp_ind) = test_results.performance.F1score;
                        test_gmean(temp_ind) = test_results.performance.gmean; test_aupr(temp_ind) = test_results.performance.aupr;
                        test_sen(temp_ind) = test_results.performance.sensitive; test_spe(temp_ind) = test_results.performance.specificity;
                        test_rec(temp_ind) = test_results.performance.recall; test_pre(temp_ind) = test_results.performance.precision;
                        runningtime(temp_ind) = toc;
                        clear val_results test_results;
                    end
                    [max_val val_ind] = max(val_auc);
                    %[max_test test_ind] = max(test_auc);
                    
                    %%
                    accuracy1(f)= val_acc(val_ind); auc1(f) = val_auc(val_ind);
                    errorrate1(f) = val_errorRate(val_ind); f11 = val_f1(val_ind);
                    aupr1(f) = val_aupr(val_ind);
                    sen1(f) = val_sen(val_ind); spe1(f) = val_spe(val_ind);
                    rec1(f) = val_rec(val_ind); pre1(f) = val_pre(val_ind);
                    nu(f) = val_nu(val_ind); v(f) = val_nu(val_ind);
                    tao(f)= val_tao(val_ind); svalue(f) = val_s(val_ind);
                    %%
                    accuracy2(f)= test_acc(val_ind); auc2(f) = test_auc(val_ind);
                    errorrate2(f) = test_errorRate(val_ind); f12 = test_f1(val_ind);
                    aupr2(f) = test_aupr(val_ind);
                    sen2(f) = test_sen(val_ind); spe2(f) = test_spe(val_ind);
                    rec2(f) = test_rec(val_ind); pre2(f) = test_pre(val_ind);
                    time(f) = runningtime(val_ind);
                    clear test_normalind train_normalind test_outlierind train_outlierind test_normal...
                        train_normal test_outlier train_outlier train_data train_lbls test_data test_lbls
                    
                end
                res.testaucfold = auc2; res.testerrorfold = errorrate2;
                res.f1fold = f12; 
                res.auprfold = aupr2; res.senfold = sen2;
                res.spefold = spe2; res.recfold = rec2;
                res.prefold = pre2; res.timefold = time;
                
                trainacc_mean = mean(accuracy1); trainauc_mean = mean(auc1); trainerrorrate_mean = mean(errorrate1);
                trainf1_mean = mean(f11); trainaupr_mean = mean(aupr1);
                trainsen_mean = mean(sen1); trainspe_mean = mean(spe1); trainrec_mean = mean(rec1); trainpre_mean = mean(pre1); time_mean = mean(time);
                train_result1 = [trainsen_mean trainspe_mean trainrec_mean trainpre_mean trainerrorrate_mean trainacc_mean trainf1_mean trainauc_mean trainaupr_mean time_mean];
                res.trainmean = train_result1;
                
                para = [nu; v; tao; svalue];
                res.parameter = para;
                
                trainacc_std = std(accuracy1); trainauc_std = std(auc1); trainerrorrate_std = std(errorrate1);
                trainf1_std = std(f11); trainaupr_std = std(aupr1);
                trainsen_std = std(sen1); trainspe_std = std(spe1); trainrec_std = std(rec1); trainpre_std = std(pre1); time_std = std(time);
                train_result2 = [trainsen_std trainspe_std trainrec_std trainpre_std trainerrorrate_std trainacc_std trainf1_std trainauc_std trainaupr_std time_std];
                res.trainstd = train_result2;
                
                testacc_mean = mean(accuracy2); testauc_mean = mean(auc2); testerrorrate_mean = mean(errorrate2);
                testf1_mean = mean(f12); testaupr_mean = mean(aupr2);
                testsen_mean = mean(sen2); testspe_mean = mean(spe2); testrec_mean = mean(rec2); testpre_mean = mean(pre2); time_mean = mean(time);
                test_result1 = [testsen_mean testspe_mean testrec_mean testpre_mean testerrorrate_mean testacc_mean testf1_mean testauc_mean testaupr_mean time_mean];
                res.testmean = test_result1;
                
                testacc_std = std(accuracy2); testauc_std = std(auc2); testerrorrate_std = std(errorrate2);
                testf1_std = std(f12); testaupr_std = std(aupr2);
                testsen_std = std(sen2); testspe_std = std(spe2); testrec_std = std(rec2); testpre_std = std(pre2); time_std = std(time);
                test_result2 = [testsen_std testspe_std testrec_std testpre_std testerrorrate_std testacc_std testf1_std testauc_std testaupr_std time_std];
                res.teststd = test_result2;
                
                clear   train_result1 train_result2 test_result1 test_result2;
                res.performance = {'sensitivity','specificity', 'recall', 'precision', 'errorrate', 'accuracy', 'f1',  'auc', 'aupr', 'time'};
                res_all{data_num,kk} = res;
                kk = kk+1;
                save(sprintf('%s%d%s','test',data_num,'.mat'),'res_all')
            end
        end
    end
end
% end
save('cardio.mat','res_all');


