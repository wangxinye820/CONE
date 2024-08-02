clearvars -except a b res_all data_num path res_acc res_gmean;

if data_num == 1
    dname = 'Cardiotocography';% 148*18
    dnames = [path dname];
end

if data_num == 2
    dname = 'Cardiotocography1';
    dnames = [path dname];
end
