load('banana.mat')
temp_ind = 0;
t_array = 0.1;
s_array = 0.1;
v_array = 0.1;
kernel = Kernel('type','gaussian','gamma',0.1);
theta_value = double(rand(1,size(trainData,1)) > 0.5)';
theta_value = theta_value.*(-1/(v_array*size(trainData,1)));
CONEParmeter = struct('nu',v_array,'tao',t_array,'svalue',s_array, 'theta', theta_value, 'kernelFunc',kernel);
CONE = BaseCONE(CONEParmeter);
CONE.train(trainData, trainLabel);
results = CONE.test(testData,testLabel);
% Visualization
svplot = CONEVisualization();
svplot.boundary(CONE);
svplot.distance(CONE, results);
svplot.testDataWithBoundary(CONE, results);
