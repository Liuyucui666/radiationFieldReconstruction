function Probility_Map
% 超参数
particle = "alpha";
xlimx = 0:2:600; ylimy = 0:2:1200; zlimz = [150,150];
[xsequence,ysequence,zsequence] = meshgrid(xlimx, ylimy, zlimz);
detectorPosition = readmatrix("detectorPosition.txt");
detectorRotate = readmatrix("detectorRotation.txt");
angleMatrix = readmatrix("angleMatrix.txt");
if particle == "proton"
    detectorCount = readmatrix("FastdetectorCount.txt");
    load('directionSpectrum_fast.mat', 'directionSpectrum');
    directionSpectrum = directionSpectrum*5;
else
    detectorCount = readmatrix("ThermaldetectorCount.txt");
    load('directionSpectrum_thermal.mat', 'directionSpectrum');
    directionSpectrum = directionSpectrum*3100/225;
end

%%% 求解方向谱
% directionSpectrum = zeros(2522, size(detectorCount,2));
% pass_pecent_SURFACE = waitbar(0,'please wait');  % 加入进度条
% for i = 1:size(detectorCount,2)
%     str=['EM中...',num2str(i/size(detectorCount,2)*100),'%'];  % 进度条
%     waitbar(i/size(detectorCount,2), pass_pecent_SURFACE, str)
%     % directionSpectrum(:,i) = EM(detectorCount(:,i), particle) * sum(detectorCount(:,i));
%     if particle == "proton"
%         directionSpectrum(:,i) = EM(detectorCount(:,i), particle) * sum(detectorCount(:,i)) * 643;   % 快中子实际强度谱
%     else
%         directionSpectrum(:,i) = EM(detectorCount(:,i), particle) * sum(detectorCount(:,i)) * 225;   % 热中子实际强度谱
%     end
% end
% save('directionSpectrum_thermal.mat', 'directionSpectrum');


%%% 方向谱展开，概率图叠加
standardDeviation = directionSpectrum;
detectorAngleProbilityMatrix = zeros(size(detectorCount,2), 2522);
detectorMapProbilityMatrix = zeros([size(xsequence),size(detectorCount,2)]);
% load("MapProbility.mat","MapProbilitySum");
for i = 1:size(standardDeviation,2)
    detectorAngleProbilityMatrix(i, :) = Probility_Convolution(standardDeviation(:,i), angleMatrix, particle);
    detectorMapProbilityMatrix(:,:,:,i) = Detector_Probility_Map(detectorPosition(:,i), detectorRotate(:,i), ...
        xsequence,ysequence,zsequence,detectorAngleProbilityMatrix(i, :));
end
MapProbilitySum = sum(detectorMapProbilityMatrix,4);
MapProbilitySum = (MapProbilitySum-min(MapProbilitySum(:))) / (max(MapProbilitySum(:))-min(MapProbilitySum(:)));
sourceCentroids = Source_Position(MapProbilitySum);
writematrix(sourceCentroids,'source.txt');

%%% 画图
% surf(xsequence(:), ysequence(:),MapProbilitySum(:));
scatter3(xsequence(:), ysequence(:), zsequence(:),10,MapProbilitySum(:),'filled');
colormap("hot");
colorbar;
c = colorbar;
c.Label.String = 'Source probability';  % 设置标题文本
c.Label.FontSize = 12;               % 设置字体大小
c.Label.FontName = 'Arial';          % 设置字体
c.Label.FontWeight = 'bold';         % 设置字体粗细
c.FontName = 'Arial';
c.FontSize = 10;
c.Ticks = 0:0.2:1;  % 设置刻度位置

xlabel('X (cm)', 'FontName', 'Arial', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Y (cm)', 'FontName', 'Arial', 'FontSize', 14, 'FontWeight', 'bold');
zlabel('Z (cm)', 'FontName', 'Arial', 'FontSize', 14, 'FontWeight', 'bold');


% 设置坐标轴刻度标签为Times New Roman
set(gca, 'FontName', 'Arial', 'FontSize', 12);

hold on
scatter3(detectorPosition(1,:), detectorPosition(2,:), detectorPosition(3,:), ...
    20, 'green', 'o', 'LineWidth', 1, 'MarkerFaceColor', 'r');
view(0,90);
axis equal;
xlim([0,600]);
ylim([0,1200]);
xticks(0:200:600);
yticks(0:200:1200);

plot([0, 200, 200, 0, 0], [960, 960, 1200, 1200, 960], ...
    'k--', 'LineWidth', 1, 'Color', [0.05 0.05 0.05]);

% 障碍物2: X>1170 & X<2760 & Y>6650 & Y<8640
plot([117, 276, 276, 117, 117], [665, 665, 864, 864, 665], ...
    'k--', 'LineWidth', 1, 'Color', [0.05 0.05 0.05]);

hold off;

% 输出保存数据
% save('detectorAngleProbilityMatrix.mat', 'detectorAngleProbilityMatrix');
% save('detectorMapProbilityMatrix_fast.mat', 'detectorMapProbilityMatrix');
% xout = xsequence(:,:,1); yout = ysequence(:,:,1); MapProbilitySumout = MapProbilitySum(:,:,1);
% data = [xout(:), yout(:),MapProbilitySumout(:)];
% writematrix(data,'outfororigion');
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 概率卷积函数
function probility = Probility_Convolution(standardDeviationDemo, angleMatrix, particle)
standardDeviation = repmat(standardDeviationDemo, 1, size(angleMatrix,2));
% standardDeviation(standardDeviation>10) = -18.34*log(0.08*log(standardDeviation(standardDeviation>10))); % 80./ standardDeviation;
% standardDeviation(standardDeviation<10) = 100./ sqrt(standardDeviation(standardDeviation<10)); 
if particle == "proton"
    standardDeviation = 5.9 + 360 ./ power(1 + standardDeviation/339, 0.69);
else
    standardDeviation = 2.6 + 360 ./ power(1 + standardDeviation/69, 0.74);
end
probilityMatrix = normpdf(angleMatrix, 0, standardDeviation);
probility = sum(probilityMatrix,1);
probility = probility';
end

% 探测器概率图
function mapProbility = Detector_Probility_Map(detectorPostion, detectorRotate, xlim, ylim, zlim, angleProbility)
xvector_old = xlim - detectorPostion(1);
yvector_old = ylim - detectorPostion(2);
zvector_old = zlim - detectorPostion(3);
[xvector, yvector, zvector] = transformVectorWithEuler(xvector_old, yvector_old, zvector_old, detectorRotate);
% distance = sqrt(xvector.^2 + yvector.^2 + zvector.^2);
% normfactor = exp(-distance);

% 检测零向量位置
normVector = sqrt(xvector.^2 + yvector.^2 + zvector.^2);
zeroVectorMask = (normVector == 0);

mapAzimuth = atan2d(yvector, xvector);
mapAzimuth(zeroVectorMask) = 0;
mapAzimuth(mapAzimuth < 0) = mapAzimuth(mapAzimuth < 0) + 360;
mapAzimuthNumber = round(mapAzimuth/5)*35;
mapAzimuthNumber(mapAzimuthNumber==2520) = 0;

cosmapPolar = zvector./normVector;
cosmapPolar(zeroVectorMask) = 0;
cosmapPolarNumber = round((1-cosmapPolar)*18);
mapNumber = mapAzimuthNumber + cosmapPolarNumber + 1;
mapNumber(cosmapPolarNumber==1) = 1;
mapNumber(cosmapPolarNumber==-1) = 2522;
mapProbility = angleProbility(mapNumber);

meanProb = mean(angleProbility(:));
mapProbility(zeroVectorMask) = meanProb;
end

function [newX, newY, newZ]  = transformVectorWithEuler(X, Y, Z, euler_angles)
    % 将欧拉角从度转换为弧度
    euler_angles = deg2rad(euler_angles);
    z_angle = euler_angles(1);
    x_angle = euler_angles(2);
    y_angle = euler_angles(3);
    
    % 计算绕z轴旋转的矩阵
    Rz = [cos(z_angle) -sin(z_angle) 0;
          sin(z_angle) cos(z_angle) 0;
          0 0 1];
    
    % 计算绕x轴旋转的矩阵
    Rx = [1 0 0;
          0 cos(x_angle) -sin(x_angle);
          0 sin(x_angle) cos(x_angle)];
    
    % 计算绕y轴旋转的矩阵
    Ry = [cos(y_angle) 0 sin(y_angle);
          0 1 0;
          -sin(y_angle) 0 cos(y_angle)];
    
    % 计算总的旋转矩阵，顺序为ZXY
    R_total = Rz * Ry * Rx;  % 坐标变换和坐标系变化不一样（https://zhuanlan.zhihu.com/p/661060377）（https://zhuanlan.zhihu.com/p/183973440#:~:text）
    
    % 将向量坐标转换为列向量
    old_coords = [X(:), Y(:), Z(:)]';
    
    % 计算新坐标系中的向量坐标
    new_coords = R_total' * old_coords; % 之所以要转置是因为实际上是坐标系旋转
    
    % 将新坐标转换回三维矩阵
    newX = reshape(new_coords(1, :), size(X));
    newY = reshape(new_coords(2, :), size(Y));
    newZ = reshape(new_coords(3, :), size(Z));
end


