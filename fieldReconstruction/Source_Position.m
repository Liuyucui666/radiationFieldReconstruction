function sourceCentroids = Source_Position(sourceProbabilityMatrix)
% 其他
probabilityThreshold = 0.9;

% 插值坐标信息
xsequence = 0:2:600; ysequence = 0:2:1200; zsequence = [1500,1500];
[X, Y, Z] = meshgrid(xsequence, ysequence, zsequence);


% 主函数
sourceCentroids = Calculate_Weighted_Centroids(X, Y, Z, sourceProbabilityMatrix, probabilityThreshold);
end



%%% 
function centroids = Calculate_Weighted_Centroids(X, Y, Z, probabilityMatrix, threshold)
    % 二值化处理
    probabilityMatrix = probabilityMatrix ./ max(max(max(probabilityMatrix)));
    probabilityMatrix(probabilityMatrix < threshold) = 0;
    % probabilityMatrix(probabilityMatrix > threshold) = 1;

    % 使用bwlabel来标记连通区域
    [L, num] = bwlabeln(probabilityMatrix > 0, 26); % 使用26连通性来标记连通区域

    % 初始化质心矩阵
    centroids = zeros(num, 3);

    % 计算每个连通区域的质心
    for i = 1:num
        % 找到属于当前连通区域的点
        index = find(L == i);
        
        % 提取这些点的概率值
        probabilities = probabilityMatrix(index);
        
        % 计算加权质心坐标
        centroidX = sum(X(index) .* probabilityMatrix(index)) / sum(probabilities);
        centroidY = sum(Y(index) .* probabilityMatrix(index)) / sum(probabilities);
        centroidZ = sum(Z(index) .* probabilityMatrix(index)) / sum(probabilities);
        
        % 将质心坐标存储到质心矩阵中
        centroids(i, :) = [centroidX, centroidY, centroidZ];
    end
end

