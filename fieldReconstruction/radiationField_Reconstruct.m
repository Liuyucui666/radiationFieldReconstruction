function radiationField_Reconstruct

% 设置参数
particle = "proton";
parallel_enabled = true;  % 是否启用并行计算
gridSpacing = 20;         % 网格间距

% 读取数据
detectorPositions = readmatrix("detectorPosition.txt");
detectorPositions = detectorPositions';

if particle == "proton"
    sourcePosition = readmatrix('source_fast.txt');
    detectorCount = readmatrix('FastdetectorCount.txt');
    responseMatrix = readmatrix("C:\Users\86970\Desktop\EM算法\PROTON_MATRIX_BN_5_sin_flat.txt");
else
    sourcePosition = readmatrix('source_thermal.txt');
    detectorCount = readmatrix('ThermaldetectorCount.txt');
    responseMatrix = readmatrix("C:\Users\86970\Desktop\EM算法\ALPHA_MATRIX_5_sin_flat.txt");
end

% 初始化并行计算
if parallel_enabled && license('test', 'Distrib_Computing_Toolbox')
    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool('local');
        end
        fprintf('并行计算已启用，使用 %d 个工作线程\n', pool.NumWorkers);
    catch
        fprintf('无法启动并行池，使用串行计算\n');
        parallel_enabled = false;
    end
else
    fprintf('并行计算未启用\n');
end

% 分离探测器计数
tic;
detectorIntensities = Detector_Count_Separate(sourcePosition, detectorCount, detectorPositions', responseMatrix, particle);
fprintf('探测器计数分离完成，耗时: %.2f 秒\n', toc);

% 准备重建参数
detectorPositions = detectorPositions(:,1:2);
spaceRange = [0, 6000, 0, 12000]; 
obstacleGrid = generateObstacleGrid(spaceRange, gridSpacing);

% 创建网格
x = spaceRange(1):gridSpacing:spaceRange(2);
y = spaceRange(3):gridSpacing:spaceRange(4);
[X, Y] = meshgrid(x, y);
numGridPoints = numel(X);

% 预分配辐射场
radiationField = zeros(size(X));

% 对每个源点进行重建
for i = 1:size(sourcePosition,1)
    fprintf('重建源点 %d/%d: (%.1f, %.1f)\n', i, size(sourcePosition,1), sourcePosition(i,1), sourcePosition(i,2));
    
    detectorIntensityI = detectorIntensities(:,i);
    sourcePositionI = sourcePosition(i,1:2);
    
    % 重建辐射场
    tic;
    if parallel_enabled
        % 并行版本
        tempField = reconstructRadiationField_parallel(spaceRange, gridSpacing, detectorPositions, detectorIntensityI, sourcePositionI, obstacleGrid);
    else
        % 串行版本
        tempField = reconstructRadiationField_optimized(spaceRange, gridSpacing, detectorPositions, detectorIntensityI, sourcePositionI, obstacleGrid);
    end
    
    fprintf('  重建完成，耗时: %.2f 秒\n', toc);
    
    % 累加到总辐射场
    radiationField = radiationField + tempField;
end

% 可视化结果
visualizeResults(radiationField, spaceRange, gridSpacing);

% 保存结果
save('radiationField.mat', 'radiationField', 'gridSpacing', 'spaceRange');
fprintf('辐射场重建完成并已保存\n');

end


%% 生成屏蔽体标识矩阵函数
function obstacleGrid = generateObstacleGrid(spaceRange, gridSpacing)
    xMin = spaceRange(1); xMax = spaceRange(2);
    yMin = spaceRange(3); yMax = spaceRange(4);
    
    x = xMin:gridSpacing:xMax;
    y = yMin:gridSpacing:yMax;
    [X, Y] = meshgrid(x, y);
    
    % 初始化障碍物网格
    obstacleGrid = zeros(size(X));
    
    % 计算每个网格点到源的距离
    obstacleGrid(X<2000 & Y>9600) = 1;
    obstacleGrid(X>1170 & X<2760 & Y>6650 & Y<8640) = 1;
end


%% 优化后的重建函数（串行版本）
function radiationField = reconstructRadiationField_optimized(spaceRange, gridSize, detectorPositions, detectorIntensities, sourcePosition, obstacleGrid)
    [X, Y] = meshgrid(spaceRange(1):gridSize:spaceRange(2), spaceRange(3):gridSize:spaceRange(4));
    [rows, cols] = size(X);
    radiationField = zeros(rows, cols);
    
    % 预计算一些常量
    numDetectors = size(detectorPositions, 1);
    eps_val = 1e-10;
    
    % ===== 新增：为自适应方法预计算探测器全局信息 =====
    detectorVectors = detectorPositions - sourcePosition;
    detectorDistances = sqrt(sum(detectorVectors.^2, 2));
    avgDetectorDistance = mean(detectorDistances);
    % ============================================
    
    % 向量化计算障碍物判断的网格坐标转换参数
    xMin = spaceRange(1);
    yMin = spaceRange(3);
    gridSpacing = gridSize;
    
    % 计算所有探测器的距离平方（用于交点强度计算）
    detectorDistSq = zeros(numDetectors, 1);
    for k = 1:numDetectors
        detectorDistSq(k) = norm(detectorPositions(k,:) - sourcePosition)^2;
    end
    
    % 主循环
    for i = 1:rows
        for j = 1:cols
            P = [X(i, j), Y(i, j)];
            SP = norm(P - sourcePosition);
            
            % 如果P点太接近源点，设为NaN
            if SP < eps_val
                radiationField(i, j) = NaN;
                continue;
            end
            
            SP_sq = SP^2;
            intersectionPoints = [];
            
            for k = 1:numDetectors
                D = detectorPositions(k, :);
                DI = detectorIntensities(k);
                
                % 快速计算交点
                dir_vec = D - sourcePosition;
                norm_dir = norm(dir_vec);
                
                if norm_dir < eps_val
                    continue;  % 源点和探测器重合，跳过
                end
                
                % 计算交点
                unit_dir = dir_vec / norm_dir;
                intersection = sourcePosition + SP * unit_dir;
                
                % 计算交点强度（使用预计算的距离平方）
                PI = DI * detectorDistSq(k) / SP_sq;
                
                % 检查交点是否有效
                if isPathObstructed_optimized(intersection, D, obstacleGrid, xMin, yMin, gridSpacing)
                    continue;
                end
                
                intersectionPoints = [intersectionPoints; intersection, PI];
            end
            
            % 插值计算 - 替换为自适应混合距离方法
            if ~isempty(intersectionPoints)
                numPoints = size(intersectionPoints, 1);
                
                % ===== 新增：计算角度距离和空间距离 =====
                angularDistances = zeros(numPoints, 1);
                spatialDistances = zeros(numPoints, 1);
                intensities = zeros(numPoints, 1);
                
                for k = 1:numPoints
                    vec1 = P - sourcePosition;
                    vec2 = intersectionPoints(k, 1:2) - sourcePosition;
                    norm1 = norm(vec1);
                    norm2 = norm(vec2);
                    
                    if norm1 < eps_val || norm2 < eps_val
                        cos_angle = 1;
                    else
                        cos_angle = dot(vec1, vec2) / (norm1 * norm2);
                        cos_angle = max(-1, min(1, cos_angle));
                    end
                    
                    angle = acos(cos_angle);
                    angularDistances(k) = SP * angle;
                    intensities(k) = intersectionPoints(k, 3);
                    
                    % 计算空间距离：插值点P到生成此交点的探测器的距离
                    % 注意：intersectionPoints(k)是由detectorPositions(k)生成的
                    spatialDistances(k) = norm(P - detectorPositions(k, :));
                end
                
                % ===== 新增：确定自适应权重参数 =====
                % 计算当前点P到所有探测器的最近距离
                allDistToDetectors = sqrt(sum((detectorPositions - P).^2, 2));
                minDistToDetector = min(allDistToDetectors);
                
                % 自适应策略
                if minDistToDetector < avgDetectorDistance * 0.5
                    alpha = 0.3; % 角度距离权重
                    beta = 0.7;  % 空间距离权重
                elseif SP > avgDetectorDistance * 1.5
                    alpha = 0.8; % 角度距离权重
                    beta = 0.2;  % 空间距离权重
                else
                    alpha = 0.5; % 角度距离权重
                    beta = 0.5;  % 空间距离权重
                end
                
                % ===== 新增：归一化并组合两种距离 =====
                maxAngular = max(angularDistances);
                maxSpatial = max(spatialDistances);
                if maxAngular > 0 && maxSpatial > 0
                    normAngular = angularDistances / maxAngular;
                    normSpatial = spatialDistances / maxSpatial;
                else
                    normAngular = angularDistances;
                    normSpatial = spatialDistances;
                end
                
                % 组合距离 = alpha * 归一化角度距离 + beta * 归一化空间距离
                combinedDistances = alpha * normAngular + beta * normSpatial;
                
                % ===== 修改：基于组合距离选择点 =====
                if numPoints > 5
                    [~, indices] = sort(combinedDistances);
                    selectedIdx = indices(1:5);
                    interdistance = combinedDistances(selectedIdx); % 使用组合距离
                    interIntensity = intensities(selectedIdx);
                else
                    interdistance = combinedDistances; % 使用组合距离
                    interIntensity = intensities;
                end
                
                % 加权平均（使用组合距离的平方反比作为权重）
                zeroIdx = find(interdistance < eps_val, 1);
                if ~isempty(zeroIdx)
                    radiationField(i, j) = interIntensity(zeroIdx);
                else
                    weights = 1 ./ (interdistance.^2 + eps_val);
                    radiationField(i, j) = sum(interIntensity .* weights) / sum(weights);
                end
            else
                radiationField(i, j) = NaN;
            end
        end
    end
end


%% 并行版本重建函数 - 已添加自适应混合距离
function radiationField = reconstructRadiationField_parallel(spaceRange, gridSize, detectorPositions, detectorIntensities, sourcePosition, obstacleGrid)
    [X, Y] = meshgrid(spaceRange(1):gridSize:spaceRange(2), spaceRange(3):gridSize:spaceRange(4));
    [rows, cols] = size(X);
    
    % 预计算一些常量
    numDetectors = size(detectorPositions, 1);
    eps_val = 1e-10;
    xMin = spaceRange(1);
    yMin = spaceRange(3);
    
    % ===== 新增：为自适应方法预计算探测器全局信息 =====
    detectorVectors = detectorPositions - sourcePosition;
    detectorDistances = sqrt(sum(detectorVectors.^2, 2));
    avgDetectorDistance = mean(detectorDistances);
    % ============================================
    
    % 计算所有探测器的距离平方
    detectorDistSq = zeros(numDetectors, 1);
    for k = 1:numDetectors
        detectorDistSq(k) = norm(detectorPositions(k,:) - sourcePosition)^2;
    end
    
    % 创建障碍物查找表（加速障碍物判断）
    [obsRows, obsCols] = size(obstacleGrid);
    obstacleLookup = containers.Map('KeyType', 'uint32', 'ValueType', 'logical');
    for r = 1:obsRows
        for c = 1:obsCols
            if obstacleGrid(r, c) == 1
                key = uint32((r-1) * obsCols + c);
                obstacleLookup(key) = true;
            end
        end
    end
    
    % 并行计算每个网格点
    radiationField = NaN(rows, cols);
    
    parfor i = 1:rows
        rowResult = NaN(1, cols);
        
        for j = 1:cols
            P = [X(i, j), Y(i, j)];
            SP = norm(P - sourcePosition);
            
            if SP < eps_val
                rowResult(j) = NaN;
                continue;
            end
            
            SP_sq = SP^2;
            validIntersections = [];
            
            for k = 1:numDetectors
                D = detectorPositions(k, :);
                DI = detectorIntensities(k);
                
                dir_vec = D - sourcePosition;
                norm_dir = norm(dir_vec);
                
                if norm_dir < eps_val
                    continue;
                end
                
                % 计算交点
                unit_dir = dir_vec / norm_dir;
                intersection = sourcePosition + SP * unit_dir;
                PI = DI * detectorDistSq(k) / SP_sq;
                
                % 检查障碍物
                if ~checkObstacle_parallel(intersection, D, obstacleGrid, xMin, yMin, gridSize, obstacleLookup)
                    validIntersections = [validIntersections; intersection, PI];
                end
            end
            
            % 插值计算 - 替换为自适应混合距离方法
            if ~isempty(validIntersections)
                numPoints = size(validIntersections, 1);
                
                % ===== 新增：计算角度距离和空间距离 =====
                angularDistances = zeros(numPoints, 1);
                spatialDistances = zeros(numPoints, 1);
                intensities = zeros(numPoints, 1);
                
                for k = 1:numPoints
                    vec1 = P - sourcePosition;
                    vec2 = validIntersections(k, 1:2) - sourcePosition;
                    norm1 = norm(vec1);
                    norm2 = norm(vec2);
                    
                    if norm1 < eps_val || norm2 < eps_val
                        cos_angle = 1;
                    else
                        cos_angle = dot(vec1, vec2) / (norm1 * norm2);
                        cos_angle = max(-1, min(1, cos_angle));
                    end
                    
                    angle = acos(cos_angle);
                    angularDistances(k) = SP * angle;
                    intensities(k) = validIntersections(k, 3);
                    
                    % 计算空间距离：插值点P到生成此交点的探测器的距离
                    spatialDistances(k) = norm(P - detectorPositions(k, :));
                end
                
                % ===== 新增：确定自适应权重参数 =====
                % 计算当前点P到所有探测器的最近距离
                % 注意：在parfor中需要重新计算，避免数据依赖
                allDistToDetectors = sqrt(sum((detectorPositions - P).^2, 2));
                minDistToDetector = min(allDistToDetectors);
                
                % 自适应策略
                if minDistToDetector < avgDetectorDistance * 0.5
                    alpha = 0.3; % 角度距离权重
                    beta = 0.7;  % 空间距离权重
                elseif SP > avgDetectorDistance * 1.5
                    alpha = 0.8; % 角度距离权重
                    beta = 0.2;  % 空间距离权重
                else
                    alpha = 0.5; % 角度距离权重
                    beta = 0.5;  % 空间距离权重
                end
                
                % ===== 新增：归一化并组合两种距离 =====
                maxAngular = max(angularDistances);
                maxSpatial = max(spatialDistances);
                if maxAngular > 0 && maxSpatial > 0
                    normAngular = angularDistances / maxAngular;
                    normSpatial = spatialDistances / maxSpatial;
                else
                    normAngular = angularDistances;
                    normSpatial = spatialDistances;
                end
                
                % 组合距离 = alpha * 归一化角度距离 + beta * 归一化空间距离
                combinedDistances = alpha * normAngular + beta * normSpatial;
                
                % ===== 修改：基于组合距离选择点 =====
                if numPoints > 5
                    [~, indices] = sort(combinedDistances);
                    selectedIdx = indices(1:5);
                    interdistance = combinedDistances(selectedIdx); % 使用组合距离
                    interIntensity = intensities(selectedIdx);
                else
                    interdistance = combinedDistances; % 使用组合距离
                    interIntensity = intensities;
                end
                
                % 加权平均（使用组合距离的平方反比作为权重）
                zeroIdx = find(interdistance < eps_val, 1);
                if ~isempty(zeroIdx)
                    rowResult(j) = interIntensity(zeroIdx);
                else
                    weights = 1 ./ (interdistance.^2 + eps_val);
                    rowResult(j) = sum(interIntensity .* weights) / sum(weights);
                end
            else
                rowResult(j) = NaN;
            end
        end
        
        radiationField(i, :) = rowResult;
    end
end


%% 优化的障碍物判断函数
function isObstructed = isPathObstructed_optimized(featurePoint, sourcePosition, obstacleGrid, xMin, yMin, gridSpacing)
    % 快速坐标转换
    srcX = floor((sourcePosition(1) - xMin) / gridSpacing) + 1;
    srcY = floor((sourcePosition(2) - yMin) / gridSpacing) + 1;
    fpX = floor((featurePoint(1) - xMin) / gridSpacing) + 1;
    fpY = floor((featurePoint(2) - yMin) / gridSpacing) + 1;
    
    % 边界检查
    [rows, cols] = size(obstacleGrid);
    if srcX < 1 || srcX > cols || srcY < 1 || srcY > rows || ...
       fpX < 1 || fpX > cols || fpY < 1 || fpY > rows
        isObstructed = false;
        return;
    end
    
    % 快速Bresenham算法
    dx = abs(fpX - srcX);
    dy = abs(fpY - srcY);
    sx = sign(fpX - srcX);
    sy = sign(fpY - srcY);
    
    if dx >= dy
        err = 2*dy - dx;
        x = srcX;
        y = srcY;
        
        for i = 0:dx
            if obstacleGrid(y, x) == 1
                isObstructed = true;
                return;
            end
            
            if err >= 0
                y = y + sy;
                err = err - 2*dx;
            end
            err = err + 2*dy;
            x = x + sx;
        end
    else
        err = 2*dx - dy;
        x = srcX;
        y = srcY;
        
        for i = 0:dy
            if obstacleGrid(y, x) == 1
                isObstructed = true;
                return;
            end
            
            if err >= 0
                x = x + sx;
                err = err - 2*dy;
            end
            err = err + 2*dx;
            y = y + sy;
        end
    end
    
    isObstructed = false;
end


%% 并行版本的障碍物检查函数
function isObstructed = checkObstacle_parallel(featurePoint, sourcePosition, obstacleGrid, xMin, yMin, gridSpacing, obstacleLookup)
    srcX = floor((sourcePosition(1) - xMin) / gridSpacing) + 1;
    srcY = floor((sourcePosition(2) - yMin) / gridSpacing) + 1;
    fpX = floor((featurePoint(1) - xMin) / gridSpacing) + 1;
    fpY = floor((featurePoint(2) - yMin) / gridSpacing) + 1;
    
    [rows, cols] = size(obstacleGrid);
    if srcX < 1 || srcX > cols || srcY < 1 || srcY > rows || ...
       fpX < 1 || fpX > cols || fpY < 1 || fpY > rows
        isObstructed = false;
        return;
    end
    
    % 使用快速Bresenham算法
    dx = abs(fpX - srcX);
    dy = abs(fpY - srcY);
    sx = sign(fpX - srcX);
    sy = sign(fpY - srcY);
    
    if dx >= dy
        err = 2*dy - dx;
        x = srcX;
        y = srcY;
        
        for i = 0:dx
            % 使用查找表快速检查障碍物
            key = uint32((y-1) * cols + x);
            if isKey(obstacleLookup, key)
                isObstructed = true;
                return;
            end
            
            if err >= 0
                y = y + sy;
                err = err - 2*dx;
            end
            err = err + 2*dy;
            x = x + sx;
        end
    else
        err = 2*dx - dy;
        x = srcX;
        y = srcY;
        
        for i = 0:dy
            key = uint32((y-1) * cols + x);
            if isKey(obstacleLookup, key)
                isObstructed = true;
                return;
            end
            
            if err >= 0
                x = x + sx;
                err = err - 2*dy;
            end
            err = err + 2*dx;
            y = y + sy;
        end
    end
    
    isObstructed = false;
end


%% 优化的探测器计数分离函数
function detectorIntensitySeparate = Detector_Count_Separate(sourcePosition, detectorCount, detectorPostion, responseMatrix, particle)
    % 计算探测器总强度
    if particle == "proton"
        detectorIntensity = sum(detectorCount, 1) * 643;
    else 
        detectorIntensity = sum(detectorCount, 1) / (33.1776 * 6) * 501611 - 278666;  % 刻度因子，注量
    end
    
    numDetectors = size(detectorCount, 2);
    numSources = size(sourcePosition, 1);
    detectorIntensitySeparate = zeros(numDetectors, numSources);
    
    % 预计算所有源点到探测器的角度索引
    angleIndices = zeros(numSources, numDetectors);
    
    for j = 1:numSources
        for i = 1:numDetectors
            xSource = sourcePosition(j,1) - detectorPostion(1,i);
            ySource = sourcePosition(j,2) - detectorPostion(2,i);
            zSource = sourcePosition(j,3) - detectorPostion(3,i);
            
            cosPolarSourceDemo = zSource / sqrt(xSource^2 + ySource^2 + zSource^2);
            azimuthSourceDemo = mod(atan2d(ySource, xSource), 360);
            
            cosPolarSource = round(cosPolarSourceDemo / (1/18)) * (1/18);
            azimuthSource = round(azimuthSourceDemo / 5) * 5;
            
            if azimuthSource == 360
                azimuthSource = 0;
            end
            
            angleIndices(j, i) = round(azimuthSource * 7 + (1 - cosPolarSource) * 18 + 1);
        end
    end
    
    % 对每个探测器进行非负最小二乘求解
    for i = 1:numDetectors
        % 构建源计数矩阵
        sourceCount = zeros(6, numSources);
        for j = 1:numSources
            idx = angleIndices(j, i);
            sourceCount(:, j) = responseMatrix(:, idx);
        end
        
        % 非负最小二乘求解
        try
            sourcePercent = lsqnonneg(sourceCount, detectorCount(:, i));
        catch
            % 备用方案：带正则化的最小二乘
            lambda = 1e-5;
            A_reg = [sourceCount; lambda * eye(numSources)];
            b_reg = [detectorCount(:, i); zeros(numSources, 1)];
            sourcePercent = lsqnonneg(A_reg, b_reg);
        end
        
        % 归一化并分配强度
        sourcePercent = sourcePercent' / sum(sourcePercent);
        detectorIntensitySeparate(i, :) = detectorIntensity(i) * sourcePercent;
    end
end

%% 可视化函数
function visualizeResults(radiationField, spaceRange, gridSpacing)
    % 绘制特征点及其强度
    x = spaceRange(1):gridSpacing:spaceRange(2);
    y = spaceRange(3):gridSpacing:spaceRange(4);
    [X, Y] = meshgrid(x, y);
    
    % 创建障碍物掩码
    obstacle_mask = (X<2000 & Y>9600) | (X>1170 & X<2760 & Y>6650 & Y<8640);
    radiationField(obstacle_mask) = NaN;
    alpha_data = ones(size(radiationField));
    alpha_data(obstacle_mask) = 0;  % 障碍物区域完全透明
    
    figure;
    h = imagesc(x, y, radiationField);
    axis xy;
    set(h, 'AlphaData', alpha_data);
    
    % 设置图形背景为白色
    set(gcf, 'Color', 'w');
    set(gca, 'Color', 'w');
    
    colormap(slanCM('turbo'));
    colorbar;
    c = colorbar;
    c.Label.String = 'Radiation intensity (cm^{-2})';  % 设置标题文本
    c.Label.FontSize = 14;               % 设置字体大小
    c.Label.FontName = 'times';          % 设置字体
    c.Label.FontWeight = 'bold';         % 设置字体粗细
    
    % 设置colorbar刻度标签为Times New Roman
    c.FontName = 'times';
    c.FontSize = 12;
    
    set(gca, 'ColorScale', 'log');
    % clim([3e5,5e10]);

    xlabel('X (mm)', 'FontName', 'times', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Y (mm)', 'FontName', 'times', 'FontSize', 14, 'FontWeight', 'bold');
    xticks(0:2000:6000);
    yticks(0:2000:12000);
    
    % 设置坐标轴刻度标签为Times New Roman
    set(gca, 'FontName', 'times', 'FontSize', 12);
    
    axis equal;
    
end