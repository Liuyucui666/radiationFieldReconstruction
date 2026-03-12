function finalNeutronSpectrum = EM(MeasurementMatrix, particle)
%%%
if particle == "proton"
    filepath_fmatrix = "C:\Users\86970\Desktop\EM算法\PROTON_MATRIX_BN_5_sin_flat.txt";
    lambda = 0.0001; %0.0001; % 正则化参数，可以根据需要调整
else
    filepath_fmatrix = "C:\Users\86970\Desktop\EM算法\ALPHA_MATRIX_5_sin_flat.txt";
    lambda = 0; %0.0001; % 正则化参数，可以根据需要调整
end
loop = 5;

% 结果角度计算（步长为L,结果为N，T=180/L-1）
% 方位角为：(N-2)//T * L: floor((a-2)/35)*5
% 极角为：（(N-2)%T+1） * L:  acosd(1 - mod((a-2),35)*1/18 - 1/18)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ResponseMatrix = importdata(filepath_fmatrix);



[Responserow,Responsecol] = size(ResponseMatrix);  

%%% InitialSpectrum
if loop == 15
    labelnumber = 2:1:265;
    labelazm = floor((labelnumber-2)/11)*15;
    labelpolarcos = 1 - mod((labelnumber-2),11)*1/6 - 1/6;
    labelpolar = acosd(labelpolarcos);
    labelazm = [0,labelazm,0];
    labelpolar = [0,labelpolar,180];
elseif loop == 5
    labelnumber = 2:1:2521;
    labelazm = floor((labelnumber-2)/35)*5;
    labelpolarcos = mod((labelnumber-2),35)*1/18 + 1/18;
    labelpolar = labelpolarcos;
    labelpolar(labelpolarcos>1) = 90 + asind(labelpolarcos(labelpolarcos>1)-1);
    labelpolar(labelpolarcos<=1) = 90 - asind(1-labelpolarcos(labelpolarcos<=1));
    labelazm = [0,labelazm,0];
    labelpolar = [0,labelpolar,180];
end


% 六个面叠加概率
frontangle = [0,90]; backangle = [180,90]; rightangle = [90,90];
leftangle = [270,90]; uponangle = [0,0]; downangle = [0,180];
frontProbility = MeasurementMatrix(1)*guass(AngleDiscrepancy(labelazm,labelpolar,frontangle));
backProbility = MeasurementMatrix(2)*guass(AngleDiscrepancy(labelazm,labelpolar,backangle));
uponProbility = MeasurementMatrix(3)*guass(AngleDiscrepancy(labelazm,labelpolar,uponangle));
downProbility = MeasurementMatrix(4)*guass(AngleDiscrepancy(labelazm,labelpolar,downangle));
rightProbility = MeasurementMatrix(5)*guass(AngleDiscrepancy(labelazm,labelpolar,rightangle));
leftProbility = MeasurementMatrix(6)*guass(AngleDiscrepancy(labelazm,labelpolar,leftangle));
InitialSpectrum = frontProbility + backProbility + uponProbility + downProbility + rightProbility + leftProbility;


NeutronSpectrum(:,1) = InitialSpectrum;

IterationNum = 100;

for i = 1:IterationNum
    for j = 1: Responsecol
        temp1 = NeutronSpectrum(j,i) ./ sum(ResponseMatrix(:,j));
        for k= 1:Responserow
            temp2 = ResponseMatrix(k,j).*MeasurementMatrix(k,1);
            for m = 1:Responsecol
                temp3(m,1) = ResponseMatrix(k,m).*NeutronSpectrum(m,i);
            end
            temp3sum = sum(temp3);
            temp4(k,1) = temp2 ./ temp3sum;
        end
        temp4sum = sum(temp4);
        NeutronSpectrum(j, i + 1) = temp1 * temp4sum;
    end
    NeutronSpectrum(:, i + 1) = NeutronSpectrum(:, i + 1) / sum(NeutronSpectrum(:, i + 1));
    NeutronSpectrum(:, i + 1) = soft_thresholding(NeutronSpectrum(:, i + 1), lambda);
    NeutronSpectrum(:, i + 1) = NeutronSpectrum(:, i + 1) / sum(NeutronSpectrum(:, i + 1));
end

finalNeutronSpectrum = NeutronSpectrum(:,IterationNum);
end


function discre=AngleDiscrepancy(calazm,calpolar,ref)
calx = sind(calpolar).*cosd(calazm);
caly = sind(calpolar).*sind(calazm);
calz = cosd(calpolar);
refx = sind(ref(2))*cosd(ref(1));
refy = sind(ref(2))*sind(ref(1));
refz = cosd(ref(2));

calxyz = [calx;caly;calz];
refxyz = [refx,refy,refz];

discre = acosd(refxyz*calxyz);
end

function g = guass(angle)
mu = 0;
sigma = 60; % 正态分布的误差
g = normpdf(angle,mu,sigma);
end

function x = soft_thresholding(x, lambda)
    % 软阈值函数，用于L1正则化
    x = sign(x) .* max(abs(x) - lambda, 0);
end




