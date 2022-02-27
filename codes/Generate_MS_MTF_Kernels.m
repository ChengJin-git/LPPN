%  ---------------------------------------------------------------
%  Copyright (c) 2021, Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang,
%  Gemine Vivone, All rights reserved.
% 
%  This work is licensed under GNU Affero General Public License 
%  v3.0 International To view a copy of this license, see the 
%  LICENSE file.
% 
%  This is a script to generate Multispectral (MS) MTF kernels for 
%  the proposed LPPN Network.
%  ---------------------------------------------------------------

% Change here for generating MTF kernels for the corresponding sensor type
sensor = 'WV3'; 

switch sensor
    case 'QB' 
        GNyq = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
    case 'IKONOS'
        GNyq = 0.3 .* ones(1,size(I_MS,3));
    case 'GeoEye1'
        GNyq = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
    case 'WV2'
        GNyq = [0.35 .* ones(1,7), 0.27];
    case 'WV3'    
        GNyq = [0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315];
    case 'none'
        GNyq = 0.3 .* ones(1,4);
end


%%% MTF

N = 7;  % Kernel size
switch sensor
    case 'QB' 
        nBands = 4; % Band Order: B,G,R,NIR
    case 'IKONOS'
        nBands = 4;
    case 'GeoEye1'
        nBands = 4; 
    case 'WV2'
        nBands = 8;
    case 'WV3'    
        nBands = 8;
    case 'none'
        nBands = 4; % Change the corresponding band for custom image source.
end

fcut = 1/4;
   
for ii = 1 : nBands
    alpha = sqrt(((N-1)*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    ms_kernel_raw(:,:,ii) = fwind1(Hd,kaiser(N));
end

ms_kernel = zeros(N,N,nBands,nBands);

for i=1:nBands
    ms_kernel(:,:,i,i) = ms_kernel_raw(:,:,i);
end

disp(strcat('Sensor type: ', sensor))
save(strcat(sensor,'_ms_kernel.mat'), "ms_kernel")
fprintf('Final MS MTF kernels for %d bands are saved.\n', nBands);





