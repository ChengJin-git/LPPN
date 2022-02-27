%  ---------------------------------------------------------------
%  Copyright (c) 2021, Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang,
%  Gemine Vivone, All rights reserved.
% 
%  This work is licensed under GNU Affero General Public License 
%  v3.0 International To view a copy of this license, see the 
%  LICENSE file.
% 
%  This is a script to generate Panchromatic (PAN) MTF kernels for 
%  the proposed LPPN Network.
%  ---------------------------------------------------------------

% Change here for generating MTF kernels for the corresponding sensor type
sensor = 'WV3';

switch sensor
    case 'QB' 
        GNyq = 0.15; 
    case 'IKONOS'
        GNyq = 0.3;
    case 'GeoEye1'
        GNyq = 0.16;
    case 'WV2'
        GNyq = 0.11;
    case 'WV3'
        GNyq = 0.14;
    case 'none'
        GNyq = 0.15;
end

N = 7; % Kernel size
ratio = 4; % Ratio between PAN and MS, default is 4.
fcut = 1/ratio;
 
alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq)));
H = fspecial('gaussian', N, alpha);
Hd = H./max(H(:));
pan_kernel = fwind1(Hd,kaiser(N));

disp(strcat('Sensor type: ', sensor))
save(strcat(sensor,'_pan_kernel.mat'), "pan_kernel")
fprintf('Final PAN MTF kernels are saved.\n');




