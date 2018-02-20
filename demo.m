[filename, pathname] = uigetfile({'*.jpg; *.jpeg; *.gif; *.bmp; *.png'}, 'File Selector');
if ~isempty(pathname) && ~isempty(filename)
    imgpath = strcat(pathname, filename);
end
img = imread(imgpath);  

debug = false;
gauss_sigma = 2.1; % Gaussian Standard Deviation
gauss_window = 3;  % Gaussian Window Size
radius = 30; % Radius of the disc
% Polarity Values -> 0 = light on dark, 1 = dark on light
polarity = 0; 
parzen = 2.1;  % Parzen Standard Deviation
grad_mag_threshold = 0.2; % Gradient Magnitude Threshold
mean_sigmoid = 2; % Mean of the Sigmoid function
sigma_sigmoid = 1; % Standard Deviation of the Sigmoid function



if ndims(img) >= 3
    img = imbinarize(img);
end
I = double(img);

g1 = fspecial('gaussian', gauss_window, gauss_sigma);
I = imfilter(I, g1, 'replicate');

[FX, FY] = gradient(I);
grad_mag = (FX .^ 2 + FY .^ 2) .^ 0.5;

[maxx, maxy] = size(I);
accum = zeros(maxx, maxy);
vote = zeros(maxx, maxy);
for x = 1:maxx
    for y = 1:maxy
        if grad_mag(x,y) > grad_mag_threshold
            theta = atan( FY(x,y)/FX(x,y) );
            xc = int32(x + (-1 ^ polarity) * (radius * sin(theta)));
            yc = int32(y + (-1 ^ polarity) * (radius * cos(theta)));
            if(xc >= 1 && xc <= maxx && yc >= 1 && yc <= maxy)
                vote(x,y) = sigmf(grad_mag(x,y), [sigma_sigmoid mean_sigmoid]);                
                accum(xc,yc) = accum(xc,yc) + vote(x,y);
            end                    
        end
    end
end

screen_size = get(0, 'ScreenSize');
outputAccPos = [uint32((screen_size(3)-screen_size(1)+1)/2) uint32((screen_size(4)-screen_size(2)+1)/4) maxy+100 maxx+100];

figure('Position', outputAccPos);

hold on;
imagesc(accum);
colormap(gray);
axis image;
title 'Accumulator';
hold off;

outputResPos = [uint32((screen_size(3)-screen_size(1)+1)/10) uint32((screen_size(4)-screen_size(2)+1)/4) maxy+100 maxx+100];

prompt = {'Number of Circles? '};             
defaults={'1'};
fields = {'cir'};
info = inputdlg(prompt, 'No of Circles to be found', [1 25], defaults);
if ~isempty(info)              
   info = cell2struct(info,fields);
   cir = str2num(info.cir);
end

lim = 0;

while lim < cir
    
    g2 = fspecial('gaussian', gauss_window, parzen);
    
    accum = imfilter(accum, g2, 'replicate');

    mask_maxima_window = uint32(radius/3);

    [row_val row_ind] = max(accum, [], 1);
    [col_val col_ind] = max(row_val);
    
    x = col_ind;
    y = row_ind(col_ind);               
    for i = x-mask_maxima_window : x+mask_maxima_window
        for j = y-mask_maxima_window : y+mask_maxima_window 
            if (j > 0 && j <= maxx && i > 0 && i < maxy)
                accum(j, i) = 0.0;
            end
        end
    end        

    if lim == 0
        figure('Position', outputResPos);

        imagesc(img);
        colormap(gray);
        axis on;
        axis image;
        title 'Identified Discs in the image';
    end
    
    hold on;
    
    N = 100;

    t=(0:N)*2*pi/N;
    xp=radius*cos(t)+x;
    yp=radius*sin(t)+y;
    p = plot(xp, yp);    
    set(p,'Color','red','LineWidth',2)        
    hold off
    
    lim = lim + 1;
end
