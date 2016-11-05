
% read image 
I = imread('obama.jpg');

% by default, detect faces. Give argument to detect Mouth
MouthDetect = vision.CascadeObjectDetector;%('Mouth','MergeThreshold',100);
BB=step(MouthDetect,I);
figure,
imshow(I); hold on
for i = 1:size(BB,1)

    %rectangle('Position',BB(i,:),'LineWidth',1,'LineStyle','-','EdgeColor','r');

    % increase rectangle size to cover whole face and chin. We assume faces
    % are not upside-down
    increase = 0.2;
    % height
    BB(i,2) = round(BB(i,2) - increase/2*BB(i,3));
    BB(i,4) = round(BB(i,4) * (1+increase));
    
    % width
    %BB(i,1) = round(BB(i,1) - increase/2*BB(i,3));
    %BB(i,3) = round(BB(i,3) * (1+increase));
    
    % remove upper half, we don't need forehead/eyes etc
    BB(i,4) = round(BB(i,4)./2);    
    BB(i,2) = BB(i,2) + BB(i,4)

    % rectangle position: [x y w h]. The x and y elements define the coordinate for the LOWER left corner
    %                                The w and h elements define the dimensions of the rectangle.
 rectangle('Position',BB(i,:),'LineWidth',1,'LineStyle','-','EdgeColor','r');
end
title('Mouth Detection');
hold off;