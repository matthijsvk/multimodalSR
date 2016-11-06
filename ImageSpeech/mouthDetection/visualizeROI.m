close all;

%% This part is for the original MAT file provided with the database
% ROIs contains a cell, that contains a row of matrices. Each matrix
% represents a mouth ROI for one video frame

% mat = ROIs{1,1}{1,1};       
% h = imagesc(mat);
% impixelregion

%% This part is for the processed MAT file, containing only the valid frames

% look for a silence 
nbValid = length(validImages);
validPhonemes2 = string(cellstr(validPhonemes));
disp(class(validPhonemes2))

silences = [];
for i=1:nbValid
   if (validPhonemes2{i} == string('sil'))
       silences(end+1) = i;
   end
end
display(silences)

for i=1:length(silences)
    figure;
    image = validImages{i};
    h = imagesc(image);
    %impixelregion
end


% % tests to see if images correspond
% imagesc(validImages{2});
% figure;
% imagesc(ROIs{1}{19})
% %spy(ROIs{1}{16} - validImages{1})

