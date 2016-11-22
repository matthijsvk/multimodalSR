close all;

%% This part is for the original MAT file provided with the database
% ROIs contains a cell, that contains a row of matrices. Each matrix
% represents a mouth ROI for one video frame

% mat = ROIs{1,1}{1,1};       
% h = imagesc(mat);
% impixelregion

% % debuggin weird error on si1618
% validImages = {};
% for i=1:length(validFrames)
%     1+validFrames(i);
%     validImages{i} = ROIs{1,1}{1,1+validFrames(i)};
% end
% 
% %% This part is for the processed MAT file, containing only the valid frames
% 
% look for a silence 
% nbValid = length(validImages);
% validPhonemes2 = string(cellstr(validPhonemes));
% disp(class(validPhonemes2))
% 
% images = [];
% for i=1:nbValid
%    if (validPhonemes2{i} == string('sil'))  % s, 
%        images(end+1) = i;
%    end
% end
% display(images)
% 
% for i=1:length(images)
%     figure;
%     imagesc(validImages{images(i)});
% end

% % % tests to see if images correspond
% % imagesc(validImages{end});
figure;
imagesc(ROIs{1,1}{1,4})
%spy(ROIs{1}{16} - validImages{1})
% impixelregion
% display(length(validImages))
