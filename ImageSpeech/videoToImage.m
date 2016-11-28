% VideoReader.getFileFormats
% 
% mov=VideoReader('/mnt/data/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/ImageSpeech/output.avi');
% vidFrames=read(mov);
% nFrames=mov.NumberOfFrames;
% for i=1:nFrames
% %    imshow(vidFrames(:,:,i),[]);  %frames are grayscale
%    imwrite(mov(:,:,:,i),strcat('frames/frame-',num2str(i),'.tif'));
% end

[vid,aud]=mmreader('output.avi');
 numFrames = vid.NumberOfFrames;
 n=numFrames;
 for i = 1:2:n
 frames = read(vid,i);
 imwrite(frames,['Image' int2str(i), '.jpg']);
 im(i)=image(frames);
 end


% Demo macro to extract frames and get frame means from an avi movie
% and save individual frames to separate image files.
% Also computes the mean gray value of the color channels.
clc; % Clear the command window.
close all; % Close all figures (except those of imtool.)
imtool close all; % Close all imtool figures.
clear; % Erase all existing variables.
workspace; % Make sure the workspace panel is showing.
fontSize = 14;

% Change the current folder to the folder of this m-file.
% if(~isdeployed)
%     cd(fileparts(which(mfilename)));
% end

% % Open the rhino.avi demo movie that ships with MATLAB.
% movieFullFileName = '/mnt/data/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/ImageSpeech/output.avi';
% % Check to see that it exists.
% if ~exist(movieFullFileName, 'file')
%     strErrorMessage = sprintf('File not found:\n%s\nYou can choose a new one, or cancel', movieFullFileName);
%     response = questdlg(strErrorMessage, 'File not found', 'OK - choose a  new movie.', 'Cancel', 'OK - choose a new movie.');
%     if strcmpi(response, 'OK - choose a new movie.')
%         [baseFileName, folderName, FilterIndex] = uigetfile('*.avi');
%         if ~isequal(baseFileName, 0)
%             movieFullFileName = fullfile(folderName, baseFileName);
%         else
%             return;
%         end
%     else
%         return;
%     end
% end
% 
% % try
%     mov = VideoReader(movieFullFileName);
%     % movie(mov);
%     % Determine how many frames there are.
%     numberOfFrames = size(mov, 2)
%     numberOfFramesWritten = 0;
%     % Prepare a figure to show the images in the upper half of the screen.
%     figure;
%     screenSize = get(0, 'ScreenSize');
%     newWindowPosition = [1 screenSize(4)/2 - 70 screenSize(3) screenSize(4)/2];
%     set(gcf, 'Position', newWindowPosition); % Maximize figure.
%     %set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
%     
%     % Ask user if they want to write the individual frames out to    disk.
%     promptMessage = sprintf('Do you want to save the individual frames out to individual disk files?');
%     button = questdlg(promptMessage, 'Save individual frames?', 'Yes','No', 'Yes');
%     if strcmp(button, 'Yes')
%         writeToDisk = true;
%         
%         % Extract out the various parts of the filename.
%         [folder, baseFileName, extentions] =  fileparts(movieFullFileName);
%         % Make up a special new output subfolder for all the separate
%         % movie frames that we're going to extract and save to disk.
%         % (Don't worry - windows can handle forward slashes in the        folder name.)
%         folder = pwd; % Make it a subfolder of the folder where this m-file lives.
%         outputFolder = sprintf('%s/Movie Frames from %s', folder,baseFileName)
%         % Create the folder if it doesn't exist already.
%         if ~exist(outputFolder, 'dir')
%             mkdir(outputFolder);
%         end
%     else
%         writeToDisk = false;
%     end
%     
%     % Loop through the movie, writing all frames out.
%     % Each frame will be in a separate file with unique name.
%     meanGrayLevels = zeros(numberOfFrames, 1);
%     meanRedLevels = zeros(numberOfFrames, 1);
%     meanGreenLevels = zeros(numberOfFrames, 1);
%     meanBlueLevels = zeros(numberOfFrames, 1);
%     for frame = 1 : numberOfFrames
%         % Extract the frame from the movie structure.
%         thisFrame = mov(frame).cdata;
%         
%         % Display it
%         hImage = subplot(1,2,1);
%         image(thisFrame);
%         axis square;
%         caption = sprintf('Frame %4d of %d.', frame, numberOfFrames);
%         title(caption, 'FontSize', fontSize);
%         drawnow; % Force it to refresh the window.
%         
%         % Write the image array to the output file, if requested.
%         if writeToDisk
%             % Construct an output image file name.
%             outputBaseFileName = sprintf('Frame %4.4d.png', frame);
%             outputFullFileName = fullfile(outputFolder,outputBaseFileName);
%             
%             % Stamp the name and frame number onto the image.
%             % At this point it's just going into the overlay,
%             % not actually getting written into the pixel values.
%             text(5, 15, outputBaseFileName, 'FontSize', 20);
%             
%             % Extract the image with the text "burned into" it.
%             frameWithText = getframe(gca);
%             % frameWithText.cdata is the image with the text
%             % actually written into the pixel values.
%             % Write it out to disk.
%             imwrite(frameWithText.cdata, outputFullFileName, 'png');
%         end
%         
%         % Calculate the mean gray level.
%         grayImage = rgb2gray(thisFrame);
%         meanGrayLevels(frame) = mean(grayImage(:));
%         
%         % Calculate the mean R, G, and B levels.
%         meanRedLevels(frame) = mean(mean(thisFrame(:, :, 1)));
%         meanGreenLevels(frame) = mean(mean(thisFrame(:, :, 2)));
%         meanBlueLevels(frame) = mean(mean(thisFrame(:, :, 3)));
%         
%         % Plot the mean gray levels.
%         hPlot = subplot(1,2,2);
%         hold off;
%         plot(meanGrayLevels, 'k-', 'LineWidth', 2);
%         hold on;
%         plot(meanRedLevels, 'r-');
%         plot(meanGreenLevels, 'g-');
%         plot(meanBlueLevels, 'b-');
%         
%         % Put title back because plot() erases the existing title.
%         title('Mean Gray Levels', 'FontSize', fontSize);
%         if frame == 1
%             xlabel('Frame Number');
%             yLabel('Gray Level');
%             % Get size data later for preallocation if we read
%             % the movie back in from disk.
%             [rows columns numberOfColorChannels] = size(thisFrame);
%         end
%         
%         % Update user with the progress. Display in the command        window.
%         if writeToDisk
%             progressIndication = sprintf('Wrote frame %4d of %d.', frame,numberOfFrames);
%         else
%             progressIndication = sprintf('Processed frame %4d of %d.',frame, numberOfFrames);
%         end
%         disp(progressIndication);
%         % Increment frame count (should eventually = numberOfFrames
%         % unless an error happens).
%         numberOfFramesWritten = numberOfFramesWritten + 1;
%     end
%     
%     % Alert user that we're done.
%     if writeToDisk
%         finishedMessage = sprintf('Done! It wrote %d frames to folder\n"%s"', numberOfFramesWritten, outputFolder);
%     else
%         finishedMessage = sprintf('Done! It processed %d frames of\n"%s"', numberOfFramesWritten, movieFullFileName);
%     end
%     disp(finishedMessage); % Write to command window.
%     uiwait(msgbox(finishedMessage)); % Also pop up a message box.
%     
%     % Exit if they didn't write any individual frames out to disk.
%     if ~writeToDisk
%         return;
%     end
%     
%     % Ask user if they want to read the individual frames from the
%     disk,
%     % that they just wrote out, back into a movie and display it.
%     promptMessage = sprintf('Do you want to recall the individualframes\nback from disk into a movie?\n(This will take severalseconds.)');
%     button = questdlg(promptMessage, 'Recall Movie?', 'Yes', 'No','Yes');
%     if strcmp(button, 'No')
%         return;
%     end
%     
%     % Read the frames back in, and convert them to a movie.
%     % I don't know of any way to preallocate recalledMovie.
%     for frame = 1 : numberOfFrames
%         % Construct an output image file name.
%         outputBaseFileName = sprintf('Frame %4.4d.png', frame);
%         outputFullFileName = fullfile(outputFolder,outputBaseFileName);
%         % Read the image in from disk.
%         thisFrame = imread(outputFullFileName);
%         % Convert the image into a "movie frame" structure.
%         recalledMovie(frame) = im2frame(thisFrame);
%     end
%     % Get rid of old image and plot.
%     delete(hImage);
%     delete(hPlot);
%     % Create new axes for our movie.
%     subPlot(1, 3, 2);
%     axis off; % Turn off axes numbers.
%     title('Movie recalled from disk', 'FontSize', fontSize);
%     % Play the movie in the axes.
%     movie(recalledMovie);
%     % Note: if you want to display graphics or text in the overlay
%     % as the movie plays back then you need to do it like I did at
%     first
%     % (at the top of this file where you extract and imshow a frame at
%     a time.)
%     msgbox('Done with this demo!');
%     
% % catch ME
% %     % Some error happened if you get here.
% %     stError = lasterror;
% %     strErrorMessage = sprintf('Error extracting movie frames from:\n\n%s\n\nError: %s\n\n)', movieFullFileName, stError.message);
% %     msgbox(strErrorMessage);
% % end


