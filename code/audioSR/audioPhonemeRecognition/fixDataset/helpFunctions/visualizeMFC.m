% EXAMPLE Simple demo of the MFCC function usage.
%
%   This script is a step by step walk-through of computation of the
%   mel frequency cepstral coefficients (MFCCs) from a speech signal
%   using the MFCC routine.
%
%   See also MFCC, COMPARE.

%   Author: Kamil Wojcicki, September 2011
% see https://nl.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
% installation of toolbox: 
%               Download zip, extract to Matlab bin/MATLAB/R2016b/toolbox
%               Then click 'Home', 'Set Path'; Add the 'toolbox/mfcc' folder; 
%               Click 'Save'.

format short g

    % Clean-up MATLAB's environment
    clear all; close all; clc; 
    
    % Define variables
    Tw = 25;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 3700;              % upper frequency limit (Hz)
    wav_file = 'si650.wav';  % input audio filename
    mfc_file = 'si650.mfc';

    % Read MFC file for comparison (see bottom)
    htkmfc = readhtk(mfc_file);
    
    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread( wav_file );
    info = audioinfo(wav_file);
    nbits = info.BitsPerSample
  
    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = ...
                    mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

     disp(MFCCs)
     disp(frames)
     
    % Generate data needed for plotting 
    [ Nw, NF ] = size( frames );                % frame length and number of frames
    time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames 
    time = [ 0:length(speech)-1 ]/fs;           % time vector (s) for signal samples 
    logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
    logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
    logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range
    
    % Generate plots
    figure('Position', [30 30 800 600], 'PaperPositionMode', 'auto', ... 
              'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

    subplot( 311 );
    plot( time, speech, 'k' );
    xlim( [ min(time_frames) max(time_frames) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Amplitude' ); 
    title( 'Speech waveform'); 

    subplot( 312 );
    imagesc( time_frames, [1:M], logFBEs ); 
    axis( 'xy' );
    xlim( [ min(time_frames) max(time_frames) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Channel index' ); 
    title( 'Log (mel) filterbank energies'); 

    subplot( 313 );
    imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
    %imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
    axis( 'xy' );
    xlim( [ min(time_frames) max(time_frames) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Cepstrum index' );
    title( 'Mel frequency cepstrum' );

    % Set color map to grayscale
    colormap( 1-colormap('gray') ); 

    % Print figure to pdf and png files
    print('-dpdf', sprintf('%s.pdf', mfilename)); 
    print('-dpng', sprintf('%s.png', mfilename)); 
    

    % Calculate HTK-style MFCCs
    mfc = melfcc(speech, fs, 'lifterexp', -22, 'nbands', 20, ...
      'dcttype', 3, 'maxfreq',8000, 'fbtype', 'htkmel', 'sumpower', 0);    
   
    figure; %compare with HTK generated MFCC
    % Reorder and scale to be like mefcc output
    htkmfc = 2*htkmfc(:, [13 [1:12]])';
    % (melfcc.m is 2x HCopy because it deals in power, not magnitude, spectra)
    subplot(311)
    imagesc(htkmfc); axis xy; colorbar
    title('HTK MFCC');
    subplot(312)
    imagesc(mfc); axis xy; colorbar
    title('melfcc MFCC');
    subplot(313)
    imagesc(htkmfc - mfc); axis xy; colorbar
    title('difference HTK - melfcc');
    % Difference occasionally peaks at as much as a few percent (unexplained), 
    % but is basically negligable

% EOF
