% COMPARE Comparison of selected MFCC feature extraction tools.
%
%   This script compares the cepstral features extracted using the 
%   included MFCC routine, against those extracted using HTK [1] 
%   and MELFCC [2] tools. HTK file format input/output is achieved 
%   using the included simple HTKREAD_LITE and HTKWRITE_LITE 
%   routines. Further functionality can be achieved by installing
%   the VOICEBOX toolbox [3]. 
%
%   This script can be run as is, without HTK [1] and MELFCC [2] tools
%   installed (as these are not included). In such case, features 
%   extracted using the above tools are supplied so that comparisons 
%   can be made. If HTK [1] and/or MELFCC [2] tools are installed, 
%   then their use can be enabled (see the comments in source code). 
%
%   Note that this work has been done and tested on Linux only.
%
%   References
%
%           [1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
%               Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
%               Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
%               Version 3.4.1). Engineering Department, Cambridge University.
%               (see also: http://htk.eng.cam.ac.uk)
%
%           [2] Ellis, D., Reproducing the feature outputs of common programs 
%               using Matlab and melfcc.m. Online resource, url: 
%               http://labrosa.ee.columbia.edu/matlab/rastamat/mfccs.html
%
%           [3] Brookes, M., VOICEBOX: Speech Processing Toolbox for MATLAB.
%               On-line resource, url: 
%               http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%   See also MFCC, EXAMPLE, HTKREAD, HTKREAD_LITE, HTKWRITE, HTKWRITE_LITE.

%   Author: Kamil Wojcicki, September 2011


    %% PRELIMINARIES

    % clean-up MATLAB's environment
    clear all; close all; clc; % fprintf( '.\n' );


    % log everything to file for future reference
    diary( sprintf('%s.txt',mfilename) ); 


    % function handle for mean square error computation 
    MSE = @(x,y)(mean((x(:)-y(:)).^2));


    % function handle for newline display
    newline = @()( fprintf('\n') );


    % note: changing the following _two_ settings
    % will require additional recoding, i.e., 
    % these two fields do not automatically propagate
    % through the simple examples presented here.
    TARGETKIND = 'MFCC_0';  % HTK feature type 
    TYPECODE = 6+8192;      % HTK type code (see Sec. 5.10.1 of [1], pp. 80-81)

    Tw = 25;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 12;                 % number of cepstral coefficients 
    L = 22;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 3700;              % upper frequency limit (Hz)
    wav_file = 'sp10.wav';  % input audio filename

    % get the base part (without extension) of the audio filename
    [ basename, basename ] = fileparts( wav_file );

    % read speech samples, sampling rate and precision from file
    [ speech, fs, nbits ] = wavread( wav_file );


    %% HTK'S [1] HCOPY TOOL

    % specify HTK feature filename
    htk.feature_file = sprintf( '%s_htk.mfc', basename );

    % enable the following if you have HTK tools installed
    % and want to run HCopy feature extraction
    if false

        % generate HTK configuration
        htk.config = {};
        htk.config{end+1} = sprintf( '# MATLAB generated HTK config' );
        htk.config{end+1} = sprintf( 'SOURCEFORMAT = WAV' );
        htk.config{end+1} = sprintf( 'TARGETKIND = %s', TARGETKIND );
        htk.config{end+1} = sprintf( 'WINDOWSIZE = %0.1f', Tw*1E4 ); % in 100 ns units
        htk.config{end+1} = sprintf( 'TARGETRATE = %0.1f', Ts*1E4 ); % in 100 ns units
        htk.config{end+1} = sprintf( 'PREEMCOEF = %0.2f', alpha );
        htk.config{end+1} = sprintf( 'NUMCHANS = %d', M );
        htk.config{end+1} = sprintf( 'CEPLIFTER = %d', L );
        htk.config{end+1} = sprintf( 'NUMCEPS = %d', C );
        htk.config{end+1} = sprintf( 'LOFREQ = %d', LF );
        htk.config{end+1} = sprintf( 'HIFREQ = %d', HF );
        htk.config{end+1} = sprintf( 'USEHAMMING = T' );
        htk.config{end+1} = sprintf( 'SAVEWITHCRC = F');
    
        % specify HTK configuration file
        htk.config_file = 'htkmfcc.conf';

        % write HTK config to file
        cell2file( htk.config, htk.config_file );

        % run HTK's HCopy feature extraction
        eval( sprintf('! HCopy -A -D -V -T 1 -C %s %s %s', ...
                             htk.config_file, wav_file, htk.feature_file) ); % requires HTK [1] (not included)

    end

    % read the HCopy extracted features and rearrange them, since in HTK [1]
    % the 0th cepstral coefficients are stored at the end of each vector
    % where as here the 0th cepstral coefficient comes at the start

    % read HTK features from file
    % htk.mfcc = readhtk( htk.feature_file ); % requires VOICEBOX [3] (not included)
    htk.mfcc = readhtk_lite( htk.feature_file ); % requires 'vanilla' implementation (included)

    % rearrange and make column vectors
    htk.mfcc = htk.mfcc(:, [end 1:end-1]).'; 


    %% MELFCC [2] TOOL BY DAN ELLIS

    % specify MELFCC feature filename
    ellis.feature_file = sprintf( '%s_melfcc.mfc', basename );

    % enable the following if you have Dan Ellis' rastamat tools [2] 
    % installed and want to run MELFCC based feature extraction
    if false
        % extract MFCCs using RASTAMAT tools [2] (not included)
        ellis.mfcc = 0.5*melfcc( speech, fs, 'wintime', Tw*1E-3, ...
            'hoptime', Ts*1E-3, 'preemph', 0.97, 'minfreq', LF, ...
            'maxfreq', HF, 'nbands', M, 'numcep', C+1, 'lifterexp', -L, ...
            'dcttype', 3, 'fbtype', 'htkmel', 'sumpower', 0 );

        % write MELFCC extracted features to file
        % writehtk( ellis.feature_file, ellis.mfcc([2:end 1],:).', Ts*1E-3, TYPECODE ); % requires VOICEBOX [3] (not included)
        writehtk_lite( ellis.feature_file, ellis.mfcc([2:end 1],:).', Ts*1E-3, TYPECODE ); % requires 'vanilla' implementation (included)
    else
        %ellis.mfcc = readhtk( ellis.feature_file ); % requires VOICEBOX [3] (not included)
        ellis.mfcc = readhtk_lite( ellis.feature_file ); % requires 'vanilla' implementation (included)

        % rearrange and make column vectors
        ellis.mfcc = ellis.mfcc(:, [end 1:end-1]).';
    end


    %% MFCC TOOL (THIS IMPLEMENTATION)

    % specify MFCC feature filename
    this.feature_file = sprintf( '%s_mfcc.mfc', basename );

    % extract using the included mfcc(...) function (this implementation)
    this.mfcc = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    % write MFCC extracted features to file
    % writehtk( this.feature_file, this.mfcc([2:end 1],:).', Ts*1E-3, TYPECODE ); % requires VOICEBOX [3] (not included)
    writehtk_lite( this.feature_file, this.mfcc([2:end 1],:).', Ts*1E-3, TYPECODE ); % requries 'vanilla' implementation (included)


    %% HTK'S HLIST COMPARISONS
    
    % enable the following if you have HTK [1] tools installed and want to run HList 
    % as a sanity check for features extracted using different tools compared here
    if false
        Nv = 3; % number of vectors to list

        % run HList feature display for HCopy extracted features
        eval( sprintf('! HList -A -T 1 -o -h -e %i -i %i %s', ...
                        Nv, C+1, htk.feature_file) ); newline(); % requires HTK [1] (not included)

        % run HList feature display for MELFCC extracted features
        eval( sprintf('! HList -A -T 1 -o -h -e %i -i %i %s', ...
                        Nv, C+1, ellis.feature_file) ); newline(); % requires HTK [1] (not included)

        % run HList feature display for MFCC extracted features
        eval( sprintf('! HList -A -T 1 -o -h -e %i -i %i %s', ...
                        Nv, C+1, this.feature_file) ); newline(); % requires HTK [1] (not included)
    end

    
    %% PLOT COMPARISONS

    % time vector (s) for frames or features
    time = [0:size(this.mfcc,2)-1]*Ts*1E-3+0.5*Tw*1E-3; 

    % feature plots for the three considered implementations
    figure('Position', [30 30 800 600], 'PaperPositionMode', 'auto', ... 
              'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

    subplot( 3,1,1 );    
    imagesc( time, [0:M-1], htk.mfcc ); axis( 'xy' );
    xlim( [ min(time) max(time) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Cepstrum index' );
    title( 'Mel frequency cepstrum: HTK' );

    subplot( 3,1,2 );    
    imagesc( time, [0:M-1], ellis.mfcc ); axis( 'xy' );
    xlim( [ min(time) max(time) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Cepstrum index' );
    title( 'Mel frequency cepstrum: MELFCC' );

    subplot( 3,1,3 );    
    imagesc( time, [0:M-1], this.mfcc ); axis( 'xy' );
    xlim( [ min(time) max(time) ] );
    xlabel( 'Time (s)' ); 
    ylabel( 'Cepstrum index' );
    title( 'Mel frequency cepstrum: THIS' );

    print('-dpdf', sprintf('%s.pdf', mfilename)); 
    print('-dpng', sprintf('%s.png', mfilename)); 


    %% MSE COMPARISONS

    % print mean square error values
    fprintf( '\nMFCC MSE(MELFCC,THIS) : %7.2f\n', MSE(ellis.mfcc,this.mfcc) );
    fprintf( 'MFCC MSE(HTK,MELFCC)  : %7.2f\n', MSE(htk.mfcc,ellis.mfcc) );
    fprintf( 'MFCC MSE(HTK,THIS)    : %7.2f\n\n', MSE(htk.mfcc,this.mfcc) );

    % print variance of HTK MFCCs
    fprintf( 'HTK MFCC variance     : %7.2f\n\n', var(htk.mfcc(:)) );


    % flush and close the log file
    diary( 'off' );


% EOF
