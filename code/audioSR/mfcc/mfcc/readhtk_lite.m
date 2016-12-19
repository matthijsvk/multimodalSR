function [ features, sampPeriod, parmKind ] = readhtk_lite( filename )
% READHTK_LITE Simple routine for reading HTK feature files.
%
%   [ FEATURES, SAMPPERIOD, PARMKIND ] = READHTK_LITE( FILENAME )
%   returns FEATURES from HTK [1] feature file specified by FILENAME,
%   along with sample period (s) in SAMPPERIOD and parameter kind
%   in PARAMKIND. Note that this function provides a trivial 
%   implementation with limited functionality. For fully featured 
%   support of HTK I/O refer for example to the VOICEBOX toolbox [2].
%   
%   Inputs
%           FILENAME is a filename as string of a HTK feature file
%
%   Outputs
%           FEATURES is a feature matrix with feature vectors 
%           as rows and feature dimensions as columns
%
%           SAMPPERIOD is a sample period (s)
%
%           PARMKIND is a code indicating a sample kind
%           (see Sec. 5.10.1 of [1], pp. 80-81)
%
%   Example
%           [ features, sampPeriod, parmKind ] = readhtk_lite( 'sp10_htk.mfc' );
%
%   References
%
%           [1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
%               Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
%               Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
%               Version 3.4.1). Engineering Department, Cambridge University.
%               (see also: http://htk.eng.cam.ac.uk)
%
%           [2] VOICEBOX: MATLAB toolbox for speech processing by Mike Brookes
%               url: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

%   Author: Kamil Wojcicki, September 2011
   

    mfcfile = fopen( filename, 'r', 'b' );

    nSamples = fread( mfcfile, 1, 'int32' );
    sampPeriod = fread( mfcfile, 1, 'int32' )*1E-7;
    sampSize = 0.25*fread( mfcfile, 1, 'int16' );
    parmKind = fread( mfcfile, 1, 'int16' );
    
    features = fread( mfcfile, [ sampSize, nSamples ], 'float' ).';

    fclose( mfcfile );


% EOF
