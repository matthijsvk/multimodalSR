function writehtk_lite( filename, features, sampPeriod, parmKind )
% WRITEHTK_LITE Simple routine for writing HTK feature files.
%
%   WRITEHTK_LITE( FILENAME, FEATURES, SAMPPERIOD, PARMKIND )
%   writes FEATURES to HTK [1] feature file specified by FILENAME,
%   with sample period (s) defined in SAMPPERIOD and parameter kind
%   in PARAMKIND. Note that this function provides a trivial 
%   implementation with limited functionality. For fully featured 
%   support of HTK I/O refer for example to the VOICEBOX toolbox [2].
%   
%   Inputs
%           FILENAME is a filename as string for a HTK feature file
%
%           FEATURES is a feature matrix with feature vectors 
%           as rows and feature dimensions as columns
%
%           SAMPPERIOD is a sample period (s)
%
%           PARMKIND is a code indicating a sample kind
%           (see Sec. 5.10.1 of [1], pp. 80-81)
%
%   Example
%           % write features to sp10_htk.mfc file with sample period 
%           % set to 10 ms and feature type specified as MFCC_0
%           readhtk_lite( 'sp10_htk.mfc', features, 10E-3, 6+8192 );
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


    mfcfile = fopen( filename, 'w', 'b' );
    [ nSamples, sampSize ] = size( features );
    
    fwrite( mfcfile, nSamples, 'int32' );
    fwrite( mfcfile, sampPeriod*1E7, 'int32' );
    fwrite( mfcfile, 4*sampSize, 'int16' );
    fwrite( mfcfile, parmKind, 'int16' );
    
    count = fwrite( mfcfile, features.', 'float' );
    fclose( mfcfile );

    if count~=nSamples*sampSize
        error( sprintf('write_HTK_file: count!=nSamples*sampSize (%i!=%i), filename: %s', count, nSamples*sampSize, filename)); 
    end


% EOF
