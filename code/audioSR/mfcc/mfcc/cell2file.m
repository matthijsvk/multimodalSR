function cell2file( list, file )
% CELL2FILE Write cell array of strings to file.
%
%   CELL2FILE(LIST,FILE) writes each element of a cell array of strings 
%   (columnwise) as lines to an ASCII file.
%   
%   Example
%           % cell array of strings
%           list = { 'audio1.wav', 'audio2.wav', 'audio3.wav' };
%
%           % output filename
%           file = 'file.txt';
%
%           % write each line of list to file
%           cell2file( list, file );
%
%           % verify contents of the file
%           if isunix, system(sprintf('cat %s',file)); 
%           else, edit(file); end;
%
%   See also FILE2CELL.

%   Author: Kamil Wojcicki, June 2011


    % very lite input validation
    if nargin~=2, error(sprinft('See usage information:\n help %s',mfilename)); end;

    % open an ASCII file for writing, overwrite if exists
    fid = fopen( file, 'w+' );

    % write each element of list (columnwise) to file 
    fprintf( fid, '%s\n', list{:} );

    % clean up
    fclose( fid );


% EOF 
