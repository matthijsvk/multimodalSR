[d,sr] = audioread('sp10.wav');
% Calculate HTK-style MFCCs
 mfc = mfcc(d, sr, 'lifterexp', -22, 'nbands', 20, ...
      'dcttype', 3, 'maxfreq',8000, 'fbtype', 'htkmel', 'sumpower', 0);
 % Load the features from HCopy and compare:
 htkmfc = readhtk('sp10-mfcc.htk');
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
  
 % Invert the HTK features back to waveform, auditory spectrogram, 
 % regular spectrogram (same args as melfcc())
 [dr,aspec,spec] = invmelfcc(htkmfc, sr, 'lifterexp', -22, 'nbands', 20, ...
      'dcttype', 3, 'maxfreq',8000, 'fbtype', 'htkmel', 'sumpower', 0);
 subplot(311)
 imagesc(10*log10(spec)); axis xy; colorbar
 title('Short-time power spectrum inverted from HTK MFCCs')
 subplot(312)
 specgram(dr,512,sr); colorbar
 title('Spectrogram of reconstructed (noise-excited) waveform');
 subplot(313)
 specgram(d,512,sr); colorbar
 title('Original signal spectrogram');
 % Spectrograms look pretty close, although noise-excitation
 % of reconstruction gives it a weird 'whispering crowd' sound