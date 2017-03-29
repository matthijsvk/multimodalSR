def preprocess_dataset(source_path, VERBOSE=False, visualize=False):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    i = 0
    X = []
    Y = []
    fig = []
    num_plot = 4

    for dirName, subdirList, fileList in os.walk(source_path):
        for fname in fileList:
            if not fname.endswith('.PHN') or (fname.startswith("SA")):
                continue

            phn_fname = dirName + '\\' + fname
            wav_fname = dirName + '\\' + fname[0:-4] + '_.WAV'

            total_duration = get_total_duration(phn_fname)
            fr = open(phn_fname)

            if visualize:
                curr_fig = plt.figure(i)
                wav_file = wave.open(wav_fname, 'r')
                signal = wav_file.readframes(-1)
                signal = np.fromstring(signal, 'Int16')
                frame_rate = wav_file.getframerate()

                if wav_file.getnchannels() == 2:
                    print('ONLY MONO FILES')

                x_axis = np.linspace(0, len(signal) / frame_rate, num=len(signal))
                ax1 = plt.subplot(num_plot, 1, 1)
                # plt.title('Original wave data')
                plt.plot(x_axis, signal)
                ax1.set_xlim([0, len(signal) / frame_rate])

                plt.ylabel('Original wave data')
                plt.tick_params(
                        axis='both',  # changes apply to the axis
                        which='both',  # both major and minor ticks are affected
                        bottom='off',  # ticks along the bottom
                        top='off',  # ticks along the top
                        right='off',  # ticks along the right
                        left='off',  # ticks along the left
                        labelbottom='off',  # labels along the bottom
                        labelleft='off')  # labels along the top

            # plt.gca().axes.get_xaxis().set_visible(False)


            X_val, total_frames = create_mfcc('DUMMY', wav_fname)
            total_frames = int(total_frames)

            X.append(X_val)

            y_val = np.zeros(total_frames) - 1
            start_ind = 0
            for line in fr:
                [start_time, end_time, phoneme] = line.rstrip('\n').split()
                start_time = int(start_time)
                end_time = int(end_time)

                phoneme_num = find_phoneme(phoneme)
                end_ind = np.round((end_time) / total_duration * total_frames)
                y_val[start_ind:end_ind] = phoneme_num

                start_ind = end_ind
            fr.close()

            if -1 in y_val:
                print('WARNING: -1 detected in TARGET')
                print(y_val)

            Y.append(y_val.astype('int32'))

            i += 1
            if VERBOSE:
                print()
                print('({}) create_target_vector: {}'.format(i, phn_fname[:-4]))
                print('type(X_val): \t\t {}'.format(type(X_val)))
                print('X_val.shape: \t\t {}'.format(X_val.shape))
                print('type(X_val[0][0]):\t {}'.format(type(X_val[0][0])))
            else:
                print(i, end=' ', flush = True)
                if i >= debug_size and DEBUG:
                    break

            if i >= debug_size and DEBUG:
                break
        print()
        return X, Y, fig
