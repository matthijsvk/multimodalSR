import sys


# def update_progress(amtDone):
#     sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
#     sys.stdout.flush()

def show_progress(frac_done, bar_length=20):
    # for i in range(end_val):
    hashes = '#' * int(round(frac_done * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rProgress: [{0}] {1}% ".format(hashes + spaces, int(round(frac_done * 100))))
    sys.stdout.flush()


if __name__ == '__main__':
    show_progress(0.8)
