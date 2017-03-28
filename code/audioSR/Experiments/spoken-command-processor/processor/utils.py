

def edit_distance(s1, s2):
    """Calculates the Levenshtein distance between two strings."""
    if s1 == s2: # if equal, then distance is zero
        return 0

    m, n = len(s1), len(s2)

    # if one string is empty, then distance is the length of the other string
    if not s1:
        return n
    elif not s2:
        return m

    # originally matrix of distances: size (m+1) by (n+1)
    # ds[i, j] has dist for first i chars of s1 and first j chars of s2
    # ds = np.zeros((m+1, n+1), dtype=np.int32)

    # optimization: use only two rows (c & d) at a time (working down)
    c = None
    d = range(n+1) # s1 to empty string by j deletions

    for i in range(1, m+1):
        # move current row to previous row
        # create new row, index 0: t to empty string by i deletions, rest 0
        c, d = d, [i]+[0]*n

        # calculate dists for current row
        for j in range(1, n+1):
            sub_cost = int(s1[i-1] != s2[j-1])
            d[j] = min(c[j] + 1,           # deletion
                       d[j-1] + 1,         # insertion
                       c[j-1] + sub_cost)  # substitution

    return d[n]
