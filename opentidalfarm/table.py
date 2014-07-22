def tabulate_data(header, rows, alignment=None):
    width = 80

    n_rows = len(rows)
    n_cols = len(rows[0])

    # We must have data to print
    assert len(rows) > 0

    # Check the alignment is of the correct length
    if alignment is not None:
        assert len(alignment) == len(rows[0])
    else:
        alignment = ["c"]*n_cols

    # Create a header
    edge = "|"
    header = edge + header.center(width-2) + edge
    underline = "+" + "-"*(width-2) + "+"
    table = underline + "\n" + header + "\n" + underline +"\n"

    # Format the body
    # Subtract 2 for the sides of the table.
    # Subtract n_cols-1 to divide up the columns.
    col_width = (width-n_cols-3)/n_cols

    def left(string):
        return string.ljust(col_width)

    def center(string):
        return string.center(col_width)

    def right(string):
        return string.rjust(col_width)

    formatter = {"l": left,
                 "c": center,
                 "r": right}

    # We could have less than width characters so calculate a padding to add at
    # the end.
    padding = 0

    # Add 2
    # Add n_cols-1 for the divisions betweens columns.
    true_width = (col_width*n_cols)+n_cols+1
    padding = width - true_width
    lpad = padding/2
    rpad = padding - lpad

    add_space = [1]*n_cols
    add_space[-1] = 0

    for row in rows:
        table += edge + " "*lpad
        for i in range(n_cols):
            table += formatter[alignment[i]](str(row[i])) + " "*add_space[i]
        table += " "*rpad + edge + "\n"

    table += underline
    return table
