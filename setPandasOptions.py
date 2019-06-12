import pandas as pd

def setPandasOptions():
    options = {
        'display': {
            'max_columns': 8,
            'max_colwidth': 15,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 10,
            'max_seq_items': 10,         # Max length of printed sequence
            'precision': 3,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(op, value)