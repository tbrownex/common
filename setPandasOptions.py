import pandas as pd
import numpy as np

def setPandasOptions():
    np.set_printoptions(suppress=True)
    
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
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(op, value)
