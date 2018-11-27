import sys
import os
this_file_path = '/'.join(__file__.split('/')[:-1])
full_path = os.path.join(os.getcwd(), this_file_path)
sys.path.append(full_path)
